"""Inflation nowcasting from scraped price data."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats

# CPI Basket Weights (approximation of BLS weights)
CPI_WEIGHTS = {
    "grocery": 0.143,
    "housing": 0.424,
    "apparel": 0.026,
    "transportation": 0.160,
    "medical": 0.085,
    "recreation": 0.054,
    "education": 0.062,
    "other": 0.046,
}


class NowcastConfig(BaseModel):
    """Configuration for inflation nowcasting."""

    data_path: str = Field(default="./data/processed")
    base_period: str = "2023-01-01"
    seasonal_adjustment: bool = True
    use_weights: bool = True
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    min_observations: int = Field(default=10, ge=1)
    smoothing_window: int = Field(default=7, ge=1)


@dataclass
class NowcastResult:
    """Result of inflation nowcast calculation."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    inflation_rate: float = 0.0
    inflation_rate_yoy: float = 0.0  # Year-over-year
    inflation_rate_mom: float = 0.0  # Month-over-month
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    price_index: float = 100.0
    category_indices: dict[str, float] = field(default_factory=dict)
    category_contributions: dict[str, float] = field(default_factory=dict)
    observation_count: int = 0
    is_seasonally_adjusted: bool = False
    comparison_with_cpi: Optional[dict[str, float]] = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "inflation_rate": self.inflation_rate,
            "inflation_rate_yoy": self.inflation_rate_yoy,
            "inflation_rate_mom": self.inflation_rate_mom,
            "confidence_interval": {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1],
            },
            "price_index": self.price_index,
            "category_indices": self.category_indices,
            "category_contributions": self.category_contributions,
            "observation_count": self.observation_count,
            "is_seasonally_adjusted": self.is_seasonally_adjusted,
            "comparison_with_cpi": self.comparison_with_cpi,
        }


class InflationNowcaster:
    """Computes real-time inflation estimates from scraped price data."""

    def __init__(self, config=None):
        """Initialize the inflation nowcaster."""
        self.config = config or NowcastConfig()
        self.weights = CPI_WEIGHTS
        self._price_data: Optional[pd.DataFrame] = None
        self._base_prices: Optional[pd.DataFrame] = None
        self._seasonal_factors: Optional[dict[str, pd.Series]] = None

    def load_data(self, data=None):
        """Load price data from file or DataFrame."""
        if data is not None:
            self._price_data = data
        else:
            data_path = Path(self.config.data_path)
            latest_file = data_path / "latest.parquet"

            if latest_file.exists():
                self._price_data = pd.read_parquet(latest_file)
            else:
                # Try to load most recent file
                parquet_files = list(data_path.glob("prices_*.parquet"))
                if parquet_files:
                    latest = max(parquet_files, key=lambda p: p.stat().st_mtime)
                    self._price_data = pd.read_parquet(latest)
                else:
                    raise FileNotFoundError(f"No price data found in {data_path}")

        # Ensure timestamp is datetime
        if "timestamp" in self._price_data.columns:
            self._price_data["timestamp"] = pd.to_datetime(
                self._price_data["timestamp"]
            )
            self._price_data["date"] = self._price_data["timestamp"].dt.date

        logger.info(f"Loaded {len(self._price_data)} price records")

    def _calculate_base_prices(self):
        """Calculate base period average prices by category and product."""
        if self._price_data is None:
            raise ValueError("No price data loaded")

        base_date = pd.to_datetime(self.config.base_period)
        base_start = base_date
        base_end = base_date + timedelta(days=30)

        # Filter data to base period
        mask = (self._price_data["timestamp"] >= base_start) & (
            self._price_data["timestamp"] < base_end
        )
        base_data = self._price_data[mask]

        if base_data.empty:
            # Use earliest available data as base
            logger.warning("No data in base period, using earliest available")
            base_data = self._price_data.head(1000)

        # Calculate average price by product
        base_prices = (
            base_data.groupby(["product_id", "category"])["price"]
            .mean()
            .reset_index()
            .rename(columns={"price": "base_price"})
        )

        self._base_prices = base_prices
        return base_prices

    def _calculate_category_index(self, category, current_prices):
        """Calculate Laspeyres price index for a category (base=100)."""
        # FIXME: should handle substitution bias
        if self._base_prices is None:
            self._calculate_base_prices()

        # Get category prices
        category_current = current_prices[current_prices["category"] == category]
        category_base = self._base_prices[self._base_prices["category"] == category]

        if category_current.empty or category_base.empty:
            return 100.0  # No change if no data

        # Merge current with base prices
        merged = category_current.merge(
            category_base[["product_id", "base_price"]],
            on="product_id",
            how="inner",
        )

        if merged.empty or len(merged) < self.config.min_observations:
            return 100.0

        # Calculate price relatives and average
        merged["price_relative"] = merged["price"] / merged["base_price"]

        # Use geometric mean for price index
        index = np.exp(np.log(merged["price_relative"]).mean()) * 100

        return float(index)

    def _apply_seasonal_adjustment(self, series, category):
        """Apply seasonal adjustment (simple decomposition, not X-13ARIMA)."""
        if len(series) < 12:
            return series

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Decompose the series
            decomposition = seasonal_decompose(
                series, model="multiplicative", period=12, extrapolate_trend="freq"
            )

            # Store seasonal factors
            if self._seasonal_factors is None:
                self._seasonal_factors = {}
            self._seasonal_factors[category] = decomposition.seasonal

            # Return seasonally adjusted series
            return series / decomposition.seasonal

        except Exception as e:
            logger.warning(f"Seasonal adjustment failed for {category}: {e}")
            return series

    def _calculate_confidence_interval(self, index_values):
        """Calculate confidence interval for inflation estimate."""
        if len(index_values) < 2:
            return (0.0, 0.0)

        values = np.array(index_values)
        mean = np.mean(values)
        sem = stats.sem(values)

        # Calculate confidence interval
        ci = stats.t.interval(
            self.config.confidence_level,
            len(values) - 1,
            loc=mean,
            scale=sem,
        )

        return (float(ci[0]), float(ci[1]))

    def compute_nowcast(self, as_of_date=None):
        """Compute the current inflation nowcast."""
        if self._price_data is None:
            self.load_data()

        if self._price_data is None or self._price_data.empty:
            raise ValueError("No price data available")

        as_of_date = as_of_date or datetime.utcnow()
        result = NowcastResult(timestamp=as_of_date)

        # Get recent prices (last 7 days)
        recent_start = as_of_date - timedelta(days=self.config.smoothing_window)
        recent_mask = (self._price_data["timestamp"] >= recent_start) & (
            self._price_data["timestamp"] <= as_of_date
        )
        recent_prices = self._price_data[recent_mask]

        if recent_prices.empty:
            logger.warning("No recent price data available")
            return result

        # Calculate base prices if not done
        if self._base_prices is None:
            self._calculate_base_prices()

        # Calculate index for each category
        category_indices = {}
        category_contributions = {}
        index_values = []

        for category, weight in self.weights.items():
            category_index = self._calculate_category_index(category, recent_prices)
            category_indices[category] = category_index

            # Calculate contribution to overall index
            contribution = (category_index - 100) * weight
            category_contributions[category] = contribution

            if self.config.use_weights:
                index_values.append(category_index * weight)
            else:
                index_values.append(category_index)

        # Calculate overall price index
        if self.config.use_weights:
            overall_index = sum(index_values)
        else:
            overall_index = np.mean(index_values)

        result.price_index = overall_index
        result.category_indices = category_indices
        result.category_contributions = category_contributions

        # Calculate inflation rates
        result.inflation_rate = overall_index - 100  # Simple percent change from base
        result.inflation_rate_yoy = self._calculate_yoy_inflation(
            as_of_date, overall_index
        )
        result.inflation_rate_mom = self._calculate_mom_inflation(
            as_of_date, overall_index
        )

        # Calculate confidence interval
        result.confidence_interval = self._calculate_confidence_interval(
            list(category_indices.values())
        )

        # Get observation count
        result.observation_count = len(recent_prices)
        result.is_seasonally_adjusted = self.config.seasonal_adjustment

        logger.info(
            f"Nowcast computed: inflation={result.inflation_rate:.2f}%, "
            f"index={result.price_index:.2f}, n={result.observation_count}"
        )

        return result

    def _calculate_yoy_inflation(self, current_date, current_index):
        """Calculate year-over-year inflation rate."""
        if self._price_data is None:
            return 0.0

        # Get prices from one year ago
        year_ago_start = current_date - timedelta(
            days=365 + self.config.smoothing_window
        )
        year_ago_end = current_date - timedelta(days=365)

        mask = (self._price_data["timestamp"] >= year_ago_start) & (
            self._price_data["timestamp"] <= year_ago_end
        )
        year_ago_prices = self._price_data[mask]

        if year_ago_prices.empty:
            return 0.0

        # Calculate index for year ago
        year_ago_index = 0.0
        for category, weight in self.weights.items():
            category_index = self._calculate_category_index(category, year_ago_prices)
            year_ago_index += category_index * weight

        if year_ago_index <= 0:
            return 0.0

        return ((current_index / year_ago_index) - 1) * 100

    def _calculate_mom_inflation(self, current_date, current_index):
        """Calculate month-over-month inflation rate."""
        if self._price_data is None:
            return 0.0

        # Get prices from one month ago
        month_ago_start = current_date - timedelta(
            days=30 + self.config.smoothing_window
        )
        month_ago_end = current_date - timedelta(days=30)

        mask = (self._price_data["timestamp"] >= month_ago_start) & (
            self._price_data["timestamp"] <= month_ago_end
        )
        month_ago_prices = self._price_data[mask]

        if month_ago_prices.empty:
            return 0.0

        # Calculate index for month ago
        month_ago_index = 0.0
        for category, weight in self.weights.items():
            category_index = self._calculate_category_index(category, month_ago_prices)
            month_ago_index += category_index * weight

        if month_ago_index <= 0:
            return 0.0

        return ((current_index / month_ago_index) - 1) * 100

    def compare_with_cpi(self, official_cpi):
        """Compare nowcast with official CPI data."""
        if self._price_data is None:
            self.load_data()

        comparison = {
            "correlation": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "bias": 0.0,
            "tracking_error": 0.0,
            "lead_lag_correlation": {},
        }

        # Align dates
        official_cpi["date"] = pd.to_datetime(official_cpi["date"]).dt.date

        nowcast_values = []
        cpi_values = []

        for _, row in official_cpi.iterrows():
            date = pd.to_datetime(row["date"])
            try:
                nowcast = self.compute_nowcast(as_of_date=date)
                nowcast_values.append(nowcast.price_index)
                cpi_values.append(row["cpi_value"])
            except Exception as e:
                logger.warning(f"Error computing nowcast for {date}: {e}")

        if len(nowcast_values) < 2:
            return comparison

        nowcast_arr = np.array(nowcast_values)
        cpi_arr = np.array(cpi_values)

        # Calculate comparison metrics
        comparison["correlation"] = float(np.corrcoef(nowcast_arr, cpi_arr)[0, 1])
        comparison["rmse"] = float(np.sqrt(np.mean((nowcast_arr - cpi_arr) ** 2)))
        comparison["mae"] = float(np.mean(np.abs(nowcast_arr - cpi_arr)))
        comparison["bias"] = float(np.mean(nowcast_arr - cpi_arr))
        comparison["tracking_error"] = float(np.std(nowcast_arr - cpi_arr))

        # Calculate lead-lag correlations
        for lag in range(-4, 5):
            if lag < 0:
                corr = np.corrcoef(nowcast_arr[:lag], cpi_arr[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(nowcast_arr[lag:], cpi_arr[:-lag])[0, 1]
            else:
                corr = np.corrcoef(nowcast_arr, cpi_arr)[0, 1]
            comparison["lead_lag_correlation"][f"lag_{lag}"] = float(corr)

        logger.info(
            f"CPI comparison: correlation={comparison['correlation']:.3f}, "
            f"RMSE={comparison['rmse']:.3f}"
        )

        return comparison

    def get_category_breakdown(self, as_of_date=None):
        """Get detailed breakdown of inflation by category."""
        nowcast = self.compute_nowcast(as_of_date)

        data = []
        for category in self.weights.keys():
            data.append(
                {
                    "category": category,
                    "weight": self.weights[category],
                    "price_index": nowcast.category_indices.get(category, 100.0),
                    "inflation_rate": nowcast.category_indices.get(category, 100.0)
                    - 100,
                    "contribution": nowcast.category_contributions.get(category, 0.0),
                }
            )

        return pd.DataFrame(data)

    def get_historical_nowcasts(self, start_date, end_date, frequency="D"):
        """Generate historical nowcasts for backtesting."""
        dates = pd.date_range(start_date, end_date, freq=frequency)
        results = []

        for date in dates:
            try:
                nowcast = self.compute_nowcast(as_of_date=date)
                results.append(
                    {
                        "date": date,
                        "price_index": nowcast.price_index,
                        "inflation_rate": nowcast.inflation_rate,
                        "inflation_yoy": nowcast.inflation_rate_yoy,
                        "inflation_mom": nowcast.inflation_rate_mom,
                        "observation_count": nowcast.observation_count,
                    }
                )
            except Exception as e:
                logger.warning(f"Error computing nowcast for {date}: {e}")

        return pd.DataFrame(results)


def main():
    """Main entry point for running nowcast."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute inflation nowcast")
    parser.add_argument(
        "--data-path",
        "-d",
        default="./data/processed",
        help="Path to processed price data",
    )
    parser.add_argument(
        "--date",
        "-t",
        default=None,
        help="Date for nowcast (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (JSON)",
    )

    args = parser.parse_args()

    config = NowcastConfig(data_path=args.data_path)
    nowcaster = InflationNowcaster(config)

    as_of_date = None
    if args.date:
        as_of_date = datetime.strptime(args.date, "%Y-%m-%d")

    result = nowcaster.compute_nowcast(as_of_date=as_of_date)

    print("\n=== Inflation Nowcast ===")
    print(f"Timestamp: {result.timestamp}")
    print(f"Price Index: {result.price_index:.2f}")
    print(f"Inflation Rate: {result.inflation_rate:.2f}%")
    print(f"YoY Inflation: {result.inflation_rate_yoy:.2f}%")
    print(f"MoM Inflation: {result.inflation_rate_mom:.2f}%")
    print(
        f"Confidence Interval: ({result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f})"
    )
    print(f"Observations: {result.observation_count}")

    print("\n=== Category Breakdown ===")
    for category, index in result.category_indices.items():
        contribution = result.category_contributions.get(category, 0.0)
        print(f"  {category}: index={index:.2f}, contribution={contribution:.2f}pp")

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

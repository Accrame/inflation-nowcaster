"""Time series forecasting for inflation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats


class ForecastConfig(BaseModel):
    """Configuration for inflation forecasting."""

    model_type: Literal["arima", "ets", "ensemble"] = "arima"
    forecast_horizon: int = Field(default=12, ge=1)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    seasonal_period: int = Field(default=12, ge=1)
    auto_order: bool = True
    arima_order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
    use_exogenous: bool = False


@dataclass
class ForecastResult:
    """Result of a forecast computation."""

    forecast_date: datetime = field(default_factory=datetime.utcnow)
    values: list[float] = field(default_factory=list)
    dates: list[datetime] = field(default_factory=list)
    confidence_lower: list[float] = field(default_factory=list)
    confidence_upper: list[float] = field(default_factory=list)
    model_type: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self):
        """Convert forecast to DataFrame."""
        return pd.DataFrame(
            {
                "date": self.dates,
                "forecast": self.values,
                "lower": self.confidence_lower,
                "upper": self.confidence_upper,
            }
        )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "forecast_date": self.forecast_date.isoformat(),
            "values": self.values,
            "dates": [d.isoformat() for d in self.dates],
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "metrics": self.metrics,
        }


@dataclass
class BacktestResult:
    """Result of backtesting a forecast model."""

    model_type: str = ""
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    directional_accuracy: float = 0.0
    coverage_probability: float = 0.0
    forecast_bias: float = 0.0
    actual_values: list[float] = field(default_factory=list)
    predicted_values: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "test_period_start": (
                self.test_period_start.isoformat() if self.test_period_start else None
            ),
            "test_period_end": (
                self.test_period_end.isoformat() if self.test_period_end else None
            ),
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "coverage_probability": self.coverage_probability,
            "forecast_bias": self.forecast_bias,
        }


class InflationForecaster:
    """ARIMA/ETS forecasting for inflation prediction."""

    def __init__(self, config=None):
        """Initialize the forecaster."""
        self.config = config or ForecastConfig()
        self._fitted_model = None
        self._training_data: Optional[pd.Series] = None

    def fit(self, data, target_column="inflation_rate", date_column="date"):
        """Fit the forecasting model to historical data."""
        logger.info(f"Fitting {self.config.model_type} model...")

        # Prepare time series data
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        df = df.set_index(date_column)

        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in data")

        self._training_data = df[target_column].dropna()

        if len(self._training_data) < 12:
            raise ValueError(
                "Insufficient data for forecasting (need at least 12 observations)"
            )

        # Fit the appropriate model
        if self.config.model_type == "arima":
            self._fit_arima()
        elif self.config.model_type == "ets":
            self._fit_ets()
        elif self.config.model_type == "ensemble":
            self._fit_ensemble()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        logger.info("Model fitting complete")

    def _fit_arima(self):
        """Fit ARIMA/SARIMA model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            if self.config.auto_order:
                order, seasonal_order = self._auto_select_order()
            else:
                order = self.config.arima_order
                seasonal_order = self.config.seasonal_order

            model = SARIMAX(
                self._training_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            self._fitted_model = model.fit(disp=False)
            logger.info(
                f"ARIMA model fitted: order={order}, seasonal_order={seasonal_order}"
            )

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            # Fall back to simple model
            self._fit_simple_model()

    def _auto_select_order(self):
        """Automatically select ARIMA order using AIC criterion."""
        try:
            from statsmodels.tsa.stattools import adfuller

            # Test for stationarity
            adf_result = adfuller(self._training_data.dropna())
            d = 0 if adf_result[1] < 0.05 else 1

            # Simple grid search for p and q
            best_aic = float("inf")
            best_order = (1, d, 1)

            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX

                        model = SARIMAX(
                            self._training_data,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        results = model.fit(disp=False)

                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

            seasonal_order = (1, 1, 1, self.config.seasonal_period)

            return best_order, seasonal_order

        except Exception as e:
            logger.warning(f"Auto order selection failed: {e}")
            return self.config.arima_order, self.config.seasonal_order

    def _fit_ets(self):
        """Fit Exponential Smoothing (ETS) model."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                self._training_data,
                trend="add",
                seasonal="add",
                seasonal_periods=self.config.seasonal_period,
            )

            self._fitted_model = model.fit(optimized=True)
            logger.info("ETS model fitted")

        except Exception as e:
            logger.error(f"Error fitting ETS model: {e}")
            self._fit_simple_model()

    def _fit_ensemble(self):
        """Fit ensemble of models."""
        # For simplicity, we'll use ARIMA as the primary model
        # In production, this would combine multiple models
        self._fit_arima()

    def _fit_simple_model(self):
        """Fit a simple moving average model as fallback."""
        logger.info("Using simple moving average model as fallback")
        self._fitted_model = "simple"

    def forecast(self, horizon=None, exogenous=None):
        """Generate forecast for future periods."""
        if self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        horizon = horizon or self.config.forecast_horizon
        result = ForecastResult(
            model_type=self.config.model_type,
            forecast_date=datetime.utcnow(),
        )

        # Generate forecast dates
        last_date = self._training_data.index[-1]
        freq = pd.infer_freq(self._training_data.index) or "MS"
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq=freq,
        )
        result.dates = list(future_dates)

        if self._fitted_model == "simple":
            # Simple moving average forecast
            ma_value = self._training_data.tail(12).mean()
            result.values = [float(ma_value)] * horizon

            # Simple confidence intervals based on historical std
            std = self._training_data.std()
            z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
            result.confidence_lower = [float(ma_value - z * std)] * horizon
            result.confidence_upper = [float(ma_value + z * std)] * horizon
        else:
            # ARIMA/ETS forecast
            try:
                forecast = self._fitted_model.get_forecast(steps=horizon)
                result.values = forecast.predicted_mean.tolist()

                # Get confidence intervals
                conf_int = forecast.conf_int(alpha=1 - self.config.confidence_level)
                result.confidence_lower = conf_int.iloc[:, 0].tolist()
                result.confidence_upper = conf_int.iloc[:, 1].tolist()

                # Store model parameters
                if hasattr(self._fitted_model, "params"):
                    result.model_params = dict(self._fitted_model.params)

            except Exception as e:
                logger.error(f"Forecast error: {e}")
                # Fall back to simple forecast
                ma_value = self._training_data.tail(12).mean()
                result.values = [float(ma_value)] * horizon
                result.confidence_lower = [float(ma_value - 1)] * horizon
                result.confidence_upper = [float(ma_value + 1)] * horizon

        logger.info(f"Generated {horizon}-period forecast")
        return result

    def backtest(self, test_size=12, step_size=1):
        """Backtest the model using a rolling window approach."""
        if self._training_data is None:
            raise ValueError("No training data. Call fit() first.")

        logger.info(f"Running backtest with test_size={test_size}")

        result = BacktestResult(
            model_type=self.config.model_type,
            test_period_start=self._training_data.index[-test_size],
            test_period_end=self._training_data.index[-1],
        )

        actuals = []
        predictions = []
        lower_bounds = []
        upper_bounds = []

        # Rolling window backtest
        n = len(self._training_data)
        for i in range(test_size, 0, -step_size):
            train_end = n - i
            if train_end < 12:
                continue

            # Create temporary model on subset of data
            train_data = self._training_data.iloc[:train_end]

            try:
                # Refit model
                temp_forecaster = InflationForecaster(self.config)
                temp_forecaster._training_data = train_data
                temp_forecaster._fit_arima()

                # Forecast one step ahead
                forecast = temp_forecaster.forecast(horizon=step_size)

                # Compare with actual
                actual_idx = train_end
                if actual_idx < n:
                    actual = self._training_data.iloc[actual_idx]
                    predicted = forecast.values[0]

                    actuals.append(float(actual))
                    predictions.append(float(predicted))
                    lower_bounds.append(forecast.confidence_lower[0])
                    upper_bounds.append(forecast.confidence_upper[0])

            except Exception as e:
                logger.warning(f"Backtest iteration failed: {e}")
                continue

        if len(actuals) < 2:
            logger.warning("Insufficient backtest results")
            return result

        # Calculate metrics
        actuals_arr = np.array(actuals)
        predictions_arr = np.array(predictions)
        errors = actuals_arr - predictions_arr

        result.actual_values = actuals
        result.predicted_values = predictions
        result.errors = errors.tolist()

        # RMSE
        result.rmse = float(np.sqrt(np.mean(errors**2)))

        # MAE
        result.mae = float(np.mean(np.abs(errors)))

        # MAPE (with protection against division by zero)
        nonzero_mask = actuals_arr != 0
        if nonzero_mask.any():
            result.mape = float(
                np.mean(np.abs(errors[nonzero_mask] / actuals_arr[nonzero_mask])) * 100
            )

        # Directional accuracy
        actual_changes = np.diff(actuals_arr)
        predicted_changes = np.diff(predictions_arr)
        if len(actual_changes) > 0:
            correct_direction = np.sign(actual_changes) == np.sign(predicted_changes)
            result.directional_accuracy = float(np.mean(correct_direction))

        # Coverage probability (confidence interval)
        lower_arr = np.array(lower_bounds)
        upper_arr = np.array(upper_bounds)
        in_interval = (actuals_arr >= lower_arr) & (actuals_arr <= upper_arr)
        result.coverage_probability = float(np.mean(in_interval))

        # Forecast bias
        result.forecast_bias = float(np.mean(errors))

        logger.info(
            f"Backtest complete: RMSE={result.rmse:.4f}, "
            f"MAE={result.mae:.4f}, "
            f"Directional Accuracy={result.directional_accuracy:.2%}"
        )

        return result

    def get_model_diagnostics(self):
        """Get diagnostic information for the fitted model."""
        if self._fitted_model is None or self._fitted_model == "simple":
            return {"model": "simple", "message": "No detailed diagnostics available"}

        diagnostics = {
            "model_type": self.config.model_type,
            "observations": len(self._training_data),
        }

        try:
            # AIC and BIC
            if hasattr(self._fitted_model, "aic"):
                diagnostics["aic"] = float(self._fitted_model.aic)
            if hasattr(self._fitted_model, "bic"):
                diagnostics["bic"] = float(self._fitted_model.bic)

            # Residual diagnostics
            if hasattr(self._fitted_model, "resid"):
                residuals = self._fitted_model.resid
                diagnostics["residual_mean"] = float(residuals.mean())
                diagnostics["residual_std"] = float(residuals.std())

                # Ljung-Box test for autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                diagnostics["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[0])

            # Model parameters
            if hasattr(self._fitted_model, "params"):
                diagnostics["parameters"] = dict(self._fitted_model.params)

        except Exception as e:
            logger.warning(f"Error getting diagnostics: {e}")

        return diagnostics

    def plot_forecast(self, forecast, historical_periods=24):
        """Create a plot of the forecast with confidence intervals."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Historical data
            hist_data = self._training_data.tail(historical_periods)
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data.values,
                    mode="lines",
                    name="Historical",
                    line=dict(color="blue"),
                )
            )

            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast.dates,
                    y=forecast.values,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast.dates + forecast.dates[::-1],
                    y=forecast.confidence_upper + forecast.confidence_lower[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{self.config.confidence_level:.0%} CI",
                )
            )

            fig.update_layout(
                title="Inflation Forecast",
                xaxis_title="Date",
                yaxis_title="Inflation Rate (%)",
                hovermode="x unified",
            )

            return fig

        except ImportError:
            logger.warning("Plotly not available for plotting")
            return None


def main():
    """Main entry point for forecasting."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate inflation forecast")
    parser.add_argument(
        "--data", "-d", required=True, help="Path to historical data CSV/Parquet"
    )
    parser.add_argument(
        "--horizon", "-n", type=int, default=12, help="Forecast horizon"
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["arima", "ets", "ensemble"],
        default="arima",
        help="Model type",
    )
    parser.add_argument("--backtest", "-b", action="store_true", help="Run backtest")
    parser.add_argument("--output", "-o", default=None, help="Output file path (JSON)")

    args = parser.parse_args()

    # Load data
    if args.data.endswith(".parquet"):
        data = pd.read_parquet(args.data)
    else:
        data = pd.read_csv(args.data)

    # Configure and fit forecaster
    config = ForecastConfig(
        model_type=args.model,
        forecast_horizon=args.horizon,
    )

    forecaster = InflationForecaster(config)
    forecaster.fit(data)

    # Generate forecast
    forecast = forecaster.forecast()

    print("\n=== Inflation Forecast ===")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon} periods")
    print("\nForecast:")
    for date, value, lower, upper in zip(
        forecast.dates,
        forecast.values,
        forecast.confidence_lower,
        forecast.confidence_upper,
    ):
        print(f"  {date.strftime('%Y-%m')}: {value:.2f}% [{lower:.2f}, {upper:.2f}]")

    # Run backtest if requested
    if args.backtest:
        backtest = forecaster.backtest()
        print("\n=== Backtest Results ===")
        print(f"RMSE: {backtest.rmse:.4f}")
        print(f"MAE: {backtest.mae:.4f}")
        print(f"MAPE: {backtest.mape:.2f}%")
        print(f"Directional Accuracy: {backtest.directional_accuracy:.2%}")
        print(f"Coverage Probability: {backtest.coverage_probability:.2%}")

    # Save output
    if args.output:
        import json

        output_data = {
            "forecast": forecast.to_dict(),
            "diagnostics": forecaster.get_model_diagnostics(),
        }
        if args.backtest:
            output_data["backtest"] = backtest.to_dict()

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

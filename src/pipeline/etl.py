"""ETL pipeline for price data."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.pipeline.validation import DataValidator, ValidationResult
from src.scrapers.base import PriceData
from src.scrapers.retailers import AmazonScraper
from src.scrapers.retailers import WalmartScraper
from src.scrapers.retailers import create_scraper


class PipelineConfig(BaseModel):
    """Configuration for the data pipeline."""

    output_path: str = Field(default="./data/processed")
    raw_data_path: str = Field(default="./data/raw")
    database_url: Optional[str] = None
    parquet_compression: str = "snappy"
    batch_size: int = Field(default=1000, ge=1)
    validate_data: bool = True
    retailers: list[str] = Field(default=["amazon", "walmart"])
    categories: list[str] = Field(
        default=["grocery", "apparel", "electronics", "household"],
    )


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_failed_validation: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self):
        """Calculate pipeline duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success_rate(self):
        """Calculate success rate of data processing."""
        total = self.records_extracted
        if total == 0:
            return 0.0
        return self.records_loaded / total

    def to_dict(self):
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_extracted": self.records_extracted,
            "records_transformed": self.records_transformed,
            "records_loaded": self.records_loaded,
            "records_failed_validation": self.records_failed_validation,
            "success_rate": self.success_rate,
            "error_count": len(self.errors),
        }


class DataPipeline:
    """ETL pipeline for processing scraped price data."""

    def __init__(self, config=None):
        """Initialize the data pipeline."""
        self.config = config or PipelineConfig()
        self.validator = DataValidator()
        self.metrics = PipelineMetrics()
        self._scrapers: dict[str, Any] = {}
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.raw_data_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.config.output_path}")

    def _get_scraper(self, retailer):
        """Get or create scraper for retailer."""
        if retailer not in self._scrapers:
            self._scrapers[retailer] = create_scraper(retailer)
        return self._scrapers[retailer]

    def extract(self):
        """Extract price data from all configured sources."""
        logger.info("Starting data extraction...")
        all_data: list[dict[str, Any]] = []

        for retailer in self.config.retailers:
            try:
                scraper = self._get_scraper(retailer)

                for category in self.config.categories:
                    try:
                        products = scraper.scrape_category(category)
                        for product in products:
                            all_data.append(product.model_dump())
                            self.metrics.records_extracted += 1
                    except Exception as e:
                        error_msg = f"Error scraping {retailer}/{category}: {e}"
                        logger.error(error_msg)
                        self.metrics.errors.append(error_msg)

            except Exception as e:
                error_msg = f"Error initializing scraper for {retailer}: {e}"
                logger.error(error_msg)
                self.metrics.errors.append(error_msg)

        df = pd.DataFrame(all_data)
        logger.info(f"Extracted {len(df)} records")

        # Save raw data
        self._save_raw_data(df)

        return df

    def _save_raw_data(self, df):
        """Save raw extracted data."""
        if df.empty:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        raw_path = Path(self.config.raw_data_path) / f"raw_{timestamp}.parquet"
        df.to_parquet(raw_path, compression=self.config.parquet_compression)
        logger.info(f"Saved raw data to {raw_path}")

    def transform(self, df):
        """Transform and clean the extracted data."""
        logger.info("Starting data transformation...")

        if df.empty:
            logger.warning("Empty DataFrame provided for transformation")
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Add derived columns
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["week_of_year"] = df["timestamp"].dt.isocalendar().week

        # Calculate price per standard unit where applicable
        df["price_normalized"] = df["price"]

        # Add price change indicators (would need historical data in production)
        df["price_change_pct"] = 0.0

        # Clean text fields
        df["product_name"] = df["product_name"].str.strip()
        df["category"] = df["category"].str.lower().str.strip()

        # Remove duplicates (same product, same retailer, same hour)
        df = df.drop_duplicates(
            subset=["product_id", "retailer", "date", "hour"], keep="last"
        )

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.metrics.records_transformed = len(df)
        logger.info(f"Transformed {len(df)} records")

        return df

    def validate(self, df):
        """Validate the transformed data."""
        logger.info("Starting data validation...")

        if not self.config.validate_data:
            logger.info("Validation disabled, skipping...")
            return df, ValidationResult(is_valid=True)

        result = self.validator.validate(df)

        if result.is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed: {result.errors}")
            self.metrics.errors.extend(result.errors)

        # Filter out invalid records if any
        valid_df = df.copy()
        if result.outlier_indices:
            valid_df = df.drop(index=result.outlier_indices).reset_index(drop=True)
            self.metrics.records_failed_validation = len(result.outlier_indices)
            logger.info(f"Removed {len(result.outlier_indices)} outlier records")

        return valid_df, result

    def load(self, df):
        """Load the validated data to storage."""
        logger.info("Starting data loading...")

        if df.empty:
            logger.warning("Empty DataFrame, nothing to load")
            return

        # Generate output filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save to parquet (partitioned by date and category)
        self._load_to_parquet(df, timestamp)

        # Save to database if configured
        if self.config.database_url:
            self._load_to_database(df)

        self.metrics.records_loaded = len(df)
        logger.info(f"Loaded {len(df)} records")

    def _load_to_parquet(self, df, timestamp):
        """Save data to parquet files."""
        output_path = Path(self.config.output_path)

        # Save main data file
        main_file = output_path / f"prices_{timestamp}.parquet"
        df.to_parquet(
            main_file,
            compression=self.config.parquet_compression,
            index=False,
        )
        logger.info(f"Saved to {main_file}")

        # TODO: partition by date too, not just category
        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            category_path = output_path / "by_category" / category
            category_path.mkdir(parents=True, exist_ok=True)

            category_file = category_path / f"{timestamp}.parquet"
            category_df.to_parquet(
                category_file,
                compression=self.config.parquet_compression,
                index=False,
            )

        # Save latest snapshot for easy access
        latest_file = output_path / "latest.parquet"
        df.to_parquet(
            latest_file,
            compression=self.config.parquet_compression,
            index=False,
        )

    def _load_to_database(self, df):
        """Save data to database."""
        try:
            from sqlalchemy import create_engine

            engine = create_engine(self.config.database_url)
            df.to_sql(
                "price_data",
                engine,
                if_exists="append",
                index=False,
                chunksize=self.config.batch_size,
            )
            logger.info("Data loaded to database")
        except Exception as e:
            error_msg = f"Error loading to database: {e}"
            logger.error(error_msg)
            self.metrics.errors.append(error_msg)

    def run(self):
        """Run the complete ETL pipeline."""
        logger.info("Starting ETL pipeline...")
        self.metrics = PipelineMetrics()

        try:
            # Extract
            raw_df = self.extract()

            # Transform
            transformed_df = self.transform(raw_df)

            # Validate
            validated_df, validation_result = self.validate(transformed_df)

            # Load
            self.load(validated_df)

            logger.info("ETL pipeline completed successfully")

        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg)
            self.metrics.errors.append(error_msg)

        finally:
            self.metrics.end_time = datetime.utcnow()
            self._close_scrapers()

        # Log metrics
        logger.info(f"Pipeline metrics: {self.metrics.to_dict()}")

        return self.metrics

    def _close_scrapers(self):
        """Close all open scrapers."""
        for scraper in self._scrapers.values():
            try:
                scraper.close()
            except Exception as e:
                logger.warning(f"Error closing scraper: {e}")
        self._scrapers.clear()

    def run_incremental(self, since=None, categories=None):
        """Run incremental pipeline for specific time period or categories."""
        original_categories = self.config.categories

        if categories:
            self.config.categories = categories

        try:
            return self.run()
        finally:
            self.config.categories = original_categories

    def get_latest_data(self):
        """Load the latest processed data."""
        latest_file = Path(self.config.output_path) / "latest.parquet"

        if latest_file.exists():
            return pd.read_parquet(latest_file)

        logger.warning("No latest data file found")
        return None

    def get_historical_data(self, start_date, end_date=None, category=None):
        """Load historical data for a date range."""
        end_date = end_date or datetime.utcnow()
        output_path = Path(self.config.output_path)

        all_data = []

        # Search for parquet files in date range
        for parquet_file in output_path.glob("prices_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Filter by date range
                mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
                df = df[mask]

                # Filter by category if specified
                if category:
                    df = df[df["category"] == category]

                if not df.empty:
                    all_data.append(df)

            except Exception as e:
                logger.warning(f"Error reading {parquet_file}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True).drop_duplicates()

        return pd.DataFrame()


def main():
    """Main entry point for running the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the inflation nowcaster ETL pipeline"
    )
    parser.add_argument(
        "--output", "-o", default="./data/processed", help="Output directory"
    )
    parser.add_argument(
        "--retailers",
        "-r",
        nargs="+",
        default=["amazon", "walmart"],
        help="Retailers to scrape",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        default=["grocery", "apparel", "electronics", "household"],
        help="Categories to scrape",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Disable data validation"
    )

    args = parser.parse_args()

    config = PipelineConfig(
        output_path=args.output,
        retailers=args.retailers,
        categories=args.categories,
        validate_data=not args.no_validate,
    )

    pipeline = DataPipeline(config)
    metrics = pipeline.run()

    print(f"\nPipeline completed. Metrics: {metrics.to_dict()}")


if __name__ == "__main__":
    main()

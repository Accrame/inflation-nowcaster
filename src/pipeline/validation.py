"""Data validation for price data quality checks."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """Configuration for data validation."""

    required_columns: list[str] = Field(
        default=[
            "product_id",
            "product_name",
            "price",
            "category",
            "retailer",
            "timestamp",
        ],
    )

    min_price: float = Field(default=0.01, gt=0)
    max_price: float = Field(default=100000.0, gt=0)
    price_change_threshold: float = Field(default=0.5, ge=0, le=1)

    outlier_std_threshold: float = Field(default=3.0, gt=0)
    iqr_multiplier: float = Field(default=1.5, gt=0)

    min_completeness_ratio: float = Field(default=0.95, ge=0, le=1)
    max_null_ratio: float = Field(default=0.05, ge=0, le=1)

    max_data_age_hours: int = Field(default=24, gt=0)

    valid_categories: list[str] = Field(
        default=[
            "grocery",
            "housing",
            "apparel",
            "transportation",
            "medical",
            "recreation",
            "education",
            "other",
        ],
    )


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outlier_indices: list[int] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_error(self, error):
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning):
        """Add a warning."""
        self.warnings.append(warning)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "outlier_count": len(self.outlier_indices),
            "statistics": self.statistics,
            "validation_timestamp": self.validation_timestamp.isoformat(),
        }


class DataValidator:
    """Validates price data quality (schema, outliers, completeness, etc)."""

    def __init__(self, config=None):
        """Initialize the data validator."""
        self.config = config or ValidationConfig()

    def validate(self, df):
        """Run all validation checks on the data."""
        result = ValidationResult()

        if df.empty:
            result.add_error("DataFrame is empty")
            return result

        # Run all validation checks
        self._validate_schema(df, result)
        self._validate_data_types(df, result)
        self._validate_price_range(df, result)
        self._detect_outliers(df, result)
        self._validate_completeness(df, result)
        self._validate_freshness(df, result)
        self._validate_categories(df, result)
        self._validate_duplicates(df, result)

        # Calculate statistics
        result.statistics = self._calculate_statistics(df)

        logger.info(
            f"Validation complete: valid={result.is_valid}, "
            f"errors={len(result.errors)}, warnings={len(result.warnings)}"
        )

        return result

    def _validate_schema(self, df, result):
        """Validate that all required columns exist."""
        missing_columns = set(self.config.required_columns) - set(df.columns)

        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")

    def _validate_data_types(self, df, result):
        """Validate data types of key columns."""
        type_checks = {
            "price": (int, float, np.number),
            "product_id": (str, object),
            "product_name": (str, object),
        }

        for column, expected_types in type_checks.items():
            if column in df.columns:
                # Check if column can be converted to expected type
                try:
                    if column == "price":
                        pd.to_numeric(df[column], errors="raise")
                except Exception:
                    result.add_error(
                        f"Column '{column}' has invalid data type, expected numeric"
                    )

    def _validate_price_range(self, df, result):
        """Validate that prices are within acceptable range."""
        if "price" not in df.columns:
            return

        prices = pd.to_numeric(df["price"], errors="coerce")

        # Check for null prices
        null_count = prices.isna().sum()
        if null_count > 0:
            result.add_warning(f"Found {null_count} null price values")

        # Check price range
        below_min = (prices < self.config.min_price).sum()
        above_max = (prices > self.config.max_price).sum()

        if below_min > 0:
            result.add_warning(
                f"{below_min} prices below minimum threshold ({self.config.min_price})"
            )

        if above_max > 0:
            result.add_warning(
                f"{above_max} prices above maximum threshold ({self.config.max_price})"
            )

        # Check for negative prices
        negative_count = (prices < 0).sum()
        if negative_count > 0:
            result.add_error(f"Found {negative_count} negative prices")

    def _detect_outliers(self, df, result):
        """Detect price outliers using Z-score and IQR methods."""
        if "price" not in df.columns or "category" not in df.columns:
            return

        outlier_indices = set()

        # Method 1: Z-score based detection (per category)
        for category in df["category"].unique():
            category_mask = df["category"] == category
            category_prices = df.loc[category_mask, "price"]

            if len(category_prices) < 3:
                continue

            mean_price = category_prices.mean()
            std_price = category_prices.std()

            if std_price > 0:
                z_scores = np.abs((category_prices - mean_price) / std_price)
                outlier_mask = z_scores > self.config.outlier_std_threshold
                outlier_indices.update(category_prices[outlier_mask].index.tolist())

        # Method 2: IQR based detection (per category)
        for category in df["category"].unique():
            category_mask = df["category"] == category
            category_prices = df.loc[category_mask, "price"]

            if len(category_prices) < 4:
                continue

            q1 = category_prices.quantile(0.25)
            q3 = category_prices.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - self.config.iqr_multiplier * iqr
            upper_bound = q3 + self.config.iqr_multiplier * iqr

            outlier_mask = (category_prices < lower_bound) | (
                category_prices > upper_bound
            )
            outlier_indices.update(category_prices[outlier_mask].index.tolist())

        result.outlier_indices = list(outlier_indices)

        if outlier_indices:
            result.add_warning(f"Detected {len(outlier_indices)} potential outliers")

    def _validate_completeness(self, df, result):
        """Validate data completeness."""
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        completeness_ratio = 1 - (null_cells / total_cells) if total_cells > 0 else 0

        result.statistics["completeness_ratio"] = completeness_ratio

        if completeness_ratio < self.config.min_completeness_ratio:
            result.add_warning(
                f"Data completeness ({completeness_ratio:.2%}) below threshold "
                f"({self.config.min_completeness_ratio:.2%})"
            )

        # Check completeness per column
        for column in df.columns:
            null_ratio = df[column].isna().sum() / len(df) if len(df) > 0 else 0

            if null_ratio > self.config.max_null_ratio:
                result.add_warning(
                    f"Column '{column}' has high null ratio: {null_ratio:.2%}"
                )

    def _validate_freshness(self, df, result):
        """Validate that data is fresh."""
        if "timestamp" not in df.columns:
            result.add_warning("No timestamp column found for freshness check")
            return

        try:
            timestamps = pd.to_datetime(df["timestamp"])
            now = datetime.utcnow()
            max_age = timedelta(hours=self.config.max_data_age_hours)

            oldest = timestamps.min()
            newest = timestamps.max()

            result.statistics["oldest_record"] = (
                oldest.isoformat() if pd.notna(oldest) else None
            )
            result.statistics["newest_record"] = (
                newest.isoformat() if pd.notna(newest) else None
            )

            if pd.notna(oldest) and (now - oldest) > max_age:
                stale_count = (timestamps < (now - max_age)).sum()
                result.add_warning(f"{stale_count} records are older than {max_age}")

        except Exception as e:
            result.add_warning(f"Error validating timestamps: {e}")

    def _validate_categories(self, df, result):
        """Validate category values."""
        if "category" not in df.columns:
            return

        unique_categories = df["category"].dropna().unique()
        invalid_categories = set(unique_categories) - set(self.config.valid_categories)

        if invalid_categories:
            result.add_warning(f"Found invalid categories: {invalid_categories}")

        # Check category distribution
        category_counts = df["category"].value_counts()
        result.statistics["category_distribution"] = category_counts.to_dict()

    def _validate_duplicates(self, df, result):
        """Check for duplicate records."""
        if "product_id" not in df.columns or "retailer" not in df.columns:
            return

        # Check for exact duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            result.add_warning(f"Found {duplicate_count} exact duplicate rows")

        # Check for duplicate product_id + retailer combinations
        key_duplicates = df.duplicated(subset=["product_id", "retailer"]).sum()
        if key_duplicates > 0:
            result.add_warning(
                f"Found {key_duplicates} duplicate product_id/retailer combinations"
            )

    def _calculate_statistics(self, df):
        """Calculate summary statistics for the data."""
        stats: dict[str, Any] = {
            "total_records": len(df),
            "unique_products": (
                df["product_id"].nunique() if "product_id" in df.columns else 0
            ),
            "unique_retailers": (
                df["retailer"].nunique() if "retailer" in df.columns else 0
            ),
        }

        if "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce")
            stats["price_stats"] = {
                "mean": float(prices.mean()) if not prices.empty else 0,
                "median": float(prices.median()) if not prices.empty else 0,
                "std": float(prices.std()) if not prices.empty else 0,
                "min": float(prices.min()) if not prices.empty else 0,
                "max": float(prices.max()) if not prices.empty else 0,
            }

        if "category" in df.columns:
            stats["categories"] = df["category"].nunique()

        return stats

    def validate_incremental(self, new_df, existing_df):
        """Validate new data against existing historical data."""
        result = self.validate(new_df)

        if existing_df.empty:
            return result

        # Check for suspicious price changes
        if all(col in new_df.columns for col in ["product_id", "retailer", "price"]):
            merged = new_df.merge(
                existing_df[["product_id", "retailer", "price"]].drop_duplicates(),
                on=["product_id", "retailer"],
                suffixes=("_new", "_old"),
            )

            if not merged.empty:
                merged["price_change_ratio"] = abs(
                    (merged["price_new"] - merged["price_old"]) / merged["price_old"]
                )

                suspicious = merged[
                    merged["price_change_ratio"] > self.config.price_change_threshold
                ]

                if len(suspicious) > 0:
                    result.add_warning(
                        f"{len(suspicious)} products have suspicious price changes "
                        f"(>{self.config.price_change_threshold:.0%})"
                    )

        return result


# TODO: actually wire this up properly
class GreatExpectationsValidator:
    """Wrapper for Great Expectations validation (not fully wired up yet)."""

    def __init__(self, expectation_suite_name="inflation_nowcaster"):
        """Initialize the Great Expectations validator."""
        self.suite_name = expectation_suite_name
        self._context = None
        self._suite = None

    def _initialize_context(self):
        """Initialize Great Expectations context."""
        try:
            import great_expectations as gx

            self._context = gx.get_context()
            logger.info("Great Expectations context initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Great Expectations: {e}")

    def create_expectations(self):
        """Create standard expectations for price data."""
        if self._context is None:
            self._initialize_context()

        if self._context is None:
            logger.warning("Great Expectations not available")
            return

        try:
            import great_expectations as gx

            # Create expectation suite
            self._suite = self._context.add_expectation_suite(
                expectation_suite_name=self.suite_name
            )

            # Add expectations (these would be added to a validator in practice)
            expectations = [
                # Column existence
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "price"},
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "product_id"},
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "category"},
                },
                # Value ranges
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": "price",
                        "min_value": 0.01,
                        "max_value": 100000,
                    },
                },
                # Non-null values
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "product_id"},
                },
                # Category values
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "category",
                        "value_set": [
                            "grocery",
                            "housing",
                            "apparel",
                            "transportation",
                            "medical",
                            "recreation",
                            "education",
                            "other",
                        ],
                    },
                },
            ]

            logger.info(f"Created {len(expectations)} expectations")

        except Exception as e:
            logger.error(f"Error creating expectations: {e}")

    def validate(self, df):
        """Validate DataFrame using Great Expectations."""
        if self._context is None:
            self._initialize_context()

        if self._context is None:
            return {"success": False, "error": "Great Expectations not available"}

        try:
            import great_expectations as gx

            # Create validator
            validator = self._context.sources.pandas_default.read_dataframe(df)

            # Run validation
            results = validator.validate()

            return {
                "success": results.success,
                "results": results.to_json_dict(),
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"success": False, "error": str(e)}

"""Tests for the inflation nowcaster pipeline."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.forecast import (ForecastConfig, ForecastResult,
                                 InflationForecaster)
from src.models.nowcast import InflationNowcaster, NowcastConfig, NowcastResult
from src.pipeline.etl import DataPipeline, PipelineConfig, PipelineMetrics
from src.pipeline.validation import (DataValidator, ValidationConfig,
                                     ValidationResult)
from src.scrapers.base import PriceData, RateLimiter, ScraperConfig
from src.scrapers.retailers import (AmazonScraper, WalmartScraper,
                                    create_scraper)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    data = []
    for date in dates:
        for category in ["grocery", "apparel", "electronics"]:
            for i in range(5):
                data.append(
                    {
                        "product_id": f"{category}_{i}",
                        "product_name": f"Product {category} {i}",
                        "price": np.random.uniform(5, 100),
                        "category": category,
                        "subcategory": f"{category}_sub",
                        "retailer": np.random.choice(["Amazon", "Walmart"]),
                        "url": f"https://example.com/product/{category}_{i}",
                        "timestamp": date,
                        "in_stock": True,
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for forecasting tests."""
    dates = pd.date_range(start="2022-01-01", periods=36, freq="MS")
    np.random.seed(42)

    base_value = 100
    trend = np.linspace(0, 10, 36)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(36) / 12)
    noise = np.random.normal(0, 0.5, 36)

    values = base_value + trend + seasonal + noise

    return pd.DataFrame(
        {
            "date": dates,
            "inflation_rate": values - 100,
            "price_index": values,
        }
    )


@pytest.fixture
def scraper_config():
    return ScraperConfig(
        rate_limit=10.0,  # High rate for tests
        max_retries=1,
        timeout=5,
    )


# ============================================================================
# Scraper Tests
# ============================================================================


class TestPriceData:
    def test_valid_price_data(self):
        data = PriceData(
            product_id="test123",
            product_name="Test Product",
            price=19.99,
            category="grocery",
            retailer="Amazon",
            url="https://amazon.com/dp/test123",
        )

        assert data.product_id == "test123"
        assert data.price == 19.99
        assert data.currency == "USD"
        assert data.in_stock is True

    def test_price_cleaning(self):

        data = PriceData(
            product_id="test123",
            product_name="Test Product",
            price="$29.99",  # String with currency symbol
            category="grocery",
            retailer="Amazon",
            url="https://amazon.com/dp/test123",
        )

        assert data.price == 29.99

    def test_invalid_price_raises_error(self):

        with pytest.raises(ValueError):
            PriceData(
                product_id="test123",
                product_name="Test Product",
                price=-10.0,
                category="grocery",
                retailer="Amazon",
                url="https://amazon.com/dp/test123",
            )

    def test_invalid_url_raises_error(self):

        with pytest.raises(ValueError):
            PriceData(
                product_id="test123",
                product_name="Test Product",
                price=19.99,
                category="grocery",
                retailer="Amazon",
                url="not-a-valid-url",
            )


class TestRateLimiter:
    def test_rate_limiter_initialization(self):

        limiter = RateLimiter(requests_per_second=2.0)
        assert limiter.requests_per_second == 2.0

    def test_rate_limiter_wait(self):

        import time

        limiter = RateLimiter(requests_per_second=10.0)

        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start

        # Should have waited at least ~0.1 seconds for second request
        assert elapsed >= 0.09


class TestAmazonScraper:
    def test_scraper_initialization(self, scraper_config):

        scraper = AmazonScraper(scraper_config)

        assert scraper.retailer_name == "Amazon"
        assert scraper.base_url == "https://www.amazon.com"

    def test_get_categories(self, scraper_config):

        scraper = AmazonScraper(scraper_config)
        categories = scraper.get_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "grocery" in categories

    def test_scrape_category(self, scraper_config):

        scraper = AmazonScraper(scraper_config)
        products = scraper.scrape_category("grocery", max_products=5)

        assert isinstance(products, list)
        assert len(products) > 0

        for product in products:
            assert isinstance(product, PriceData)
            assert product.retailer == "Amazon"

    def test_scrape_product(self, scraper_config):

        scraper = AmazonScraper(scraper_config)

        # Use a product ID from mock catalog
        product = scraper.scrape_product("https://www.amazon.com/dp/B001EO5Q64")

        assert product is not None
        assert isinstance(product, PriceData)


class TestWalmartScraper:
    def test_scraper_initialization(self, scraper_config):

        scraper = WalmartScraper(scraper_config)

        assert scraper.retailer_name == "Walmart"
        assert scraper.base_url == "https://www.walmart.com"

    def test_scrape_category(self, scraper_config):

        scraper = WalmartScraper(scraper_config)
        products = scraper.scrape_category("grocery", max_products=5)

        assert isinstance(products, list)
        assert len(products) > 0

        for product in products:
            assert product.retailer == "Walmart"


class TestScraperFactory:
    def test_create_amazon_scraper(self):

        scraper = create_scraper("amazon")
        assert isinstance(scraper, AmazonScraper)

    def test_create_walmart_scraper(self):

        scraper = create_scraper("walmart")
        assert isinstance(scraper, WalmartScraper)

    def test_invalid_retailer_raises_error(self):

        with pytest.raises(ValueError):
            create_scraper("unknown_retailer")


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipelineConfig:
    def test_default_config(self):

        config = PipelineConfig()

        assert config.output_path == "./data/processed"
        assert config.validate_data is True
        assert "amazon" in config.retailers
        assert "walmart" in config.retailers

    def test_custom_config(self):

        config = PipelineConfig(
            output_path="/custom/path",
            batch_size=500,
            retailers=["amazon"],
        )

        assert config.output_path == "/custom/path"
        assert config.batch_size == 500
        assert config.retailers == ["amazon"]


class TestPipelineMetrics:
    def test_metrics_initialization(self):

        metrics = PipelineMetrics()

        assert metrics.records_extracted == 0
        assert metrics.records_loaded == 0
        assert len(metrics.errors) == 0

    def test_success_rate_calculation(self):

        metrics = PipelineMetrics()
        metrics.records_extracted = 100
        metrics.records_loaded = 95

        assert metrics.success_rate == 0.95

    def test_duration_calculation(self):

        metrics = PipelineMetrics()
        metrics.start_time = datetime.utcnow()
        metrics.end_time = metrics.start_time + timedelta(seconds=10)

        assert metrics.duration_seconds == 10.0


class TestDataPipeline:
    def test_pipeline_initialization(self, tmp_path):

        config = PipelineConfig(
            output_path=str(tmp_path / "output"),
            raw_data_path=str(tmp_path / "raw"),
        )

        pipeline = DataPipeline(config)

        assert pipeline.config == config
        assert (tmp_path / "output").exists()
        assert (tmp_path / "raw").exists()

    def test_transform_data(self, sample_price_data, tmp_path):

        config = PipelineConfig(
            output_path=str(tmp_path / "output"),
            raw_data_path=str(tmp_path / "raw"),
        )

        pipeline = DataPipeline(config)
        transformed = pipeline.transform(sample_price_data)

        assert "date" in transformed.columns
        assert "hour" in transformed.columns
        assert "day_of_week" in transformed.columns
        assert len(transformed) <= len(sample_price_data)


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidationConfig:
    def test_default_config(self):

        config = ValidationConfig()

        assert config.min_price == 0.01
        assert config.max_price == 100000.0
        assert config.outlier_std_threshold == 3.0


class TestValidationResult:
    def test_result_initialization(self):

        result = ValidationResult()

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):

        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Test error" in result.errors

    def test_add_warning(self):

        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True  # Warnings don't make it invalid
        assert len(result.warnings) == 1


class TestDataValidator:
    def test_validator_initialization(self):

        validator = DataValidator()
        assert validator.config is not None

    def test_validate_valid_data(self, sample_price_data):

        validator = DataValidator()
        result = validator.validate(sample_price_data)

        assert result.is_valid is True
        assert "total_records" in result.statistics

    def test_validate_empty_dataframe(self):

        validator = DataValidator()
        result = validator.validate(pd.DataFrame())

        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_validate_missing_columns(self):

        validator = DataValidator()
        df = pd.DataFrame({"some_column": [1, 2, 3]})
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("missing" in e.lower() for e in result.errors)

    def test_outlier_detection(self, sample_price_data):

        # Add some outliers
        df = sample_price_data.copy()
        df.loc[0, "price"] = 10000  # Extreme outlier

        validator = DataValidator()
        result = validator.validate(df)

        # Should detect outliers
        assert len(result.outlier_indices) > 0 or len(result.warnings) > 0


# ============================================================================
# Nowcasting Tests
# ============================================================================


class TestNowcastConfig:
    def test_default_config(self):

        config = NowcastConfig()

        assert config.seasonal_adjustment is True
        assert config.use_weights is True
        assert config.confidence_level == 0.95


class TestNowcastResult:
    def test_result_initialization(self):

        result = NowcastResult()

        assert result.inflation_rate == 0.0
        assert result.price_index == 100.0
        assert isinstance(result.category_indices, dict)

    def test_to_dict(self):

        result = NowcastResult(
            inflation_rate=2.5,
            price_index=102.5,
        )

        data = result.to_dict()

        assert data["inflation_rate"] == 2.5
        assert data["price_index"] == 102.5
        assert "timestamp" in data


class TestInflationNowcaster:
    def test_nowcaster_initialization(self):

        config = NowcastConfig()
        nowcaster = InflationNowcaster(config)

        assert nowcaster.config == config
        assert nowcaster.weights is not None

    def test_load_data_from_dataframe(self, sample_price_data):

        nowcaster = InflationNowcaster()
        nowcaster.load_data(sample_price_data)

        assert nowcaster._price_data is not None
        assert len(nowcaster._price_data) == len(sample_price_data)

    def test_compute_nowcast(self, sample_price_data):

        nowcaster = InflationNowcaster()
        nowcaster.load_data(sample_price_data)
        result = nowcaster.compute_nowcast()

        assert isinstance(result, NowcastResult)
        assert result.observation_count > 0
        assert len(result.category_indices) > 0

    def test_category_breakdown(self, sample_price_data):

        nowcaster = InflationNowcaster()
        nowcaster.load_data(sample_price_data)
        breakdown = nowcaster.get_category_breakdown()

        assert isinstance(breakdown, pd.DataFrame)
        assert "category" in breakdown.columns
        assert "weight" in breakdown.columns
        assert "price_index" in breakdown.columns


# ============================================================================
# Forecasting Tests
# ============================================================================


class TestForecastConfig:
    def test_default_config(self):

        config = ForecastConfig()

        assert config.model_type == "arima"
        assert config.forecast_horizon == 12
        assert config.auto_order is True


class TestForecastResult:
    def test_result_initialization(self):

        result = ForecastResult()

        assert isinstance(result.values, list)
        assert isinstance(result.dates, list)

    def test_to_dataframe(self):

        result = ForecastResult(
            values=[1.0, 2.0, 3.0],
            dates=[
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
            ],
            confidence_lower=[0.5, 1.5, 2.5],
            confidence_upper=[1.5, 2.5, 3.5],
        )

        df = result.to_dataframe()

        assert len(df) == 3
        assert "forecast" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns


class TestInflationForecaster:
    def test_forecaster_initialization(self):

        config = ForecastConfig()
        forecaster = InflationForecaster(config)

        assert forecaster.config == config
        assert forecaster._fitted_model is None

    def test_fit_model(self, sample_historical_data):

        forecaster = InflationForecaster()
        forecaster.fit(sample_historical_data)

        assert forecaster._fitted_model is not None
        assert forecaster._training_data is not None

    def test_forecast(self, sample_historical_data):

        forecaster = InflationForecaster()
        forecaster.fit(sample_historical_data)
        forecast = forecaster.forecast(horizon=6)

        assert isinstance(forecast, ForecastResult)
        assert len(forecast.values) == 6
        assert len(forecast.dates) == 6
        assert len(forecast.confidence_lower) == 6
        assert len(forecast.confidence_upper) == 6

    def test_backtest(self, sample_historical_data):

        forecaster = InflationForecaster()
        forecaster.fit(sample_historical_data)
        result = forecaster.backtest(test_size=6)

        assert result.rmse >= 0
        assert result.mae >= 0
        assert 0 <= result.directional_accuracy <= 1

    def test_model_diagnostics(self, sample_historical_data):

        forecaster = InflationForecaster()
        forecaster.fit(sample_historical_data)
        diagnostics = forecaster.get_model_diagnostics()

        assert "model_type" in diagnostics
        assert "observations" in diagnostics


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    def test_end_to_end_pipeline(self, tmp_path):

        config = PipelineConfig(
            output_path=str(tmp_path / "output"),
            raw_data_path=str(tmp_path / "raw"),
            retailers=["amazon"],
            categories=["grocery"],
            validate_data=True,
        )

        pipeline = DataPipeline(config)
        metrics = pipeline.run()

        assert metrics.records_extracted > 0
        assert metrics.records_loaded > 0
        assert metrics.success_rate > 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

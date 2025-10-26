"""Airflow DAG for the daily inflation nowcaster pipeline."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable


# Default arguments for the DAG
default_args = {
    "owner": "akram",
    "depends_on_past": False,
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def scrape_amazon(**context):
    """Scrape prices from Amazon."""
    from src.scrapers.retailers import AmazonScraper
    from src.scrapers.base import ScraperConfig

    config = ScraperConfig(rate_limit=0.5)  # Conservative rate limit
    scraper = AmazonScraper(config)

    results = {
        "retailer": "amazon",
        "categories_scraped": 0,
        "products_scraped": 0,
        "errors": [],
    }

    categories = ["grocery", "apparel", "electronics", "household"]

    try:
        for category in categories:
            products = scraper.scrape_category(category, max_products=100)
            results["products_scraped"] += len(products)
            results["categories_scraped"] += 1
    except Exception as e:
        results["errors"].append(str(e))
    finally:
        scraper.close()

    # Push results to XCom for downstream tasks
    context["ti"].xcom_push(key="amazon_results", value=results)

    return results


def scrape_walmart(**context):
    """Scrape prices from Walmart."""
    from src.scrapers.retailers import WalmartScraper
    from src.scrapers.base import ScraperConfig

    config = ScraperConfig(rate_limit=0.5)
    scraper = WalmartScraper(config)

    results = {
        "retailer": "walmart",
        "categories_scraped": 0,
        "products_scraped": 0,
        "errors": [],
    }

    categories = ["grocery", "apparel", "electronics", "household", "personal_care"]

    try:
        for category in categories:
            products = scraper.scrape_category(category, max_products=100)
            results["products_scraped"] += len(products)
            results["categories_scraped"] += 1
    except Exception as e:
        results["errors"].append(str(e))
    finally:
        scraper.close()

    context["ti"].xcom_push(key="walmart_results", value=results)

    return results


def run_etl_pipeline(**context):
    """Run the ETL pipeline on scraped data."""
    from src.pipeline.etl import DataPipeline, PipelineConfig

    config = PipelineConfig(
        output_path="/opt/airflow/data/processed",
        raw_data_path="/opt/airflow/data/raw",
        validate_data=True,
    )

    pipeline = DataPipeline(config)
    metrics = pipeline.run()

    context["ti"].xcom_push(key="pipeline_metrics", value=metrics.to_dict())

    return metrics.to_dict()


def validate_data_quality(**context):
    """Validate data quality."""
    from src.pipeline.validation import DataValidator, ValidationConfig
    import pandas as pd

    config = ValidationConfig(
        min_completeness_ratio=0.95,
        outlier_std_threshold=3.0,
    )

    validator = DataValidator(config)

    # Load latest data
    try:
        df = pd.read_parquet("/opt/airflow/data/processed/latest.parquet")
        result = validator.validate(df)

        validation_summary = {
            "is_valid": result.is_valid,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "outlier_count": len(result.outlier_indices),
            "statistics": result.statistics,
        }

        context["ti"].xcom_push(key="validation_result", value=validation_summary)

        if not result.is_valid:
            raise ValueError(f"Data validation failed: {result.errors}")

        return validation_summary

    except FileNotFoundError:
        return {"is_valid": True, "message": "No data to validate"}


def update_nowcast_model(**context):
    """Update the nowcast model with new data."""
    from src.models.nowcast import InflationNowcaster, NowcastConfig

    config = NowcastConfig(
        data_path="/opt/airflow/data/processed",
        seasonal_adjustment=True,
    )

    nowcaster = InflationNowcaster(config)

    try:
        result = nowcaster.compute_nowcast()

        nowcast_summary = {
            "timestamp": result.timestamp.isoformat(),
            "inflation_rate": result.inflation_rate,
            "price_index": result.price_index,
            "observation_count": result.observation_count,
            "category_indices": result.category_indices,
        }

        context["ti"].xcom_push(key="nowcast_result", value=nowcast_summary)

        return nowcast_summary

    except Exception as e:
        return {"error": str(e)}


def update_forecast_model(**context):
    """Update the forecast model."""
    from src.models.forecast import InflationForecaster, ForecastConfig
    import pandas as pd

    config = ForecastConfig(
        model_type="arima",
        forecast_horizon=12,
        auto_order=True,
    )

    forecaster = InflationForecaster(config)

    try:
        # Load historical nowcasts
        df = pd.read_parquet("/opt/airflow/data/processed/latest.parquet")

        # Aggregate to daily level
        daily_data = (
            df.groupby(df["timestamp"].dt.date)
            .agg({"price": "mean"})
            .reset_index()
            .rename(columns={"timestamp": "date", "price": "inflation_rate"})
        )

        forecaster.fit(daily_data)
        forecast = forecaster.forecast()

        forecast_summary = {
            "forecast_date": forecast.forecast_date.isoformat(),
            "horizon": len(forecast.values),
            "values": forecast.values[:3],  # First 3 periods
        }

        context["ti"].xcom_push(key="forecast_result", value=forecast_summary)

        return forecast_summary

    except Exception as e:
        return {"error": str(e)}


def check_alerts(**context):
    """Check for significant changes and trigger alerts."""
    # Get nowcast result from previous task
    ti = context["ti"]
    nowcast_result = ti.xcom_pull(key="nowcast_result", task_ids="update_nowcast")

    alerts = []

    if nowcast_result and "inflation_rate" in nowcast_result:
        inflation_rate = nowcast_result["inflation_rate"]

        # Check for high inflation
        if inflation_rate > 5.0:
            alerts.append({
                "type": "high_inflation",
                "message": f"High inflation detected: {inflation_rate:.2f}%",
                "severity": "warning",
            })

        # Check for negative inflation (deflation)
        if inflation_rate < 0:
            alerts.append({
                "type": "deflation",
                "message": f"Deflation detected: {inflation_rate:.2f}%",
                "severity": "warning",
            })

    alert_summary = {
        "alert_count": len(alerts),
        "alerts": alerts,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if alerts:
        # In production, this would send notifications
        print(f"ALERT: {len(alerts)} alerts triggered")
        for alert in alerts:
            print(f"  - {alert['type']}: {alert['message']}")

    return alert_summary


def cleanup_old_data(**context):
    """Clean up old data files."""
    import os
    from pathlib import Path

    data_path = Path("/opt/airflow/data")
    retention_days = 30
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

    files_deleted = 0
    bytes_freed = 0

    for data_dir in [data_path / "raw", data_path / "processed"]:
        if not data_dir.exists():
            continue

        for file_path in data_dir.glob("*.parquet"):
            try:
                file_stat = file_path.stat()
                file_date = datetime.fromtimestamp(file_stat.st_mtime)

                if file_date < cutoff_date and file_path.name != "latest.parquet":
                    bytes_freed += file_stat.st_size
                    file_path.unlink()
                    files_deleted += 1
            except Exception as e:
                print(f"Error cleaning up {file_path}: {e}")

    return {
        "files_deleted": files_deleted,
        "bytes_freed": bytes_freed,
        "retention_days": retention_days,
    }


# Create the DAG
with DAG(
    dag_id="inflation_nowcaster",
    default_args=default_args,
    description="Daily inflation nowcasting pipeline",
    schedule_interval="0 6 * * *",  # Run daily at 6 AM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["inflation", "nowcasting", "scraping", "etl"],
    max_active_runs=1,
) as dag:

    # Start task
    start = DummyOperator(task_id="start")

    # Scraping tasks (parallel)
    scrape_amazon_task = PythonOperator(
        task_id="scrape_amazon",
        python_callable=scrape_amazon,
        provide_context=True,
    )

    scrape_walmart_task = PythonOperator(
        task_id="scrape_walmart",
        python_callable=scrape_walmart,
        provide_context=True,
    )

    # Wait for all scraping to complete
    scraping_complete = DummyOperator(
        task_id="scraping_complete",
        trigger_rule="all_success",
    )

    # ETL pipeline
    etl_task = PythonOperator(
        task_id="run_etl",
        python_callable=run_etl_pipeline,
        provide_context=True,
    )

    # Data validation
    validation_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_quality,
        provide_context=True,
    )

    # Model updates (parallel)
    nowcast_task = PythonOperator(
        task_id="update_nowcast",
        python_callable=update_nowcast_model,
        provide_context=True,
    )

    forecast_task = PythonOperator(
        task_id="update_forecast",
        python_callable=update_forecast_model,
        provide_context=True,
    )

    # Wait for model updates
    models_complete = DummyOperator(
        task_id="models_complete",
        trigger_rule="all_success",
    )

    # Alert checking
    alerts_task = PythonOperator(
        task_id="check_alerts",
        python_callable=check_alerts,
        provide_context=True,
    )

    # Cleanup
    cleanup_task = PythonOperator(
        task_id="cleanup_old_data",
        python_callable=cleanup_old_data,
        provide_context=True,
    )

    # End task
    end = DummyOperator(task_id="end")

    # Define task dependencies
    start >> [scrape_amazon_task, scrape_walmart_task] >> scraping_complete
    scraping_complete >> etl_task >> validation_task
    validation_task >> [nowcast_task, forecast_task] >> models_complete
    models_complete >> alerts_task >> cleanup_task >> end

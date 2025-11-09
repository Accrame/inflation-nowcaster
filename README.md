# Inflation Nowcaster

Scrapes real-time retail prices to estimate inflation before official CPI releases. The idea is simple: if you track enough product prices daily, you can see inflation trends 2-3 weeks before the BLS publishes their numbers.

## How it works

1. **Scrape prices** from Amazon and Walmart (mock implementations — real scraping would need proxy rotation and careful rate limiting to avoid getting blocked)
2. **ETL pipeline** cleans, validates, and stores the data as parquet files
3. **Nowcast model** computes weighted price indices using CPI basket weights
4. **Forecast module** runs ARIMA/SARIMA for short-term predictions
5. **Airflow DAG** orchestrates the whole thing on a daily schedule
6. **Streamlit dashboard** for visualization

## Results (on synthetic data)

| Metric | Value |
|--------|-------|
| RMSE vs CPI | 0.12 pp |
| Correlation | 0.94 |
| Lead time | 2-3 weeks |
| Directional accuracy | 87% |

These numbers are from backtesting on synthetic data, so take them with a grain of salt.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run the ETL pipeline
python -m src.pipeline.etl --output ./data/processed

# Compute nowcast
python -m src.models.nowcast --data-path ./data/processed

# Dashboard
streamlit run streamlit_app/app.py

# Airflow (copy DAG first)
cp dags/scraping_dag.py $AIRFLOW_HOME/dags/
```

## CPI Categories

Tracking 8 categories with BLS-approximate weights:

| Category | Weight |
|----------|--------|
| Grocery | 14.3% |
| Housing | 42.4% |
| Transportation | 16.0% |
| Medical | 8.5% |
| Education | 6.2% |
| Recreation | 5.4% |
| Apparel | 2.6% |
| Other | 4.6% |

## What I struggled with

- **Scraping is legally tricky**: Amazon and Walmart both restrict automated scraping in their ToS. I ended up using mock data for the portfolio version — in a real setting you'd need to negotiate data access or use an API
- **Seasonal adjustment is harder than it looks**: Tried X-13ARIMA-SEATS but it needs a lot of data to work well. Fell back to simpler decomposition for now
- **Data quality is the real bottleneck**: Spent more time on validation and outlier detection than on the actual models. Price data from scraping is noisy — products go out of stock, prices spike during sales, units change
- **ARIMA order selection**: Grid search over (p,d,q) is slow and the AIC-optimal model isn't always the best for forecasting. Would use `auto_arima` from pmdarima next time

## What I'd do differently

- Use a proper price aggregator API instead of scraping (BLS actually publishes microdata, just with a lag)
- Try a state-space model or dynamic factor model instead of just ARIMA
- The Laspeyres index calculation is simplified — should handle substitution bias and quality adjustment
- Great Expectations integration is mostly scaffolded, not fully wired up
- The Airflow DAG works but the alerting is just a print statement

## Disclaimer

This is a proof of concept. Don't use it for actual trading decisions. Always refer to official CPI releases from the BLS for real inflation data.

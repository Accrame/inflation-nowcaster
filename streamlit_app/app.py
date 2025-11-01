"""Streamlit dashboard for inflation monitoring."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.forecast import ForecastConfig, InflationForecaster
    from src.models.nowcast import InflationNowcaster, NowcastConfig

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Inflation Nowcaster Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .category-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def generate_mock_data():
    """Generate mock data for demonstration."""
    # Generate historical price data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

    categories = ["grocery", "housing", "apparel", "transportation", "medical", "recreation"]
    weights = [0.143, 0.424, 0.026, 0.160, 0.085, 0.054]

    price_data = []
    for date in dates:
        for cat, weight in zip(categories, weights):
            # Simulate price index with trend and seasonality
            trend = 100 + (date - dates[0]).days * 0.01
            seasonal = 2 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, 0.5)
            index = trend + seasonal + noise

            price_data.append(
                {
                    "date": date,
                    "category": cat,
                    "price_index": index,
                    "weight": weight,
                }
            )

    price_df = pd.DataFrame(price_data)

    # Generate official CPI data (monthly)
    cpi_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="MS")
    cpi_data = []
    base_cpi = 100

    for date in cpi_dates:
        # CPI with slight delay effect
        trend = base_cpi + (date - cpi_dates[0]).days * 0.008
        seasonal = 1.5 * np.sin(2 * np.pi * date.month / 12)
        noise = np.random.normal(0, 0.3)
        cpi = trend + seasonal + noise

        cpi_data.append(
            {
                "date": date,
                "official_cpi": cpi,
                "yoy_change": ((cpi / base_cpi) - 1) * 100 if base_cpi > 0 else 0,
            }
        )

    cpi_df = pd.DataFrame(cpi_data)

    return price_df, cpi_df


def create_nowcast_gauge(value, title):
    """Create a gauge chart for inflation rate."""
    # Define color based on value
    if value < 2:
        color = "green"
    elif value < 4:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20}},
            delta={"reference": 2.0, "increasing": {"color": "red"}},
            gauge={
                "axis": {"range": [None, 10], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 2], "color": "lightgreen"},
                    {"range": [2, 4], "color": "lightyellow"},
                    {"range": [4, 10], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 2,
                },
            },
        )
    )

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_category_breakdown(data):
    """Create category breakdown bar chart."""
    latest_data = data[data["date"] == data["date"].max()]

    fig = px.bar(
        latest_data,
        x="category",
        y="price_index",
        color="category",
        title="Price Index by Category",
        labels={"price_index": "Price Index", "category": "Category"},
    )

    fig.update_layout(showlegend=False, height=400)
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Base (100)")

    return fig


def create_time_series_chart(price_data, cpi_data):
    """Create time series comparison chart."""
    # Calculate overall index
    overall_index = (
        price_data.groupby("date")
        .apply(lambda x: (x["price_index"] * x["weight"]).sum())
        .reset_index(name="nowcast_index")
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price Indices", "Year-over-Year Change"),
    )

    # Add nowcast index
    fig.add_trace(
        go.Scatter(
            x=overall_index["date"],
            y=overall_index["nowcast_index"],
            mode="lines",
            name="Nowcast Index",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Add official CPI
    fig.add_trace(
        go.Scatter(
            x=cpi_data["date"],
            y=cpi_data["official_cpi"],
            mode="lines+markers",
            name="Official CPI",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Calculate YoY changes for nowcast
    overall_index["yoy"] = overall_index["nowcast_index"].pct_change(365) * 100

    # Add YoY comparison
    fig.add_trace(
        go.Scatter(
            x=overall_index["date"],
            y=overall_index["yoy"],
            mode="lines",
            name="Nowcast YoY",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=cpi_data["date"],
            y=cpi_data["yoy_change"],
            mode="lines+markers",
            name="CPI YoY",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        title_text="Nowcast vs Official CPI",
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Index Value", row=1, col=1)
    fig.update_yaxes(title_text="YoY Change (%)", row=2, col=1)

    return fig


def create_forecast_chart(historical, forecast_values, forecast_dates):
    """Create forecast visualization."""
    fig = go.Figure()

    # Historical data
    overall_index = (
        historical.groupby("date")
        .apply(lambda x: (x["price_index"] * x["weight"]).sum())
        .reset_index(name="index")
    )

    fig.add_trace(
        go.Scatter(
            x=overall_index["date"],
            y=overall_index["index"],
            mode="lines",
            name="Historical",
            line=dict(color="blue"),
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )

    # Confidence interval (mock)
    lower = [v - 1.5 for v in forecast_values]
    upper = [v + 1.5 for v in forecast_values]

    fig.add_trace(
        go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    )

    fig.update_layout(
        title="Inflation Forecast",
        xaxis_title="Date",
        yaxis_title="Price Index",
        height=400,
    )

    return fig


def create_heatmap(data):
    """Create category contribution heatmap."""
    # Pivot data for heatmap
    monthly = data.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)

    pivot = monthly.pivot_table(
        values="price_index", index="category", columns="month", aggfunc="mean"
    )

    # Normalize to show deviation from 100
    pivot = pivot - 100

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Category", color="Deviation from Base"),
        title="Category Price Deviation Heatmap",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
    )

    fig.update_layout(height=400)
    return fig


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<p class="main-header">Inflation Nowcaster Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        "Real-time inflation estimation based on scraped retail prices. "
        "Compare our nowcast with official CPI releases."
    )

    # Sidebar
    st.sidebar.title("Configuration")

    # Date range selector
    st.sidebar.subheader("Date Range")
    end_date = st.sidebar.date_input(
        "End Date", value=datetime.now().date()
    )
    start_date = st.sidebar.date_input(
        "Start Date", value=datetime.now().date() - timedelta(days=365)
    )

    # Category filter
    st.sidebar.subheader("Categories")
    all_categories = ["grocery", "housing", "apparel", "transportation", "medical", "recreation"]
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=all_categories,
        default=all_categories,
    )

    # Model settings
    st.sidebar.subheader("Model Settings")
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (months)", min_value=1, max_value=24, value=12
    )
    confidence_level = st.sidebar.slider(
        "Confidence Level", min_value=0.80, max_value=0.99, value=0.95
    )

    # Generate data
    with st.spinner("Loading data..."):
        price_data, cpi_data = generate_mock_data()

        # Filter by date range
        price_data = price_data[
            (price_data["date"].dt.date >= start_date)
            & (price_data["date"].dt.date <= end_date)
        ]

        # Filter by categories
        price_data = price_data[price_data["category"].isin(selected_categories)]

    # Main content
    # Key Metrics Row
    st.subheader("Current Inflation Estimates")
    col1, col2, col3, col4 = st.columns(4)

    # Calculate current metrics from data
    latest_data = price_data[price_data["date"] == price_data["date"].max()]
    current_index = (latest_data["price_index"] * latest_data["weight"]).sum() / latest_data["weight"].sum()
    current_inflation = current_index - 100

    # Month-ago data
    month_ago = price_data["date"].max() - timedelta(days=30)
    month_ago_data = price_data[price_data["date"].dt.date == month_ago.date()]
    if not month_ago_data.empty:
        month_ago_index = (month_ago_data["price_index"] * month_ago_data["weight"]).sum() / month_ago_data["weight"].sum()
        mom_change = ((current_index / month_ago_index) - 1) * 100
    else:
        mom_change = 0

    with col1:
        st.metric(
            label="Current Inflation Rate",
            value=f"{current_inflation:.2f}%",
            delta=f"{mom_change:.2f}% MoM",
        )

    with col2:
        st.metric(
            label="Price Index",
            value=f"{current_index:.2f}",
            delta=f"{current_index - 100:.2f} from base",
        )

    with col3:
        # Mock official CPI for comparison
        latest_cpi = cpi_data["official_cpi"].iloc[-1] if not cpi_data.empty else 100
        st.metric(
            label="Official CPI",
            value=f"{latest_cpi:.2f}",
            delta=f"{latest_cpi - 100:.2f}",
        )

    with col4:
        tracking_error = abs(current_index - latest_cpi)
        st.metric(
            label="Tracking Error",
            value=f"{tracking_error:.2f}",
            delta="vs Official CPI",
            delta_color="off",
        )

    st.divider()

    # Gauge charts row
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = create_nowcast_gauge(current_inflation, "Current Inflation")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Year-over-year inflation (mock)
        yoy_inflation = current_inflation * 1.2  # Mock calculation
        fig = create_nowcast_gauge(yoy_inflation, "YoY Inflation")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Core inflation (excluding food and energy - mock)
        core_inflation = current_inflation * 0.85
        fig = create_nowcast_gauge(core_inflation, "Core Inflation")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Time series comparison
    st.subheader("Nowcast vs Official CPI")
    fig = create_time_series_chart(price_data, cpi_data)
    st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Breakdown")
        fig = create_category_breakdown(price_data)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Category Heatmap")
        fig = create_heatmap(price_data)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Forecast section
    st.subheader("Inflation Forecast")

    # Generate mock forecast
    last_date = price_data["date"].max()
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=30),
        periods=forecast_horizon,
        freq="MS",
    ).tolist()

    # Simple forecast (mock)
    base_forecast = current_index
    forecast_values = [
        base_forecast + i * 0.15 + np.random.normal(0, 0.2)
        for i in range(forecast_horizon)
    ]

    fig = create_forecast_chart(price_data, forecast_values, forecast_dates)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("Forecast Details")
    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast": forecast_values,
            "Lower 95% CI": [v - 1.5 for v in forecast_values],
            "Upper 95% CI": [v + 1.5 for v in forecast_values],
        }
    )
    forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.divider()

    # Data quality metrics
    st.subheader("Data Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Observations", f"{len(price_data):,}")

    with col2:
        st.metric("Categories Tracked", len(selected_categories))

    with col3:
        st.metric("Data Freshness", "< 1 hour")

    with col4:
        st.metric("Coverage", "95%")

    # Footer
    st.divider()
    st.markdown(
        """
        ---
        **Inflation Nowcaster** | Built by Akram Boudouaour

        *Data is for demonstration purposes. Always refer to official sources for authoritative inflation data.*
        """
    )


if __name__ == "__main__":
    main()

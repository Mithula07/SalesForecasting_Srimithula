import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)
st.info(
    """
    This dashboard provides interactive sales analytics,
    demand forecasting, anomaly detection,
    and product demand segmentation.
    """
)

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    return df

df = load_data()

st.sidebar.title("📋 Dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "📈 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Demand Segments"
    ]
)
# ==============================
# PAGE 1 : SALES OVERVIEW
# ==============================

if page == "📈 Sales Overview":

    st.title("📈 Sales Overview Dashboard")

    st.info(
        "This dashboard provides an overview of historical sales performance "
        "using the Superstore sales dataset."
    )

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()

    # -----------------------------
    # KPI Cards
    # -----------------------------
    total_sales = df["Sales"].sum()
    total_orders = len(df)
    avg_order = df["Sales"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("🛒 Total Orders", total_orders)
    col3.metric("📦 Average Order Value", f"${avg_order:,.2f}")

    st.divider()

    # -----------------------------
    # Total Sales by Year
    # -----------------------------
    st.subheader("📊 Total Sales by Year")

    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        color="Sales",
        title="Year-wise Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Monthly Sales Trend
    # -----------------------------
    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        df.set_index("Order Date")
          .resample("M")["Sales"]
          .sum()
          .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Region & Category Filters
    # -----------------------------
    st.subheader("🔍 Filter Sales Data")

    c1, c2 = st.columns(2)

    region = c1.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    category = c2.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered_df = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True
    )

    # -----------------------------
    # Download Button
    # -----------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Filtered Data",
        csv,
        "filtered_sales.csv",
        "text/csv"
    )
elif page == "🔮 Forecast Explorer":

    st.title("🔮 Forecast Explorer")

    st.info("Explore future sales predictions generated using the XGBoost forecasting model.")

    # -------------------------
    # Forecast Data
    # -------------------------
    category_forecasts = {
        "Furniture": [9716.00, 6214.69, 16723.81],
        "Technology": [20370.97, 24370.32, 30328.62],
        "Office Supplies": [25796.03, 25957.26, 29761.83]
    }

    region_forecasts = {
        "East": [25088.46, 25353.45, 27580.83],
        "West": [11175.51, 15125.34, 21355.62]
    }

    forecast_type = st.radio(
        "Forecast Type",
        ["Category", "Region"],
        horizontal=True
    )

    horizon = st.slider(
        "Forecast Horizon",
        1,
        3,
        3
    )

    if forecast_type == "Category":
        selected = st.selectbox(
            "Select Category",
            ["Furniture", "Technology", "Office Supplies"]
        )
        forecast = category_forecasts[selected]

    else:
        selected = st.selectbox(
            "Select Region",
            ["East", "West"]
        )
        forecast = region_forecasts[selected]

    forecast = forecast[:horizon]

    forecast_df = pd.DataFrame({
        "Month": [f"Month {i}" for i in range(1, horizon + 1)],
        "Forecast Sales": forecast
    })

    st.subheader("Forecast Table")

    st.dataframe(
        forecast_df,
        width="stretch",
        hide_index=True
    )

    fig = px.line(
        forecast_df,
        x="Month",
        y="Forecast Sales",
        markers=True,
        title=f"{selected} Sales Forecast"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "13,915.32")
    c2.metric("RMSE", "18,893.85")
    c3.metric("MAPE", "13.29%")

    st.success("Best Performing Model : XGBoost")
# ==============================
# PAGE 3 : ANOMALY REPORT
# ==============================

elif page == "🚨 Anomaly Report":

    from sklearn.ensemble import IsolationForest

    st.title("🚨 Weekly Sales Anomaly Report")

    st.info(
        "Isolation Forest is used to identify unusual sales weeks."
    )

    weekly_sales = (
        df.set_index("Order Date")
          .resample("W")["Sales"]
          .sum()
          .reset_index()
    )

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = model.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    fig = px.line(
        weekly_sales,
        x="Order Date",
        y="Sales",
        title="Weekly Sales"
    )

    fig.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="Anomaly"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies[["Order Date", "Sales"]],
        hide_index=True,
        width="stretch"
    )

    st.metric(
        "Total Anomalies",
        len(anomalies)
    )

# ==============================
# PAGE 4 : PRODUCT SEGMENTS
# ==============================

elif page == "📦 Product Demand Segments":

    st.title("📦 Product Demand Segmentation")

    st.info(
        "K-Means clustering groups product sub-categories "
        "based on demand characteristics."
    )

    cluster_table = pd.DataFrame({

        "Sub Category":[
            "Accessories",
            "Appliances",
            "Art",
            "Binders",
            "Bookcases",
            "Chairs",
            "Copiers",
            "Envelopes",
            "Fasteners",
            "Furnishings",
            "Labels",
            "Machines",
            "Paper",
            "Phones",
            "Storage",
            "Supplies",
            "Tables"
        ],

        "Cluster":[
            0,2,2,0,2,0,1,2,2,2,2,3,2,0,0,2,0
        ]
    })

    cluster_names = {

        0:"High Volume Stable Demand",

        1:"Premium Products",

        2:"Growing Demand",

        3:"Specialized Products"

    }

    cluster_table["Demand Segment"] = cluster_table["Cluster"].map(
        cluster_names
    )

    st.subheader("Cluster Membership")

    st.dataframe(
        cluster_table,
        hide_index=True,
        width="stretch"
    )

    cluster_count = (
        cluster_table["Demand Segment"]
        .value_counts()
        .reset_index()
    )

    cluster_count.columns = [
        "Demand Segment",
        "Count"
    ]

    fig = px.bar(
        cluster_count,
        x="Demand Segment",
        y="Count",
        color="Demand Segment",
        title="Products in Each Demand Segment"
    )

    st.plotly_chart(fig, width="stretch")

    strategy = pd.DataFrame({

        "Demand Segment":[
            "High Volume Stable Demand",
            "Premium Products",
            "Growing Demand",
            "Specialized Products"
        ],

        "Recommended Stocking Strategy":[
            "Maintain higher inventory levels.",
            "Keep moderate stock and monitor demand.",
            "Gradually increase inventory.",
            "Maintain limited stock."
        ]
    })

    st.subheader("Recommended Stocking Strategy")

    st.table(strategy)
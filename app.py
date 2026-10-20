
from pathlib import Path
import pickle
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TXRH Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market Dashboard")
st.caption(
    "Explore historical market data and your TXRH Decision Tree model."
)

# ---------------- CUSTOM STYLE ----------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD MODEL ----------------
MODEL_FILES = [
    Path("txrh_decision_tree.pkl"),
    Path("txrh_decision_tree (1).pkl"),
]

@st.cache_resource
def load_model():
    for path in MODEL_FILES:
        if path.exists():
            with path.open("rb") as file:
                return pickle.load(file), path.name
    return None, None


model, model_filename = load_model()

with st.sidebar:
    st.header("⚙️ Dashboard Settings")

    ticker = st.text_input(
        "Stock ticker",
        value="TXRH",
        help="Example: TXRH, AAPL, MSFT",
    ).strip().upper()

    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )

    st.caption("Market data is supplied by Yahoo Finance.")
    st.caption("Data availability and delays may vary.")

# ---------------- DOWNLOAD STOCK DATA ----------------
@st.cache_data(ttl=300)
def get_stock_data(symbol, selected_period):
    data = yf.Ticker(symbol).history(
        period=selected_period,
        interval="1d",
        auto_adjust=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.reset_index()
    data.columns = [
        str(column).strip() for column in data.columns
    ]
    return data


if not ticker:
    st.warning("Enter a stock ticker in the sidebar.")
    st.stop()

with st.spinner(f"Loading {ticker} market data..."):
    try:
        data = get_stock_data(ticker, period)
    except Exception as error:
        st.error(f"Could not retrieve market data: {error}")
        st.info(
            "Check your internet connection and ticker symbol, "
            "then try again."
        )
        st.stop()

if data.empty:
    st.error(
        f"No historical data was returned for {ticker}. "
        "Check the ticker symbol and try again."
    )
    st.stop()

# ---------------- CLEAN DATA ----------------
for column in ["Open", "High", "Low", "Close",
               "Adj Close", "Volume"]:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

data = data.dropna(subset=["Close"])

if data.empty:
    st.error("No valid closing-price data is available.")
    st.stop()

latest = data.iloc[-1]
previous_close = (
    float(data.iloc[-2]["Close"])
    if len(data) > 1
    else float(latest["Close"])
)

latest_close = float(latest["Close"])
change = latest_close - previous_close
change_pct = (
    change / previous_close * 100
    if previous_close else 0
)

# ---------------- KEY METRICS ----------------
st.subheader(f"{ticker} Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest Close",
    f"{latest_close:,.2f}",
    f"{change:+.2f} ({change_pct:+.2f}%)",
)
c2.metric(
    "Day High",
    f"{float(latest['High']):,.2f}"
    if pd.notna(latest.get("High"))
    else "N/A",
)
c3.metric(
    "Day Low",
    f"{float(latest['Low']):,.2f}"
    if pd.notna(latest.get("Low"))
    else "N/A",
)
c4.metric(
    "Volume",
    f"{int(latest['Volume']):,}"
    if pd.notna(latest.get("Volume"))
    else "N/A",
)

st.caption(
    f"Most recent data date: {latest['Date']}"
)

# ---------------- PRICE CHART ----------------
st.subheader("📊 Historical Closing Price")

price_fig = px.line(
    data,
    x="Date",
    y="Close",
    title=f"{ticker} Closing Price",
    labels={
        "Date": "Date",
        "Close": "Closing Price (USD)",
    },
)
price_fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Price",
)
st.plotly_chart(price_fig, use_container_width=True)

# ---------------- VOLUME CHART ----------------
if "Volume" in data.columns:
    st.subheader("📦 Trading Volume")

    volume_fig = px.bar(
        data,
        x="Date",
        y="Volume",
        title=f"{ticker} Daily Trading Volume",
        labels={"Volume": "Shares traded"},
    )
    st.plotly_chart(
        volume_fig,
        use_container_width=True,
    )

# ---------------- MODEL PREDICTION ----------------
st.divider()
st.subheader("🤖 Decision Tree Model")

if ticker != "TXRH":
    st.info(
        "Your saved model was built for the TXRH project. "
        "Select TXRH to use it. Other tickers can be charted, "
        "but require an appropriately trained model for prediction."
    )

elif model is None:
    st.warning(
        "Model file not found. Add either "
        "'txrh_decision_tree.pkl' or "
        "'txrh_decision_tree (1).pkl' beside app.py "
        "in your GitHub repository."
    )

else:
    st.success(f"Loaded model: {model_filename}")

    # Feature names and order must match model training.
    available_features = {
        "Open": latest.get("Open"),
        "High": latest.get("High"),
        "Low": latest.get("Low"),
        "Close": latest.get("Close"),
        "Adj Close": latest.get(
            "Adj Close", latest.get("Close")
        ),
        "Volume": latest.get("Volume"),
    }

    # Support models trained with either capitalization style.
    feature_aliases = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }

    expected_features = getattr(
        model, "feature_names_in_", None
    )

    if expected_features is not None:
        feature_names = list(expected_features)
    else:
        feature_names = [
            "Open", "High", "Low",
            "Close", "Adj Close", "Volume"
        ]

    input_values = {}
    missing_features = []

    for feature in feature_names:
        canonical = feature_aliases.get(
            str(feature).strip().lower()
        )

        if canonical is None:
            missing_features.append(feature)
        else:
            value = available_features.get(canonical)
            if value is None or pd.isna(value):
                missing_features.append(feature)
            else:
                input_values[feature] = float(value)

    if missing_features:
        st.warning(
            "The model expects features that this dashboard "
            f"cannot supply: {missing_features}. "
            "Update the feature mapping to match your training code."
        )
    else:
        input_frame = pd.DataFrame(
            [input_values],
            columns=feature_names,
        )

        try:
            prediction = model.predict(input_frame)[0]

            st.write("Model output class:")
            st.subheader(str(prediction))

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    input_frame
                )[0]

                probability_data = pd.DataFrame({
                    "Class": [
                        str(label)
                        for label in model.classes_
                    ],
                    "Model probability (%)": [
                        round(float(p) * 100, 2)
                        for p in probabilities
                    ],
                })

                st.write("Model class probabilities")
                st.dataframe(
                    probability_data,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "The output is a model prediction, not a "
                "guaranteed future price movement. Confirm "
                "the training target's class meanings before "
                "interpreting the output."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ---------------- DATA TABLE ----------------
st.divider()
st.subheader("📋 Historical Market Data")

display_columns = [
    column for column in [
        "Date", "Open", "High", "Low",
        "Close", "Adj Close", "Volume"
    ]
    if column in data.columns
]

st.dataframe(
    data[display_columns].sort_values(
        "Date", ascending=False
    ),
    use_container_width=True,
    hide_index=True,
)

csv_data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Historical Data (CSV)",
    data=csv_data,
    file_name=f"{ticker}_historical_data.csv",
    mime="text/csv",
)

st.caption(
    "Educational project only. Historical data and model "
    "outputs do not guarantee future results."
)

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Based Stock Price Prediction System")
st.write("Predict stock closing prices using Machine Learning.")

# -----------------------------
# STOCK DROPDOWN
# -----------------------------
stocks = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ITC": "ITC.NS"
}

selected_stock_name = st.selectbox(
    "Select a Stock",
    list(stocks.keys())
)

stock_symbol = stocks[selected_stock_name]


# -----------------------------
# DOWNLOAD DATA FUNCTION
# -----------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(
        symbol,
        start="2020-01-01",
        end="2024-01-01",
        auto_adjust=True
    )

    # Fix multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


# Load data
data = load_data(stock_symbol)

st.success("✅ Data Loaded Successfully!")


# Show raw data
with st.expander("View Raw Data"):
    st.dataframe(data.tail())


# -----------------------------
# PREPARE ML DATA
# -----------------------------
features = ["Open", "High", "Low", "Volume"]

X = data[features]
y = data["Close"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL BUTTON
# -----------------------------
if st.button("🚀 Train AI Model"):

    with st.spinner("Training model... Please wait"):

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    st.success("✅ Model Trained Successfully!")

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.metric("Model Accuracy (R2 Score)", f"{r2:.2f}")
    col2.metric("Average Error (₹)", f"{mae:.2f}")

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("📊 Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(y_test.values, label="Actual")
    ax.plot(predictions, label="Predicted")

    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # NEXT DAY PREDICTION
    # -----------------------------
    latest_data = data.iloc[-1]

    next_features = [[
        latest_data["Open"],
        latest_data["High"],
        latest_data["Low"],
        latest_data["Volume"]
    ]]

    next_price = model.predict(next_features)

    st.subheader("📈 Next Day Prediction")

    st.info(f"Predicted Closing Price: ₹{next_price[0]:.2f}")
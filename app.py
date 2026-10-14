import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="AI Stock Forecast | LSTM", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    h1, h2, h3 {color: #183153;}
    div[data-testid="stMetric"] {
        background: white; padding: 14px 16px; border-radius: 12px;
        border: 1px solid #e6ebf2;
    }
    .note {background:#eef5ff;padding:12px 16px;border-radius:10px;border-left:4px solid #2f6fed;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")
st.caption("Tải dữ liệu lịch sử · Huấn luyện mô hình · Dự báo 10 phiên tiếp theo")

DEFAULT_TICKERS = {
    "Vietcombank (VCB)": "VCB.VN",
    "Vinamilk (VNM)": "VNM.VN",
    "FPT": "FPT.VN",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Nhập mã khác": "CUSTOM",
}

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                        progress=False, threads=False)
    if data is None or data.empty:
        return pd.DataFrame()
    # yfinance may return MultiIndex columns depending on version/ticker
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = data.iloc[:, 0]
    else:
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    out = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    out = out.dropna()
    out.index = pd.to_datetime(out.index)
    return out[out["Close"] > 0]

def build_model(window: int, units: int, learning_rate: float):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, return_sequences=True),
        Dropout(0.15),
        LSTM(max(16, units // 2)),
        Dropout(0.10),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model

def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i, 0])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y

with st.sidebar:
    st.header("⚙️ Thiết lập")
    selected_name = st.selectbox("Mã cổ phiếu", list(DEFAULT_TICKERS.keys()), index=0)
    if selected_name == "Nhập mã khác":
        ticker = st.text_input("Ticker Yahoo Finance", value="GOOGL").strip().upper()
    else:
        ticker = DEFAULT_TICKERS[selected_name]
    period_label = st.selectbox("Khoảng dữ liệu lịch sử", ["2 năm", "5 năm", "10 năm"], index=1)
    period = {"2 năm": "2y", "5 năm": "5y", "10 năm": "10y"}[period_label]
    window = st.slider("Số phiên dùng làm đầu vào", min_value=20, max_value=120, value=60, step=10)
    units = st.select_slider("Độ lớn mô hình LSTM", options=[16, 32, 48, 64], value=32)
    epochs = st.slider("Số vòng huấn luyện tối đa (epochs)", 5, 60, 20, 5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64], value=32)
    learning_rate = st.selectbox("Learning rate", [0.0005, 0.001, 0.002], index=1)
    run = st.button("🚀 Huấn luyện & chạy dự báo", type="primary", use_container_width=True)
    st.caption("Lần đầu chạy có thể mất vài phút để tải dữ liệu và huấn luyện.")

st.markdown(
    '<div class="note">Mô hình dùng giá đóng cửa đã điều chỉnh (Adjusted Close/Close) theo dữ liệu Yahoo Finance. '
    'Dự báo là ước lượng tham khảo, không phải lời khuyên đầu tư.</div>',
    unsafe_allow_html=True
)

if run:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    with st.spinner(f"Đang tải dữ liệu {ticker}..."):
        raw = download_prices(ticker, period)

    if raw.empty:
        st.error(
            f"Không tải được dữ liệu cho mã **{ticker}**. Hãy kiểm tra mã trên Yahoo Finance. "
            "Một số mã Việt Nam có thể không có dữ liệu hoặc có hậu tố khác; bạn có thể thử AAPL hoặc MSFT để kiểm tra ứng dụng."
        )
        st.stop()

    if len(raw) < window + 40:
        st.error(f"Chỉ có {len(raw)} dòng dữ liệu, chưa đủ để huấn luyện ổn định. Hãy chọn khoảng lịch sử dài hơn hoặc giảm số phiên đầu vào.")
        st.stop()

    prices = raw[["Close"]].astype(float)
    split_idx = int(len(prices) * 0.80)
    train_prices = prices.iloc[:split_idx]
    if len(train_prices) <= window:
        st.error("Dữ liệu huấn luyện quá ngắn so với số phiên đầu vào.")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices.values)
    scaled_all = scaler.transform(prices.values)
    X_all, y_all = make_sequences(scaled_all, window)
    # The target corresponding to sequence i is price row i+window.
    target_positions = np.arange(window, len(prices))
    train_mask = target_positions < split_idx
    test_mask = target_positions >= split_idx

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    if len(X_train) == 0 or len(X_test) < 2:
        st.error("Không đủ dữ liệu sau khi chia tập. Hãy chọn khoảng lịch sử dài hơn.")
        st.stop()

    model = build_model(window, units, learning_rate)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    progress = st.progress(0, text="Đang huấn luyện mô hình LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )
    progress.progress(100, text="Hoàn tất huấn luyện.")

    pred_scaled = model.predict(X_test, verbose=0)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_actual = scaler.inverse_transform(pred_scaled).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_test_actual, pred_actual)))
    mae = float(mean_absolute_error(y_test_actual, pred_actual))
    r2 = float(r2_score(y_test_actual, pred_actual)) if len(y_test_actual) >= 2 else float("nan")
    mape = float(np.mean(np.abs((y_test_actual - pred_actual) / np.maximum(np.abs(y_test_actual), 1e-8))) * 100)

    # Recursive forecast for 10 future trading sessions (not calendar days).
    sequence = scaled_all[-window:, 0].astype(np.float32).tolist()
    future_scaled = []
    for _ in range(10):
        x = np.asarray(sequence[-window:], dtype=np.float32).reshape(1, window, 1)
        next_value = float(model.predict(x, verbose=0)[0, 0])
        future_scaled.append(next_value)
        sequence.append(next_value)
    future_prices = scaler.inverse_transform(np.asarray(future_scaled).reshape(-1, 1)).ravel()
    last_date = prices.index[-1]
    future_dates = []
    candidate = last_date
    while len(future_dates) < 10:
        candidate += pd.Timedelta(days=1)
        if candidate.weekday() < 5:
            future_dates.append(candidate)
    forecast_df = pd.DataFrame({
        "Phiên dự báo": [f"Ngày {i}" for i in range(1, 11)],
        "Ngày dự kiến": [d.strftime("%d/%m/%Y") for d in future_dates],
        "Giá dự báo": future_prices
    })
    forecast_df["Giá dự báo"] = forecast_df["Giá dự báo"].round(2)

    st.subheader(f"📊 Kết quả đánh giá mô hình — {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R² Score", f"{r2:.3f}")
    c4.metric("MAPE", f"{mape:.2f}%")
    st.caption("Các chỉ số được tính trên tập kiểm tra theo thứ tự thời gian; R² có thể âm nếu dự báo kém hơn đường chuẩn trung bình.")

    st.subheader(f"🚀 Dự báo 10 phiên tiếp theo cho {ticker}")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    csv = forecast_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Tải bảng dự báo CSV", data=csv, file_name=f"forecast_{ticker.replace('.', '_')}.csv", mime="text/csv")

    st.subheader("📉 Biểu đồ giá thực tế và dự báo trên tập kiểm tra")
    test_dates = prices.index[target_positions[test_mask]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(test_dates, y_test_actual, label="Giá thực tế", linewidth=1.8)
    ax.plot(test_dates, pred_actual, label="Giá mô hình dự báo", linewidth=1.8)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.set_title(f"{ticker} — Giá thực tế vs. dự báo trên tập kiểm tra")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("🔭 Dự báo tương lai")
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    recent = prices.tail(90)
    ax2.plot(recent.index, recent["Close"], label="Giá lịch sử", linewidth=1.8)
    ax2.plot(future_dates, future_prices, label="Dự báo 10 phiên", marker="o", linewidth=2)
    ax2.axvline(prices.index[-1], linestyle="--", alpha=0.5, label="Phiên cuối có dữ liệu")
    ax2.set_xlabel("Ngày")
    ax2.set_ylabel("Giá")
    ax2.set_title(f"{ticker} — Lịch sử gần đây và dự báo")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("Xem đường loss khi huấn luyện"):
        fig3, ax3 = plt.subplots(figsize=(9, 3.5))
        ax3.plot(history.history.get("loss", []), label="Training loss")
        ax3.plot(history.history.get("val_loss", []), label="Validation loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("MSE loss")
        ax3.legend()
        ax3.grid(True, alpha=0.25)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.success("Đã hoàn tất. Bạn có thể đổi mã cổ phiếu ở thanh bên và chạy lại.")
else:
    st.info("Chọn mã cổ phiếu ở thanh bên, điều chỉnh tham số nếu cần, rồi nhấn **Huấn luyện & chạy dự báo**.")
    st.markdown("### Các chỉ số được sử dụng")
    m1, m2, m3 = st.columns(3)
    m1.markdown("**RMSE**  \nCăn bậc hai của sai số bình phương trung bình; càng thấp thường càng tốt.")
    m2.markdown("**MAE**  \nSai số tuyệt đối trung bình, cùng đơn vị với giá cổ phiếu.")
    m3.markdown("**R² Score**  \nĐánh giá mức độ mô hình giải thích biến thiên của dữ liệu; có thể âm.")
    st.markdown("### Quy trình hoạt động")
    st.write("1. Lấy dữ liệu giá lịch sử từ Yahoo Finance.")
    st.write("2. Chuẩn hóa giá và tạo chuỗi gồm N phiên làm đầu vào.")
    st.write("3. Huấn luyện LSTM trên dữ liệu quá khứ, kiểm tra trên phần dữ liệu mới hơn.")
    st.write("4. Dự báo lặp 10 phiên tiếp theo và hiển thị bảng, biểu đồ, chỉ số đánh giá.")

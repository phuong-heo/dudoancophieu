# Website dự báo giá cổ phiếu bằng AI LSTM

## Chạy trực tuyến (không cần cài Python trên máy)
1. Tải file ZIP này và giải nén.
2. Tạo repository mới trên GitHub.
3. Tải `app.py` và `requirements.txt` lên repository.
4. Mở https://share.streamlit.io/ và đăng nhập bằng GitHub.
5. Chọn repository, branch `main`, main file path `app.py`, rồi nhấn Deploy.
6. Chờ cài thư viện và mở đường link `.streamlit.app`.

## Chạy trên Google Colab (dùng để thử nghiệm)
Streamlit cần một URL truy cập được từ trình duyệt. Cách triển khai ổn định hơn là Streamlit Community Cloud. Colab không phải nền tảng hosting web lâu dài.

## Mã cổ phiếu
Mã dùng theo quy ước Yahoo Finance. Mã Việt Nam có thể không có dữ liệu hoặc hậu tố có thể thay đổi. Nếu VCB.VN không tải được, hãy thử mã quốc tế như AAPL hoặc MSFT để xác minh ứng dụng, sau đó kiểm tra ticker chính xác trên Yahoo Finance.

## Nội dung
- Tải dữ liệu giá đã điều chỉnh bằng yfinance
- Mô hình LSTM hai tầng
- Đánh giá theo tập kiểm tra theo thứ tự thời gian: RMSE, MAE, R², MAPE
- Dự báo lặp 10 phiên tiếp theo
- Biểu đồ, loss curve và tải CSV

## Lưu ý học thuật
Đây là mô hình minh họa cho mục đích học tập, không phải lời khuyên đầu tư. Dự báo giá cổ phiếu có độ bất định cao. Các ngày dự báo chỉ bỏ qua thứ Bảy và Chủ nhật, chưa loại trừ ngày nghỉ lễ của từng sàn.

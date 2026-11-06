# RULES.md — BỘ QUY TẮC DỰ ÁN PRACTICE 3: HUGGING FACE

Bộ quy tắc này là bắt buộc đối với toàn bộ quá trình phát triển mã nguồn, thực nghiệm và lập tài liệu cho dự án **Practice 3 — Get Started with Hugging Face (Sentiment Analysis & Fine-tuning DistilBERT)**.

---

## 1. Quy Tắc Khoa Học và Tính Trung Thực Học Thuật

1. **Tuyệt đối không bịa đặt số liệu (No Hallucination):** Không được tự tạo hoặc ước lượng số liệu độ chính xác (accuracy), hàm mất mát (loss), thời gian chạy hoặc kết quả kiểm thử. Toàn bộ số liệu trong báo cáo phải được trích xuất tự động từ các tệp nhật ký thực tế (`outputs/reports/final_metrics.json`, `classification_report.csv`, `confusion_matrix.csv`).
2. **Không chỉnh sửa số liệu:** Không sửa đổi thủ công kết quả để đạt mục tiêu định trước.
3. **Ngăn chặn rò rỉ dữ liệu (No Data Leakage):**
   - Phân chia tập Train (800 mẫu), Validation (200 mẫu) và Test (200 mẫu) độc lập nghiêm ngặt bằng `.shuffle(seed=42).select(range(...))` từ các phân vùng chuẩn của IMDb, đảm bảo không trùng lặp chỉ số.
   - Tập Validation và Test tuyệt đối không tham gia vào quá trình tính gradient cập nhật trọng số.
4. **Sử dụng Test Set đúng chuẩn:** Tuyệt đối không dùng Test Set để chọn siêu tham số, điều chỉnh ngưỡng hay early stopping. Toàn bộ khảo sát và đánh giá trên Test Set chỉ được mở ra duy nhất một lần ở Bước 7 (Final Evaluation).
5. **Đối chứng khoa học chuẩn mực:**
   - Phân biệt rõ ràng giữa đối chứng cùng tập kiểm thử (Fine-tuned 74.50% vs Majority-Class 52.00% trên Test Set: tăng trưởng thực chất **+22.50 điểm %**) và đối chứng tham chiếu chéo (Untrained Head 44.00% trên Validation Set).
   - Diễn giải thống kê thận trọng: Giá trị p-value trong kiểm định Fisher phản ánh việc chưa đủ bằng chứng thống kê để khẳng định sự khác biệt trên mẫu 200 bài kiểm thử, không vội vàng quy kết rằng hiện tượng cắt cụt token hoàn toàn không ảnh hưởng đến độ chính xác.
6. **Tính tái lập (Reproducibility):** Cố định hạt giống ngẫu nhiên (`seed=42`) cho mọi khâu (torch, numpy, transformers `set_seed`) và lưu trữ tệp đối chứng dự đoán chi tiết (`outputs/reports/predictions.csv`).

---

## 2. Quy Tắc Viết Mã Nguồn (Code Style & Engineering)

1. **Phiên bản & Thư viện:** Python $\ge 3.10$. Sử dụng hệ sinh thái Hugging Face chuẩn: `transformers`, `datasets`, `evaluate`, `accelerate`, `torch`.
2. **Kiểu dữ liệu & Tài liệu hóa:** Mọi hàm và phương thức bắt buộc có type hints đầy đủ và docstrings ngắn gọn, súc tích.
3. **Tính di động (Portability):**
   - Không hard-code đường dẫn tuyệt đối của máy cá nhân. Mọi đường dẫn phải dựa trên `pathlib.Path` tương đối với thư mục gốc dự án.
   - Tự động nhận diện thiết bị tính toán: CUDA $\to$ MPS $\to$ CPU.
4. **Bảo mật và an toàn dữ liệu:**
   - Sử dụng định dạng an toàn `safetensors` cho trọng số mô hình đã fine-tune.
   - Không đưa API keys, token cá nhân hoặc file `.env` vào kho lưu trữ Git.

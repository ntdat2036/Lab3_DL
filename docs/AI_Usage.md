# BÁO CÁO SỬ DỤNG CÔNG CỤ TRÍ TUỆ NHÂN TẠO (AI USAGE REPORT)

## 1. Mục Đích Sử Dụng AI Trong Dự Án

Việc ứng dụng công cụ Trí tuệ Nhân tạo (AI) trong dự án **Practice 3 — Get Started with Hugging Face** nhằm các mục đích sau:
- Hỗ trợ xây dựng cấu trúc dự án chuẩn kỹ thuật phần mềm (Clean Architecture, Modularization).
- Rà soát tính khoa học trong phương pháp luận thực nghiệm (ngăn chặn rò rỉ dữ liệu, đảm bảo tính tái lập với `seed=42`).
- Tự động hóa việc sinh các đoạn mã mẫu (boilerplate code) cho Hugging Face `pipeline`, `Trainer`, `DataCollatorWithPadding`, và tài liệu học thuật.
- Giải thích chi tiết cơ chế hoạt động của Transformer (Self-Attention, WordPiece Tokenizer, Logits, Softmax) nhằm nâng cao hiểu biết lý thuyết của sinh viên.

---

## 2. Các Công Cụ và Mô Hình AI Đã Sử Dụng

| Công cụ / Mô hình | Nhà phát triển | Phiên bản / Môi trường | Mục đích sử dụng |
|---|---|---|---|
| **Gemini 3.8 Flash (Antigravity IDE)** | Google DeepMind | Antigravity AI Coding Assistant | Trợ lý lập trình cặp (Pair-programming), tạo tài liệu, sinh mã nguồn bài tập |

---

## 3. Phạm Vi Hỗ Trợ của AI

AI được sử dụng trong các phạm vi cụ thể:
1. **Exercise 1 — Sentiment Analysis with Hugging Face:**
   - Hướng dẫn nạp mô hình pretrained SST-2 từ Hugging Face Hub.
   - Viết hàm phân tích sâu Tokenizer (WordPiece sub-words, input_ids, attention_mask) và suy luận thủ công từ Logits qua Softmax.
2. **Exercise 2 — Fine-tuning DistilBERT on IMDb:**
   - Xây dựng quy trình 7 bước chuẩn: nạp dataset `imdb`, tiền xử lý với `DataCollatorWithPadding`, cấu hình `TrainingArguments`, huấn luyện với `Trainer`, và tính toán bộ chỉ số đa chiều (Accuracy, Precision, Recall, F1, Confusion Matrix).
3. **Phân tích lỗi (Error Analysis):**
   - Viết code lọc và trích xuất các câu review phim bị dự đoán sai có độ tự tin cao nhất vào file `outputs/reports/error_analysis.csv`.

---

## 4. Sự Đóng Góp của Sinh Viên (Student Contributions)

> *Quy định bắt buộc: Mục này do sinh viên tự điền theo đánh giá cá nhân.*

[sinh viên điền: Liệt kê chi tiết những phần việc, ý tưởng, phân tích hoặc quyết định kỹ thuật do sinh viên trực tiếp thực hiện trong dự án Practice 3]

---

## 5. Các Chỉnh Sửa và Kiểm Chứng của Sinh Viên (Student Modifications & Verification)

> *Quy định bắt buộc: Mục này do sinh viên tự điền theo đánh giá cá nhân.*

[sinh viên điền: Liệt kê các đoạn mã nguồn hoặc tài liệu đã được sinh viên tự tay rà soát, kiểm chứng tính đúng đắn, sửa lỗi logic hoặc tối ưu lại so với gợi ý ban đầu của AI]

---

## 6. Đánh Giá Tính Chuẩn Xác và Đạo Đức Học Thuật

1. **Kiểm chứng độc lập:** Toàn bộ code sinh bởi AI đều đã chạy thử nghiệm thực tế trên môi trường máy tính trước khi được ghi nhận.
2. **Tuyệt đối không làm giả dữ liệu:** Các số liệu về Loss (0.5565), Accuracy (74.50%), F1-score (74.27%) và biểu đồ trong báo cáo được trích xuất trực tiếp từ các file kết quả thực tế tại `outputs/reports/final_metrics.json`.
3. **Bảo mật và an toàn mã nguồn:** Không đưa thông tin bảo mật, khóa API hoặc token cá nhân lên Git.

---

## 7. Bài Học Kinh Nghiệm Khi Làm Việc Với AI

- **AI là công cụ hỗ trợ, con người giữ quyền kiểm soát:** Cần cung cấp đề bài rõ ràng ngay từ đầu (như ảnh chụp đề bài hoặc slide) để tránh AI tự phỏng đoán nhầm đề tài.
- **Tập trung vào bản chất:** Sử dụng AI để tự động hóa các thao tác lặp lại (boilerplate) nhưng sinh viên phải nắm vững bản chất kiến trúc Transformer và cơ chế Tokenizer bên dưới.

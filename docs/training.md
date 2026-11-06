# Training Pipeline & Fine-tuning Configuration

## Quy Trình 7 Bước Tinh Chỉnh Mô Hình & Thiết Lập Siêu Tham Số

---

## 1. Quy Trình 7 Bước Chuẩn Mực (The 7-Step Fine-Tuning Workflow)

Quy trình fine-tuning mô hình ngôn ngữ lớn trên bộ dữ liệu IMDb được chuẩn hóa và thực thi theo 7 bước:

```mermaid
graph TD
    S1["Bước 1: Cài đặt thư viện & thiết lập môi trường"] --> S2["Bước 2: Tải & phân chia dữ liệu độc lập (seed=42)"]
    S2 --> S3["Bước 3: Nạp Base Model & khởi tạo Classification Head"]
    S3 --> S4["Bước 4: Tokenization & Dynamic Padding (DataCollator)"]
    S4 --> S5["Bước 5: Cấu hình siêu tham số (TrainingArguments)"]
    S5 --> S6["Bước 6: Đo Untrained Baseline & Fine-tune qua Trainer"]
    S6 --> S7["Bước 7: Đánh giá trên Test Set & Phân tích lỗi"]
```

### Chi tiết triển khai từng bước trong mã nguồn (`src/02_train_finetune.py`):

1. **Bước 1 — Môi trường & Thiết bị:** Cài đặt các gói `transformers`, `datasets`, `evaluate`, `accelerate`, `torch`. Tự động nhận diện thiết bị tính toán (`cuda` $\to$ `mps` $\to$ `cpu`) và khóa hạt giống ngẫu nhiên (`seed=42`).
2. **Bước 2 — Nạp & Phân chia IMDb:** Tải dữ liệu từ `stanfordnlp/imdb`. Phân chia không rò rỉ: 800 Train / 200 Val / 200 Test. Xác lập mốc Majority-Class Baseline (52.00%).
3. **Bước 3 — Khởi tạo Mô hình:** Nạp `distilbert/distilbert-base-uncased` với `num_labels=2`. Khởi tạo nhãn phân loại ánh xạ `0: NEGATIVE, 1: POSITIVE`.
4. **Bước 4 — Tiền xử lý Dữ liệu:** Tokenize văn bản với `truncation=True, max_length=128`. Sử dụng `DataCollatorWithPadding` để thực hiện Dynamic Padding tối ưu.
5. **Bước 5 — Cấu hình Siêu tham số:** Khởi tạo `TrainingArguments` với tốc độ học nhỏ và cơ chế lưu best model.
6. **Bước 6 — Baseline & Huấn luyện:** Đo độ chính xác mô hình trước huấn luyện trên tập Validation (đạt 44.00%). Khởi chạy `trainer.train()` qua 2 epochs, tự động lưu checkpoint có validation accuracy cao nhất vào `outputs/final_model/`.
7. **Bước 7 — Đánh giá Toàn diện:** Đánh giá trên tập Test độc lập, tính toán khoảng tin cậy Wilson 95%, xuất ma trận nhầm lẫn, đồ thị phân tích và bảng danh sách ca dự đoán sai.

---

## 2. Bảng Cấu Hình Siêu Tham Số (Hyperparameter Specifications)

| Siêu tham số | Giá trị lựa chọn | Cơ sở lý thuyết & Thực nghiệm |
|:---|:---:|:---|
| **Thuật toán tối ưu (Optimizer)** | **AdamW** | Biến thể Adam tích hợp Weight Decay tách rời, ngăn ngừa overfitting hiệu quả trên Transformer. |
| **Tốc độ học (Learning Rate)** | **$2 \times 10^{-5}$** | Tốc độ học chuẩn cho Fine-tuning Transformer (tránh Catastrophic Forgetting — hiện tượng xóa sạch tri thức pre-trained). |
| **Kích thước Batch (`per_device_train_batch_size`)** | **16** | Cân bằng hoàn hảo giữa độ ổn định gradient và bộ nhớ RAM của CPU. |
| **Số lượng Epochs (`num_train_epochs`)** | **2** | Tổng cộng 100 optimization steps (800 mẫu / 16 = 50 steps/epoch $\times$ 2). Đạt mức cải thiện rõ rệt trên tập validation (từ 44.00% lên 76.50%) trong giới hạn thời gian CPU (~6.5 phút), chưa phát hiện dấu hiệu quá khớp nghiêm trọng trên đường cong loss. |
| **Hệ số suy giảm trọng số (`weight_decay`)** | **0.01** | Regularization chuẩn $L_2$ áp dụng cho các ma trận trọng số (loại trừ bias và LayerNorm). |
| **Độ dài cắt cụt tối đa (`max_length`)** | **128 tokens** | Tối ưu hóa thời gian chạy CPU; bao phủ đầy đủ ngữ cảnh mở đầu bài đánh giá. |
| **Chiến lược đánh giá (`eval_strategy`)** | `"epoch"` | Đánh giá lại mô hình trên tập Validation ở cuối mỗi epoch. |
| **Tiêu chí chọn mô hình (`metric_for_best_model`)** | `"accuracy"` | Chọn checkpoint đạt độ chính xác cao nhất trên tập Validation. |
| **Tự động nạp Best Model (`load_best_model_at_end`)** | `True` | Đảm bảo mô hình cuối cùng được đánh giá là phiên bản có chất lượng tốt nhất. |
| **Hạt giống ngẫu nhiên (`seed`)** | **42** | Cố định hạt giống cho torch, numpy, transformers nhằm hỗ trợ tái lập thực nghiệm trong cùng môi trường. |

---

## 3. Hàm Mục Tiêu & Tối Ưu Hóa (Loss Function & Optimization)

Mô hình sử dụng hàm mất mát **Cross-Entropy Loss** cho bài toán phân loại nhị phân:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^1 y_{i,c} \log \hat{y}_{i,c}$$

Trong đó:
- $N = 16$ là kích thước mini-batch.
- $y_{i,c} \in \{0, 1\}$ là nhãn one-hot thực tế của mẫu $i$.
- $\hat{y}_{i,c} = P(y = c \mid x_i)$ là xác suất dự đoán sau hàm Softmax.

### Lịch trình Tốc độ Học (Learning Rate Schedule):
- Sử dụng **Linear Decay Schedule**: Tốc độ học giảm dần tuyến tính từ $2 \times 10^{-5}$ về $0$ trong suốt 100 bước tối ưu.

---

## 4. Tối Ưu Hóa Kỹ Thuật (Engineering Optimizations)

1. **Tính toán Chỉ số Ngoại tuyến Xác định (Deterministic Metric Evaluation):**
   - Trong `src/02_train_finetune.py`, các chỉ số đánh giá (Accuracy, Macro-F1, Precision, Recall) được tính toán trực tiếp thông qua `scikit-learn` trong `compute_metrics`.
   - Giúp quá trình đánh giá hoàn toàn xác định, không phụ thuộc vào tải mã động từ Hugging Face Hub qua mạng, loại bỏ độ trễ và khả năng lỗi kết nối.
2. **Dynamic Padding với `DataCollatorWithPadding`:**
   - Đệm động từng batch về độ dài dài nhất trong batch đó thay vì đệm tĩnh 128 tokens cố định toàn bộ dữ liệu, giảm thiểu các phép nhân ma trận chú ý thừa với token `[PAD]`.
3. **Định Dạng Lưu Trữ An Toàn (`SafeTensors`):**
   - Trọng số xuất xưởng được ghi dưới dạng `model.safetensors`, khắc phục triệt để lỗ hổng bảo mật thực thi mã tùy ý vốn có của định dạng cũ `pickle` (`pytorch_model.bin`).

---

## 5. Thời Gian & Hiệu Năng Thực Thi Thực Tế

- **Môi trường đo đạc:** Bộ xử lý thông thường (CPU đa nhân x86_64, Windows, PyTorch 2.15.0+cpu).
- **Thời gian huấn luyện 2 epochs (100 steps):** **391.42 giây (~6.52 phút)** trên CPU.
- **Kích thước tệp mô hình xuất xưởng:**
  - `model.safetensors`: ~268 MB.
  - Bộ từ điển & cấu hình: ~1.5 MB.
- **Thư mục lưu trữ:**
  - Checkpoints trung gian: `outputs/checkpoints/`
  - Mô hình hoàn thiện tốt nhất: `outputs/final_model/`

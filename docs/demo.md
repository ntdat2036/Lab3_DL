# Demonstration & Quickstart Guide

## Hướng Dẫn Chạy Thử Nghiệm, Suy Luận Tương Tác & Tái Lập Kết Quả

---

## 1. Yêu Cầu Môi Trường & Cài Đặt (Prerequisites & Setup)

Dự án yêu cầu môi trường **Python $\ge 3.10$**.

```bash
# 1. Di chuyển vào thư mục dự án
cd DL_lab3-main

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

Danh mục các thư viện chính trong [requirements.txt](../requirements.txt):
- `torch >= 2.0.0`
- `transformers >= 4.36.0`
- `datasets >= 2.16.0`
- `evaluate >= 0.4.1`
- `accelerate >= 0.25.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.8.0`
- `pandas >= 2.0.0`

---

## 2. Demo 1: Suy Luận Với Mô Hình Pretrained (Exercise 1)

Chạy kịch bản phân tích cảm xúc nhanh và mổ xẻ cơ chế Tokenizer:

```bash
python src/01_sentiment_analysis.py
```

### Nội dung chương trình thực thi:
1. **Pipeline Inference:** Nạp mô hình `distilbert/distilbert-base-uncased-finetuned-sst-2-english` và dự đoán tức thì câu mẫu.
2. **Tokenizer Deep Dive:** Bóc tách chuỗi thành các sub-word tokens WordPiece, in danh sách `input_ids` kèm các mã đặc biệt `[CLS]` (101) và `[SEP]` (102), hiển thị vector `attention_mask`.
3. **Manual Softmax Verification:** Thực hiện forward pass thủ công qua PyTorch, tính Softmax trên raw logits và đối chiếu sai số với đầu ra của `pipeline` (độ lệch $< 10^{-5}$).
4. **Kiểm thử 5 câu mẫu đa dạng:**
   - Câu tích cực: *"The cinematography in this film was breathtaking and poetic."*
   - Câu tiêu cực: *"The plot was predictable and the pacing felt excruciatingly slow."*
   - Câu có từ phủ định: *"I did not dislike the film, but it was not extraordinary either."*
   - Câu mỉa mai: *"Oh brilliant, another two hours of my life wasted on sheer nonsense!"*
   - Câu cảm xúc hỗn hợp: *"The visual effects were stunning, although the script was weak."*

---

## 3. Demo 2: Huấn Luyện Fine-tuning Mô Hình (Exercise 2)

Thực thi toàn bộ quy trình 7 bước fine-tuning trên bộ dữ liệu IMDb:

```bash
python src/02_train_finetune.py
```

### Tiến trình tự động diễn ra:
1. Tải và phân chia tập `stanfordnlp/imdb` (800 Train / 200 Val / 200 Test).
2. Đo lường mốc Majority-Class Baseline (52.00%).
3. Đo lường Untrained Baseline trên tập Validation (44.00%).
4. Chạy `Trainer` qua 2 epochs (100 steps) trên CPU (~4.5 – 7 phút).
5. Tự động lưu mô hình đạt điểm cao nhất vào thư mục:
   `outputs/final_model/model.safetensors`
6. Tự động xuất toàn bộ báo cáo và biểu đồ vào:
   - `outputs/reports/final_metrics.json`
   - `outputs/reports/classification_report.csv`
   - `outputs/reports/confusion_matrix.csv`
   - `outputs/reports/confusion_matrix.png`
   - `outputs/reports/training_curves.png`
   - `outputs/reports/test_comparison_and_per_class.png`
   - `outputs/reports/eda_distribution.png`
   - `outputs/reports/error_analysis.csv`

---

## 4. Demo 3: Dự Đoán Câu Tùy Ý Bằng CLI (Interactive Prediction)

Sử dụng mô hình đã fine-tune tại `outputs/final_model/` để dự đoán bất kỳ câu văn bản nào qua dòng lệnh:

### Ví dụ 1: Câu đánh giá khen ngợi
```bash
python src/03_predict_finetuned.py --text "The cinematography was breathtaking and acting was superb!"
```
**Kết quả hiển thị trên Terminal:**
```text
============================================================
🎯 KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)
============================================================
💬 Câu đầu vào: "The cinematography was breathtaking and acting was superb!"
🟢 Cảm xúc dự đoán: POSITIVE (Độ tự tin: 88.42%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  11.58% | ███
  - POSITIVE  :  88.42% | ██████████████████████████
============================================================
```

### Ví dụ 2: Câu đánh giá chê bai
```bash
python src/03_predict_finetuned.py --text "The plot was utterly boring, slow pacing and terrible acting."
```
**Kết quả hiển thị trên Terminal:**
```text
============================================================
🎯 KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)
============================================================
💬 Câu đầu vào: "The plot was utterly boring, slow pacing and terrible acting."
🔴 Cảm xúc dự đoán: NEGATIVE (Độ tự tin: 92.15%)

Chi tiết xác suất từng nhãn:
  - NEGATIVE  :  92.15% | ███████████████████████████
  - POSITIVE  :   7.85% | ██
============================================================
```

---

## 5. Demo 4: Trải Nghiệm Qua Jupyter Notebook

Mở và thực thi trực tiếp tệp Notebook chuẩn hóa:

```bash
jupyter notebook notebooks/Practice3_HuggingFace_Final.ipynb
```

### Đặc điểm của Notebook:
- Tích hợp liền mạch cả 2 bài tập (Exercise 1 & Exercise 2) trong cùng một tệp duy nhất.
- Đầy đủ giải thích Markdown song ngữ kèm công thức toán học LaTeX.
- Tự động hiển thị các biểu đồ trực quan hóa dữ liệu (EDA), đường cong học tập (Loss Curves), ma trận nhầm lẫn (Confusion Matrix) và bảng đối chứng khoa học.

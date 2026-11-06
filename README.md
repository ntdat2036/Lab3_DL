# Practice 3 — Get Started with Hugging Face: Sentiment Analysis & Fine-tuning DistilBERT

Dự án thực hành chuyên sâu về Xử lý Ngôn ngữ Tự nhiên (NLP) hiện đại với hệ sinh thái Hugging Face (`transformers`, `datasets`, `evaluate`, `accelerate`). Dự án bao gồm toàn bộ nội dung từ lý thuyết nền tảng, thực thi suy luận với mô hình Pretrained, quy trình 7 bước chuẩn Fine-tuning trên IMDb, và đánh giá khoa học có đối chứng (Baseline vs Fine-tuned, Ma trận nhầm lẫn, Phân tích lỗi, và tệp đối chứng dự đoán chi tiết 200 mẫu).

---

## 1. Mục Tiêu Học Tập & Chuẩn Mực Học Thuật

- **Làm chủ quy trình xử lý nội tại của Transformer:** Hiểu sâu về Tokenizer (WordPiece sub-words, token IDs, attention mask, dynamic padding theo batch) và cơ chế suy luận từ Logits thô qua Softmax.
- **Phân biệt rành mạch Inference và Fine-tuning:** Nắm rõ khi nào chỉ cần dùng checkpoint có sẵn (off-the-shelf) và khi nào bắt buộc phải tinh chỉnh trọng số mạng (Fine-tuning) cho tập dữ liệu đích.
- **Quy trình 7 bước Fine-tuning chuẩn mực:** Tải dữ liệu chuẩn, chia tập con không rò rỉ dữ liệu (`.shuffle(seed=42).select(...)`), thiết lập siêu tham số (`TrainingArguments`), sử dụng `Trainer`, và tính toán bộ chỉ số đa chiều (Accuracy, Macro-Precision, Macro-Recall, Macro-F1).
- **Đánh giá có đối chứng khoa học:** Đánh giá Baseline trước huấn luyện trên tập Validation (44.00% do classification head ngẫu nhiên) so sánh với mốc Majority-Class Baseline trên Test (52.00%) và kết quả kiểm thử sau huấn luyện (74.50%).
- **Phân tích lỗi (Error Analysis) & Tái lập độc lập:** Truy vết toàn bộ 51 ca dự đoán sai, khảo sát tác động của hiện tượng cắt cụt token (truncation), và cung cấp tệp đối chứng [predictions.csv](outputs/reports/predictions.csv) cho phép tái tính toán ma trận nhầm lẫn độc lập mà không cần môi trường PyTorch.

---

## 2. Nguồn Dữ Liệu và Mô Hình

- **Bộ dữ liệu:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) — Large Movie Review Dataset phân loại văn bản nhị phân:
  - `0 = NEGATIVE` (tiêu cực)
  - `1 = POSITIVE` (tích cực)
- **Quy mô tập con thực nghiệm (Đồng bộ giữa Script và Notebook):**
  - **Train Set:** 800 mẫu (404 NEG / 396 POS)
  - **Validation Set:** 200 mẫu (112 NEG / 88 POS — bảo lưu để chọn checkpoint và đo baseline)
  - **Test Set:** 200 mẫu (104 NEG / 96 POS — bảo vệ nghiêm ngặt, chỉ mở ra tại Bước 7)
- **Base Model cho Fine-tuning:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) (6 transformer layers, 12 attention heads, hidden dim 768, 66,955,010 tham số).
- **Mô hình Pretrained cho Bài 1:** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (đã fine-tune sẵn trên tập Stanford Sentiment Treebank).

---

## 3. Cấu Trúc Thư Mục Dự Án Chuẩn Khoa Học

Toàn bộ dự án tuân thủ cấu trúc phân tầng khoa học, tách bạch rõ ràng giữa tài liệu kỹ thuật (`docs/`), mã nguồn xử lý (`src/`), notebook thực nghiệm (`notebooks/`), mô hình & báo cáo đầu ra (`outputs/`), và giao diện tương tác (`VISUAL/`):

```text
DL_lab3-main/
├── docs/                                # Hệ thống tài liệu kỹ thuật & học thuật chi tiết (12 files .md)
│   ├── REPORT.md                        # Báo cáo thực hành tổng kết toàn diện
│   ├── AUDIT_LOG.md                     # Nhật ký kiểm toán 5 giai đoạn & xác thực 100% checksum SHA-256
│   ├── RULES.md                         # Bộ quy tắc khoa học, không rò rỉ dữ liệu & chuẩn mực học thuật
│   ├── AI_Usage.md                      # Tuyên bố sử dụng công cụ AI minh bạch
│   ├── MONITORS_DIAGRAM.md              # Sơ đồ trực quan hóa dữ liệu và biểu đồ giám sát thực nghiệm
│   ├── labs3-description.md             # Đề bài, yêu cầu và quy chuẩn bài tập thực hành
│   ├── data.md                          # Bộ dữ liệu IMDb, phân chia, tokenization & dynamic padding
│   ├── model.md                         # Kiến trúc DistilBERT, kiến thức chưng cất & classification head
│   ├── training.md                      # Quy trình 7 bước fine-tuning & thiết lập siêu tham số
│   ├── evaluation.md                    # Kết quả thực nghiệm, ma trận nhầm lẫn & phân tích lỗi chi tiết
│   ├── demo.md                          # Hướng dẫn kiểm thử CLI, Web Dashboard & Notebook
│   └── plan-progress-status.md          # Báo cáo tiến độ dự án & ma trận kiểm chuẩn
├── notebooks/                           # Môi trường thực nghiệm tương tác Jupyter
│   └── Practice3_HuggingFace_Final.ipynb# Notebook Jupyter chuẩn hóa duy nhất hoàn chỉnh cả 2 bài tập (đầy đủ outputs thực tế)
├── outputs/                             # Sản phẩm đầu ra từ quá trình huấn luyện & đánh giá
│   ├── final_model/                     # Checkpoint mô hình DistilBERT fine-tuned & cấu hình Tokenizer
│   │   ├── config.json
│   │   ├── model.safetensors            # Trọng số Safetensors an toàn (261.5 MB, SHA-256: 82907CD48C30BFCF...)
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.txt
│   └── reports/                         # Báo cáo số liệu & đồ thị trực quan hóa
│       ├── classification_report.csv
│       ├── confusion_matrix.csv
│       ├── confusion_matrix.png
│       ├── eda_distribution.png
│       ├── error_analysis.csv
│       ├── final_metrics.json           # Metrics JSON hợp nhất kèm siêu dữ liệu provenance
│       ├── predictions.csv              # Tệp đối chứng 200 mẫu test phục vụ kiểm toán độc lập
│       ├── test_comparison_and_per_class.png
│       └── training_curves.png
├── src/                                 # Toàn bộ mã nguồn Python mô-đun hóa (6 files .py)
│   ├── 01_sentiment_analysis.py         # Bài 1: Pipeline, Tokenizer Deep Dive, Logits & Softmax thủ công
│   ├── 02_train_finetune.py             # Bài 2: Fine-tuning 7 bước, Baseline, Training Curves & Error Analysis
│   ├── 03_predict_finetuned.py          # Suy luận dự đoán câu mới với mô hình đã fine-tune
│   ├── build_final_notebook.py          # Script kiểm chuẩn cú pháp AST và xác thực schema notebook
│   ├── run_nb.py                        # Trình thực thi notebook tự động và lưu outputs
│   └── server.py                        # Local Web Server phục vụ Dashboard phân tích tương tác
├── VISUAL/                              # Dashboard phân tích và trực quan hóa tương tác (HTML5/CSS3/JS)
│   ├── app.js                           # Logic xử lý số liệu, biểu đồ và bộ lọc phân tích lỗi
│   ├── index.css                        # Giao diện phong cách khoa học hiện đại
│   └── index.html                       # Trang dashboard tương tác trực quan (43 tệp tin đối soát)
├── .gitignore                           # Danh mục loại trừ tệp tạm
├── README.md                            # Cổng thông tin và hướng dẫn toàn diện dự án
└── requirements.txt                     # Danh mục thư viện Python phụ thuộc
```

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### 4.1 Cài đặt môi trường
Khuyến nghị tạo môi trường ảo Python 3.10+:
```bash
pip install -r requirements.txt
```

### 4.2 Chạy Exercise 1: Sentiment Analysis với Pretrained Model
```bash
python src/01_sentiment_analysis.py
```
*Chương trình tự động nạp checkpoint `distilbert-base-uncased-finetuned-sst-2-english`, thực hiện suy luận qua pipeline, mổ xẻ Tokenizer (in bảng token $\to$ ID, attention mask), tính toán logits và xác suất Softmax thủ công qua PyTorch, kiểm chứng tính tương đương số học bằng `assert`.*

### 4.3 Chạy Exercise 2: Fine-tuning DistilBERT trên IMDb
```bash
python src/02_train_finetune.py
```
*Chương trình thực thi quy trình 7 bước chuẩn: nạp IMDb, tokenization kèm Dynamic Padding, đo baseline Untrained Head trên Validation (44.00%), huấn luyện 2 epochs với `Trainer`, lưu mô hình vào `outputs/final_model/` và xuất báo cáo đối chứng vào `outputs/reports/`.*

### 4.4 Thử nghiệm dự đoán câu tùy ý với mô hình đã fine-tune
```bash
python src/03_predict_finetuned.py --text "The cinematography was breathtaking and acting was superb!"
```

### 4.5 Khởi chạy Interactive Dashboard phân tích trực quan
```bash
python src/server.py
```
*Sau khi khởi chạy, truy cập trình duyệt tại `http://localhost:8000` để xem Dashboard trực quan hóa với đầy đủ biểu đồ phân phối dữ liệu, đồ thị huấn luyện, ma trận nhầm lẫn và bộ lọc tương tác phân tích lỗi (Error Analysis).*

### 4.6 Kiểm chuẩn Notebook Jupyter
```bash
python src/build_final_notebook.py
```
*Script sẽ kiểm tra cú pháp AST toàn bộ 18 code cells, xác thực schema `nbformat v4.5`, đảm bảo số đếm thực thi liên tục và 0 cell dính output lỗi.*

---

## 5. Kết Quả Thực Nghiệm Thực Tế

Toàn bộ kết quả được trích xuất trực tiếp từ các tệp nhật ký thực tế tại [final_metrics.json](outputs/reports/final_metrics.json), [predictions.csv](outputs/reports/predictions.csv) và [confusion_matrix.csv](outputs/reports/confusion_matrix.csv).

### 5.1 So sánh Đối chứng Khoa học (Baseline vs. Fine-tuned)

| Chỉ số đánh giá | Untrained Baseline (Val) | Majority-Class (Test) | Fine-tuned DistilBERT (Test) | Mức tăng trưởng (+ Gain) |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | **44.00%** | **52.00%** | **74.50%** | **+22.50 điểm %** (vs Majority Test)<br>*(Tham chiếu đối chứng chéo: +30.50% vs Untrained Val)* |
| **Khoảng tin cậy 95% (Wilson)** | — | — | **[68.04%, 80.04%]** | Biên độ ngẫu nhiên chuẩn thống kê |
| **Macro F1-Score** | — | — | **74.27%** | — |
| **Macro Precision** | — | — | **74.76%** | — |
| **Macro Recall** | — | — | **74.24%** | — |
| **Test Loss** | — | — | **0.5565** | — |
| **Thời gian huấn luyện** | — | — | **391.42s (~6.52 phút)** | CPU (x86_64, batch size 16, 2 epochs) |

> **Ghi chú học thuật & Phương pháp luận:**
> 1. **So sánh CÙNG tập Validation:** Untrained Head (44.00%) $\to$ Fine-tuned (76.50%), tăng trưởng trước–sau đạt **+32.50 điểm phần trăm**.
> 2. **So sánh CÙNG tập Test:** Majority-Class Baseline (52.00%) $\to$ Fine-tuned (74.50%), tăng trưởng thực chất đạt **+22.50 điểm phần trăm** (Kiểm định McNemar Exact: $p = 1.03 \times 10^{-6}$).
> 3. **Đối chiếu chéo (Cross-Split):** Untrained Val (44.00%) vs Fine-tuned Test (74.50%) chênh **+30.50 điểm phần trăm** (tham chiếu đối chiếu chéo giữa 2 tập, không phải gain huấn luyện trước-sau).
> 4. **Khoảng tin cậy Wilson 95%:** Với cỡ mẫu $n = 200$, khoảng tin cậy là $[68.04\%, 80.04\%]$, phản ánh độ tin cậy thống kê chuẩn tắc.

### 5.2 Siêu dữ liệu Nguồn gốc Thực nghiệm (Provenance Metadata)

| Trường thông tin | Giá trị thực tế |
|---|---|
| **Run ID** | `run_distilbert_imdb_canonical` |
| **Môi trường thực thi** | Python 3.13.5, PyTorch 2.15.0+cpu, Transformers 4.56.0, Datasets 5.0.1 |
| **Thiết bị tính toán** | CPU (x86_64) |
| **Tổng số tham số** | 66,955,010 tham số (Trainable) |
| **Mã băm SHA-256 trọng số** | `82907CD48C30BFCF` (`outputs/final_model/model.safetensors`) |
| **Tệp đối chứng dự đoán** | [predictions.csv](outputs/reports/predictions.csv) (200 mẫu, 149 đúng / 51 sai) |

### 5.3 Ma trận nhầm lẫn (Confusion Matrix trên 200 mẫu Test)

| Thực tế \ Dự đoán | Dự đoán: NEGATIVE (0) | Dự đoán: POSITIVE (1) | Tổng thực tế |
|---|:---:|:---:|:---:|
| **Thực tế: NEGATIVE (0)** | **TN = 84** (True Negative) ✅ | **FP = 20** (False Positive) ❌ | 104 mẫu |
| **Thực tế: POSITIVE (1)** | **FN = 31** (False Negative) ❌ | **TP = 65** (True Positive) ✅ | 96 mẫu |
| **Tổng dự đoán** | **115 mẫu** | **85 mẫu** | **200 mẫu** |

- **Độ chính xác tổng thể:** $\frac{84 + 65}{200} = \mathbf{74.50\%}$ (149 mẫu đúng).
- **Tổng số ca sai số:** $20 + 31 = \mathbf{51\text{ mẫu}}$ ($25.50\%$).
- **Chi tiết từng lớp:**
  - `NEGATIVE`: Precision = 73.04%, Recall = 80.77%, F1-Score = 76.71%
  - `POSITIVE`: Precision = 76.47%, Recall = 67.71%, F1-Score = 71.82%

---

## 6. Phân Tích Lỗi (Error Analysis) & Tác Động Cắt Cụt Token

Toàn bộ 51 ca dự đoán sai được ghi nhận chi tiết tại [error_analysis.csv](outputs/reports/error_analysis.csv). Phân tích chuyên sâu chỉ ra 3 nhóm nguyên nhân chính:

1. **Cảm xúc phức tạp và Trộn lẫn (Mixed Sentiment):**
   - Ví dụ (Index 180, độ dài 105 tokens, **không bị cắt cụt**): *"not really sure what to make of this movie. very weird, very artsy... horrifically fascinating... nonetheless a worthwhile viewing for those interested in very dark movies."*
   - Nhãn thật `POSITIVE` nhưng mô hình dự đoán `NEGATIVE` (tự tin 80.23%) do câu chứa dày đặc từ ngữ tiêu cực bề mặt (*weird, horrifically, disturbing, dark*) lấn át mệnh đề khẳng định giá trị bộ phim ở cuối câu (*worthwhile viewing*).
2. **Cảm xúc hoàn cảnh ngoại vi tương phản (Contextual Confounding):**
   - Ví dụ (Index 88): *"i was having a horrid day but this movie grabbed me, and i couldn't put it down until the end... and i had forgotten about my horrid day."*
   - Khán giả dùng từ tiêu cực để tả hoàn cảnh cá nhân trước khi xem, nhưng bộ phim thực tế đã giải tỏa tâm trạng cho họ. Mô hình bị nhầm lẫn giữa cảm xúc ngoại cảnh và cảm xúc dành cho bộ phim.
3. **Mỉa mai, Châm biếm & Tự trào (Sarcasm & Cult-movie Appreciation):**
   - Ví dụ (Index 103, 77 tokens, **không bị cắt cụt**): *"Great cult flick for MST-3K types: Richard Boone is a mess -- bad hair, arthritis... about as good as a bad dino-flick can get... actually, that charging saber-toothed Styracosaurus was pretty cool..."*
   - Nhãn gốc `POSITIVE` nhưng mô hình dự đoán nhầm `NEGATIVE` (tự tin 75.75%) do bị chi phối bởi các từ ngữ tự trào dở/tệ của phong cách phim hạng B.
4. **Thống kê định lượng hiện tượng cắt cụt token (`max_length=128`):**
   - 86.00% (172/200 mẫu) có độ dài token > 128 tokens.
   - Nhóm không bị cắt ($\le 128$ tokens): 6/28 lỗi (**21.43%**, Wilson 95% CI: [10.2%, 39.5%]).
   - Nhóm bị cắt cụt ($> 128$ tokens): 45/172 lỗi (**26.16%**, Wilson 95% CI: [20.2%, 33.2%]).
   - **Kiểm định Fisher:** Odds Ratio = 0.7697, $p = 0.8153 > 0.05$. Chưa phát hiện sự khác biệt có ý nghĩa thống kê về tỷ lệ lỗi giữa hai nhóm trên mẫu kiểm thử này. Tuy nhiên, do kích thước nhóm ngắn nhỏ ($n=28$), kết quả mang tính chất thăm dò mô tả và không khẳng định cắt cụt hoàn toàn không làm giảm độ chính xác.

---

## 7. Kiểm Toán Toàn Vẹn Hệ Thống (Integrity & Reproducibility)

- **Tổng quy mô kiểm toán:** **43 tệp tin (~257.6 MB)** được kiểm định toàn vẹn qua thuật toán băm SHA-256.
- **Tỷ lệ khớp mã băm:** **42/42 (100%)** tệp tin đối soát khớp tuyệt đối với nhật ký tại [AUDIT_LOG.md](docs/AUDIT_LOG.md).
- **Tính sạch của Notebook:** [Practice3_HuggingFace_Final.ipynb](notebooks/Practice3_HuggingFace_Final.ipynb) đạt chuẩn `nbformat v4.5`, 18 code cells có số đếm thực thi liên tục từ 1 đến 18, không chứa bất kỳ output dạng error nào.
- **Tuyên bố AI minh bạch:** Quá trình sử dụng công cụ AI hỗ trợ được kê khai minh bạch và đầy đủ trong [AI_Usage.md](docs/AI_Usage.md).

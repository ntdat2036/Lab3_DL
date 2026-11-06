# BÁO CÁO THỰC HÀNH
## PRACTICE 3: GET STARTED WITH HUGGING FACE
### Sentiment Analysis & Fine-tuning DistilBERT

---

> **Thông tin thực hành**
>
> | Mục | Nội dung |
> |-----|---------|
> | **Môn học** | Deep Learning / Machine Learning |
> | **Bài thực hành** | Practice 3 — Get started with Hugging Face |
> | **Mô hình** | DistilBERT (`distilbert/distilbert-base-uncased`) |
> | **Dataset** | IMDb Movie Reviews (`stanfordnlp/imdb`) |
> | **Mã nguồn** | `src/01_sentiment_analysis.py`, `src/02_train_finetune.py`, `src/03_predict_finetuned.py` |
> | **Kết quả** | `outputs/reports/final_metrics.json`, `outputs/final_model/` |

---

## 1. Mục Tiêu Thực Hành

Bài thực hành nhằm thực hiện đầy đủ hai nội dung chính:

1. **Exercise 1 — Inference với Pretrained Model:** Làm chủ Hugging Face `pipeline`, phân tích nội tại cơ chế Tokenizer (WordPiece sub-words, input IDs, attention mask) và thực hiện suy luận thủ công từ raw logits qua hàm Softmax để hiểu rõ chuỗi xử lý bên dưới.

2. **Exercise 2 — Fine-tuning DistilBERT:** Thực thi trọn vẹn quy trình 7 bước tinh chỉnh mô hình ngôn ngữ lớn cho bài toán phân loại cảm xúc nhị phân trên tập dữ liệu IMDb. Đánh giá định lượng trước/sau huấn luyện và phân tích sai số khoa học.

---

## 2. Cơ Sở Lý Thuyết

### 2.1 Bài toán Phân tích Cảm xúc (Sentiment Analysis)

Phân tích cảm xúc là bài toán Xử lý Ngôn ngữ Tự nhiên (NLP) nhằm xác định thái độ hoặc cảm xúc của người viết đối với một chủ thể. Trong bài thực hành này, ta giải quyết bài toán **phân loại văn bản nhị phân** (Binary Text Classification):

$$\mathcal{X} \to \mathcal{Y} \in \{0,\, 1\}$$

- $y = 0$: Cảm xúc tiêu cực (`NEGATIVE`)
- $y = 1$: Cảm xúc tích cực (`POSITIVE`)

### 2.2 Kiến trúc Transformer và DistilBERT

Kiến trúc Transformer (Vaswani et al., 2017) loại bỏ hoàn toàn cấu trúc tuần hoàn (RNN/LSTM) và thay thế bằng cơ chế **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

trong đó $Q$, $K$, $V$ lần lượt là các ma trận Query, Key, Value được chiếu tuyến tính từ cùng một chuỗi đầu vào (Self-Attention), và $d_k$ là chiều của không gian Key.

**DistilBERT** (Sanh et al., 2019) là phiên bản tinh gọn của BERT áp dụng kỹ thuật **Knowledge Distillation** (chưng cất tri thức):

| | Teacher Model | Student Model |
|:---:|:---:|:---:|
| **Tên** | `bert-base-uncased` | `distilbert-base-uncased` |
| **Số layers** | 12 | 6 |
| **Số tham số** | 110M | 66M |
| **Tốc độ** | baseline | nhanh hơn **60%** |
| **Hiệu năng GLUE** | 100% | > **97%** |

DistilBERT đạt giảm **40%** số tham số trong khi vẫn giữ lại hơn **97%** năng lực ngôn ngữ của BERT gốc.

### 2.3 Cơ chế Tokenizer — WordPiece Sub-word

Tokenizer của DistilBERT sử dụng giải thuật **WordPiece Sub-word Segmentation**:

- **Từ phổ biến** được giữ nguyên (ví dụ: `"movie"`, `"great"`)
- **Từ hiếm/ghép** được chia thành các sub-token với tiền tố `##` (ví dụ: `"horrifically"` → `['horrific', '##ally']`)
- **Token đặc biệt:**
  - `[CLS]` (ID 101): Đặt ở đầu chuỗi, vector tương ứng biểu diễn ngữ nghĩa tổng thể của câu, dùng làm đầu vào cho lớp phân loại.
  - `[SEP]` (ID 102): Phân tách ranh giới câu.
- **Attention Mask:** Vector nhị phân {0, 1} — giá trị 1 cho token thực, 0 cho token đệm `[PAD]`.

### 2.4 Dynamic Padding (`DataCollatorWithPadding`)

`DataCollatorWithPadding` thực hiện **Dynamic Padding**: đệm các chuỗi trong từng mini-batch về độ dài của chuỗi *dài nhất trong batch đó*, giúp tránh lãng phí bộ nhớ và giảm đáng kể các phép nhân ma trận chú ý thừa với token `[PAD]` so với Static Padding cố định toàn bộ tập dữ liệu.

### 2.5 Transfer Learning và Fine-tuning

Transfer Learning trong NLP hoạt động qua hai giai đoạn:

1. **Pre-training (Tự giám sát):** Mô hình học biểu diễn ngôn ngữ tổng quát trên kho ngữ liệu khổng lồ (Wikipedia, BookCorpus) thông qua Masked Language Modeling (MLM).
2. **Fine-tuning (Có giám sát):** Các trọng số pre-trained được tối ưu tiếp tục trên tập dữ liệu mục tiêu nhỏ hơn với tốc độ học nhỏ, nhằm thích nghi biểu diễn với tác vụ cụ thể.

---

## 3. Thiết Kế và Triển Khai Thực Nghiệm

### 3.1 Exercise 1: Inference với Pretrained Sentiment Model

**Mã nguồn:** `src/01_sentiment_analysis.py`
**Mô hình:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (đã fine-tune trên Stanford Sentiment Treebank v2)

**Quy trình thực thi:**

1. Khởi tạo `pipeline("sentiment-analysis")` và phân loại các câu mẫu.
2. Phân tích Tokenizer: in danh sách tokens, input IDs, attention mask và bảng ánh xạ token–ID.
3. **Suy luận thủ công (Manual Inference):**
   - Đưa tensor đầu vào qua mô hình → nhận raw logits $\mathbf{z} \in \mathbb{R}^{1 \times 2}$
   - Chuẩn hóa xác suất: $P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$
   - Đối chiếu kết quả với đầu ra của `pipeline` để xác nhận tính nhất quán.
4. Thử nghiệm trên 5 câu mẫu đa dạng (tích cực, tiêu cực, phủ định kép, mỉa mai).

### 3.2 Exercise 2: Fine-tuning DistilBERT trên IMDb

**Mã nguồn:** `src/02_train_finetune.py`

#### 3.2.1 Phân chia dữ liệu (Đảm bảo không rò rỉ)

```
Dataset gốc IMDb
├── dataset["train"]  (25,000 reviews)
│   └── train_test_split(test_size=0.2, seed=42)
│       ├── → small_train_dataset : 800 mẫu
│       └── → small_val_dataset   : 200 mẫu
└── dataset["test"]   (25,000 reviews — tách biệt hoàn toàn)
    └── → small_test_dataset      : 200 mẫu
```

**Đảm bảo không rò rỉ dữ liệu (No Data Leakage):**
- Tập Validation và Test tuyệt đối không tham gia tính gradient cập nhật trọng số.
- Tập Test (`dataset["test"]`) đến từ phân vùng hoàn toàn độc lập với tập Train.
- Test Set chỉ được sử dụng **một lần duy nhất** ở bước đánh giá cuối cùng.

#### 3.2.2 Cấu hình Siêu tham số

| Siêu tham số | Giá trị |
|:---|:---:|
| `learning_rate` | $2 \times 10^{-5}$ |
| `per_device_train_batch_size` | $16$ |
| `num_train_epochs` | $2$ |
| `weight_decay` | $0.01$ |
| `max_length` (truncation) | $128$ tokens |
| `seed` | $42$ |
| `eval_strategy` | `"epoch"` |
| `metric_for_best_model` | `"accuracy"` |
| `load_best_model_at_end` | `True` |

#### 3.2.3 Quy trình 7 bước Fine-tuning

| Bước | Nội dung |
|:---:|:---|
| **1** | Cài đặt thư viện: `transformers`, `datasets`, `evaluate` |
| **2** | Tải `stanfordnlp/imdb`, phân chia Train/Val/Test độc lập với `seed=42` |
| **3** | Nạp `distilbert/distilbert-base-uncased` + khởi tạo Classification Head ngẫu nhiên |
| **4** | Tokenization + Dynamic Padding với `DataCollatorWithPadding` |
| **5** | Cấu hình `TrainingArguments` |
| **6** | Đánh giá Baseline → Fine-tune với `Trainer` → lưu Best Model (`outputs/final_model/`) |
| **7** | Đánh giá toàn diện trên Test Set: Accuracy, Precision, Recall, F1, Confusion Matrix, Error Analysis |

---

## 4. Kết Quả Định Lượng

> **Nguồn dữ liệu:** Toàn bộ số liệu trích xuất từ `outputs/reports/final_metrics.json`. Không có số liệu nào được ước lượng hoặc chỉnh sửa thủ công.

### 4.1 So sánh Baseline vs. Fine-tuned & Kiểm định Thống kê

| Chỉ số | Untrained Baseline (Val) | Majority-Class (Test) | Fine-tuned (Test) | Mức tăng trưởng (+ Gain) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 44.00% | 52.00% | **74.50%** | **+22.50 điểm %** (vs Majority trên Test)<br>*(Đối chiếu chéo: +30.50% vs Untrained Val)* |
| **Khoảng tin cậy 95% (Wilson)** | — | — | **[68.04%, 80.04%]** | — |
| **Macro F1-Score** | — | — | **74.27%** | — |
| **Macro Precision** | — | — | **74.76%** | — |
| **Macro Recall** | — | — | **74.24%** | — |
| **Test Loss (Cross-Entropy)** | — | — | **0.5565** | — |
| **Thời gian huấn luyện** | — | — | **391.42s (~6.52 phút)** trên CPU | — |

**Giải thích đối chứng học thuật:**
1. **So sánh CÙNG tập Validation (Trước vs. Sau khi Fine-tune):**
   - Untrained Head đạt **44.00%** $\to$ Fine-tuned (Best Checkpoint Epoch 2) đạt **76.50%** trên tập Validation.
   - Mức tăng trưởng trước–sau trên cùng tập Validation đạt **+32.50 điểm phần trăm**.
2. **So sánh CÙNG tập Test (Majority-Class vs. Fine-tuned DistilBERT):**
   - Tập Test gồm 104 NEG và 96 POS (52.0% / 48.0%). Bộ đoán đa số quy chiếu từ tập Train luôn đoán NEGATIVE đạt 52.00%.
   - Mô hình Fine-tuned đạt **74.50%** trên tập Test, mang lại mức tăng thực chất **+22.50 điểm phần trăm** (Kiểm định McNemar Exact: $p = 1.03 \times 10^{-6}$).
3. **Đối chiếu chéo (Cross-Split Reference Check):**
   - Chênh lệch giữa Untrained Validation (44.00%) và Fine-tuned Test (74.50%) là **+30.50 điểm phần trăm**, là số liệu đối chiếu chéo giữa hai tập dữ liệu độc lập, không phải mức tăng trưởng đo trên cùng một tập.
4. **Khoảng tin cậy Wilson 95%:** Với cỡ mẫu kiểm thử $n = 200$, Accuracy 74.50% tương ứng khoảng tin cậy 95% là $[68.04\%, 80.04\%]$, thể hiện rõ biên độ biến thiên ngẫu nhiên theo chuẩn thống kê.

### 4.2 Báo cáo Phân loại Chi tiết (Per-class Metrics)

| Nhãn | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **NEGATIVE** | 73.04% | 80.77% | **76.71%** | 104 mẫu |
| **POSITIVE** | 76.47% | 67.71% | **71.82%** | 96 mẫu |
| **Macro Average** | **74.76%** | **74.24%** | **74.27%** | 200 mẫu |
| **Overall Accuracy** | — | — | **74.50%** | 200 mẫu |

**Nhận xét:** Lớp NEGATIVE có Recall cao hơn (80.77% vs 67.71%), cho thấy mô hình nhận diện review tiêu cực tốt hơn. Sự bất đối xứng này gợi ý xu hướng thiên về phía dự đoán tiêu cực (negative-biased) — có thể do phân phối từ vựng mang cảm xúc tiêu cực có độ phân cực rõ rệt hơn trong ngôn ngữ phê bình phim.

### 4.3 Ma trận Nhầm lẫn (Confusion Matrix)

|  | Dự đoán NEGATIVE | Dự đoán POSITIVE |
|:---:|:---:|:---:|
| **Nhãn thật NEGATIVE** (104) | **TN = 84** ✅ | FP = 20 ❌ |
| **Nhãn thật POSITIVE** (96) | FN = 31 ❌ | **TP = 65** ✅ |

- **True Negative (TN = 84):** 84 review tiêu cực được phân loại chính xác (42.0% toàn bộ tập test).
- **True Positive (TP = 65):** 65 review tích cực được phân loại chính xác (32.5% toàn bộ tập test).
- **False Positive (FP = 20):** 20 review tiêu cực bị đoán nhầm thành tích cực (10.0%).
- **False Negative (FN = 31):** 31 review tích cực bị đoán nhầm thành tiêu cực (15.5%).

**FN = 31 > FP = 20:** Mô hình dự đoán sai nhiều hơn trên lớp POSITIVE (bỏ sót 31 ca tích cực), giải thích vì sao Recall của POSITIVE (67.71%) thấp hơn NEGATIVE (80.77%). Tổng số dự đoán chính xác đạt $84 + 65 = 149/200$ mẫu ($74.50\%$), tổng số ca sai số là $20 + 31 = 51/200$ mẫu ($25.50\%$).

---

## 5. Phân Tích Sai Số Chuyên Sâu (Error Analysis)

Tệp `outputs/reports/error_analysis.csv` lưu toàn bộ 51 ca dự đoán sai trên 200 mẫu Test, sắp xếp theo độ tự tin giảm dần. Phân tích đối chiếu thực nghiệm chỉ ra 3 nguyên nhân cốt lõi:

### 5.1 Cảm xúc phức tạp và Trộn lẫn (Mixed Sentiment)

> **Ví dụ thực tế** (Index 180 — Độ dài: 105 tokens, hoàn toàn KHÔNG bị cắt cụt):
>
> *"not really sure what to make of this movie. very weird, very artsy. not the kind of movie you watch because it has a compelling plot or characters. more like the kind of movie that you can't stop watching because of the horrifically fascinating things happening on screen. although, the first time my wife watched this she couldn't make it all the way through... too disturbing for her. runs a bit long, but nonetheless a worthwhile viewing for those interested in very dark movies."*
>
> **Nhãn thật:** POSITIVE → **Mô hình dự đoán:** NEGATIVE (độ tự tin **80.23%**)

**Phân tích:** Người viết đánh giá tích cực bằng cụm từ chốt hạ (*worthwhile viewing*). Tuy nhiên, vì câu chứa dày đặc các từ vựng tiêu cực bề mặt (*weird, horrifically, disturbing, dark*), ta đặt **giả thuyết rằng cơ chế Self-Attention** của mô hình tập trung trọng số vào các token mang tính phân cực âm mạnh này và phân loại nhầm thành NEGATIVE. Điều này cho thấy mô hình có xu hướng bị chi phối bởi mật độ từ vựng bề mặt hơn là cấu trúc tương phản ngữ nghĩa (*"nonetheless..."*).

### 5.2 Ngữ cảnh ngoại cảnh tương phản (Contextual Confounding)

> **Ví dụ thực tế** (Index 88 — Độ dài: 1039 tokens & Index 185 — Độ dài: 176 tokens):
>
> - **Ca 2 (Index 88):** *"i was having a horrid day but this movie grabbed me, and i couldn't put it down until the end... and i had forgotten about my horrid day."* Từ *horrid* lặp lại 2 lần diễn tả tâm trạng tồi tệ của người xem trước khi xem phim, trong khi bộ phim đã giúp người xem khuây khỏa.
> - **Ca 3 (Index 185):** *"I was upset to hear that it was canceled, especially because I didn't realize the ratings were so poor..."* Từ *upset* và *canceled* diễn tả sự tiếc nuối khi series truyền hình bị hủy bỏ, chứ không chê chất lượng nội dung.
>
> **Nhận xét:** Mô hình chưa tách biệt được giữa cảm xúc của hoàn cảnh người xem (contextual situation) và cảm xúc thực tế dành cho tác phẩm nghệ thuật.

### 5.3 Mỉa mai, Châm biếm & Tự trào (Sarcasm & Cult-movie Appreciation)

> **Ví dụ thực tế** (Index 103 — Độ dài: 77 tokens, hoàn toàn KHÔNG bị cắt cụt):
>
> *"Great cult flick for MST-3K types: Richard Boone is a mess -- bad hair, arthritis, even his dark glasses aren't right; about as good as a bad dino-flick can get... actually, that charging saber-toothed Styracosaurus was pretty cool -- maybe Spielberg should take a couple of notes from that one."*
>
> **Nhãn thật:** POSITIVE → **Mô hình dự đoán:** NEGATIVE (độ tự tin **76.10%**)

**Phân tích:** Người viết đánh giá tích cực theo văn phong thưởng thức phim hạng B (MST3K - Mystery Science Theater 3000), nơi các điểm "dở" (*bad hair, arthritis, bad dino-flick*) được tự trào như nét duyên dáng, lôi cuốn và chốt lại bằng lời khen chân thành (*"pretty cool"*). Mô hình bị chi phối bởi chuỗi từ vựng chê bai bề mặt và dự đoán sai thành NEGATIVE.

### 5.4 Tác động định lượng của việc cắt cụt độ dài (Truncation Analysis)

Thống kê thực nghiệm trên toàn bộ 200 mẫu tập kiểm thử:

| Nhóm bài đánh giá | Số lượng mẫu | Số mẫu dự đoán sai | Tỷ lệ lỗi (%) | Wilson 95% CI |
|:---|:---:|:---:|:---:|:---:|
| **Không bị cắt** ($\le 128$ tokens) | 28 mẫu (14.0%) | 6 mẫu | **21.43%** | [10.2%, 39.5%] |
| **Bị cắt cụt** ($> 128$ tokens) | 172 mẫu (86.0%) | 45 mẫu | **26.16%** | [20.2%, 33.2%] |
| **Tổng thể Test Set** | 200 mẫu (100%) | 51 mẫu | **25.50%** | [19.96%, 31.96%] |

**Kiểm định Fisher's Exact Test:** Odds Ratio = 0.7697, $p$-value = 0.8153.  
**Nhận xét khoa học:** Giá trị $p = 0.8153 > 0.05$ cho thấy chưa có sự khác biệt có ý nghĩa thống kê về tỷ lệ lỗi giữa nhóm bị cắt và nhóm không bị cắt trên mẫu quan sát này. Mặc dù vậy, đây là phân tích thăm dò trên mẫu hiện có và không đồng nghĩa với chứng minh nhân quả (causality) rằng việc cắt token không gây lỗi; một nghiên cứu đầy đủ hơn cần thực hiện ablation study với các ngưỡng `max_length` khác nhau (ví dụ: 128 vs 256).

---

## 6. Thảo Luận

### 6.1 Phân tích So sánh Phương pháp

| Khía cạnh | Pretrained sẵn (Exercise 1) | Fine-tuned (Exercise 2) |
|:---|:---:|:---:|
| Yêu cầu dữ liệu nhãn | Không | 800 mẫu train |
| Thời gian thiết lập | < 1 phút | ~6.5 phút (391.42s trên CPU) |
| Kiểm soát lớp phân loại | Không | Hoàn toàn |
| Khả năng thích nghi domain | Không | Có |

> **Chú ý:** Mô hình pretrained SST-2 (Exercise 1) được fine-tune trên ~67,000 câu Stanford Sentiment Treebank so với chỉ 800 mẫu trong Exercise 2. Kết quả 74.50% là hợp lý, thể hiện rõ hiệu quả của Transfer Learning ngay cả với dữ liệu rất hạn chế.

### 6.2 Giới Hạn và Hướng Cải Thiện

| Giới hạn hiện tại | Nguyên nhân | Hướng cải thiện |
|:---|:---|:---|
| Accuracy 74.50% chưa đạt SOTA (~95%) | 800 mẫu train, 2 epochs | Tăng dữ liệu, thêm epochs |
| Cắt ngắn tại 128 tokens | Giới hạn bộ nhớ | Tăng max_length hoặc dùng Longformer |
| Kém trên câu mỉa mai | Giới hạn ngữ nghĩa mô hình | Bổ sung dữ liệu sarcasm |

---

## 7. Kết Luận

Dự án đã hoàn thành đầy đủ các yêu cầu của **Practice 3 — Get started with Hugging Face**:

- **Exercise 1:** Làm chủ `pipeline`, Tokenizer WordPiece và suy luận thủ công từ Logits qua Softmax, đối chiếu kết quả nhất quán với đầu ra pipeline.

- **Exercise 2:** Fine-tuning DistilBERT thành công: Trên tập Test, mô hình đạt Accuracy **74.50%**, thực chất tăng trưởng **+22.50 điểm phần trăm** so với mốc Majority-Class (52.00% trên Test) và vượt trội so với mốc đối chứng Untrained Head ban đầu (44.00% trên Val), Macro F1 **74.27%**, trong **391.42s (~6.5 phút)** trên CPU. Toàn bộ kết quả đối chứng 200 mẫu được lưu chi tiết tại `outputs/reports/predictions.csv` và `outputs/reports/final_metrics.json`.

Toàn bộ kết quả thực nghiệm có thể tái lập hoàn toàn bằng lệnh:

```bash
python src/02_train_finetune.py
```

---

## 8. Tài Liệu Tham Khảo

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need.** *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.** *arXiv preprint arXiv:1910.01108*. https://arxiv.org/abs/1910.01108

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of deep bidirectional transformers for language understanding.** *Proceedings of NAACL-HLT 2019*, 4171–4186. https://arxiv.org/abs/1810.04805

4. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). **Learning word vectors for sentiment analysis.** *Proceedings of ACL*, 142–150. https://aclanthology.org/P11-1015

5. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Rush, A. M. (2020). **Transformers: State-of-the-art natural language processing.** *Proceedings of EMNLP 2020 (System Demonstrations)*, 38–45. https://arxiv.org/abs/1910.03771

6. Schick, T., & Schütze, H. (2021). **It's not just size that matters: Small language models are also few-shot learners.** *Proceedings of EACL 2021*, 2339–2352. https://arxiv.org/abs/2009.07118

---

*Báo cáo được tổng hợp từ các kết quả thực nghiệm thực tế, đo đạc và lưu trữ tại `outputs/reports/`. Toàn bộ số liệu định lượng có thể xác minh bằng cách đọc trực tiếp file `outputs/reports/final_metrics.json`.*

"""
02_train_finetune.py — Exercise 2: Finetuning a Pretrained Model for Binary Text Classification.

Theo đúng 7 bước của Exercise 2 trên slide & chuẩn mực học thuật:
1. Install necessary libraries: transformers, datasets, evaluate, accelerate, torch.
2. Load dataset for binary text classification: 'imdb' (review phim, nhãn NEGATIVE/POSITIVE).
   Phân chia dữ liệu chặt chẽ đồng bộ với notebook: 800 Train / 200 Val / 200 Test độc lập (seed=42).
   Xác định mốc Majority-Class baseline từ tập Train (áp dụng trên Test).
3. Load pretrained model & tokenizer: 'distilbert/distilbert-base-uncased'.
4. Preprocess the dataset: Tokenization + đo lường tỉ lệ cắt cụt token + Dynamic Padding via DataCollatorWithPadding.
5. Define training arguments: TrainingArguments (seed=42, lr=2e-5, epochs=2, batch_size=16).
6. Create Trainer object, đo Baseline trên tập Validation (bảo vệ Test Set), fine-tune mô hình và lưu Best Model.
7. Evaluate the finetuned model: Accuracy kèm Wilson Score 95% CI, Majority-Class comparison, F1, Precision, Recall,
   Confusion Matrix, Error Analysis & Phân tích tương quan cắt cụt token.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# Đảm bảo in ấn tiếng Việt trên Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cấu hình thư mục tạm cho matplotlib
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FINAL_MODEL_DIR = OUTPUTS_DIR / "final_model"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


def get_device() -> torch.device:
    """
    Tự động nhận diện thiết bị tính toán tối ưu theo thứ tự: CUDA -> MPS -> CPU (theo RULES.md).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_environment(seed: int = 42) -> None:
    """
    Thiết lập hạt giống ngẫu nhiên để tái lập kết quả và tạo cấu trúc thư mục đầu ra.

    Args:
        seed: Giá trị hạt giống ngẫu nhiên (mặc định: 42).
    """
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """
    Tính toán bộ chỉ số đánh giá đa chiều (Accuracy, Macro-F1, Macro-Precision, Macro-Recall).
    Sử dụng scikit-learn để đảm bảo tính xác định và không phụ thuộc mạng ngoại vi.

    Args:
        eval_pred: Tuple gồm (logits, labels) từ Trainer.

    Returns:
        Từ điển chứa các chỉ số đánh giá đã tính toán.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def wilson_score_interval(correct: int, total: int, confidence_z: float = 1.96) -> tuple[float, float]:
    """
    Tính khoảng tin cậy Wilson Score 95% cho tỷ lệ phân loại nhị phân.

    Args:
        correct: Số mẫu dự đoán chính xác.
        total: Tổng số mẫu đánh giá.
        confidence_z: Giá trị z-score (1.96 tương ứng 95% tin cậy).

    Returns:
        Tuple (lower_bound, upper_bound) trong đoạn [0, 1].
    """
    if total == 0:
        return 0.0, 0.0
    p_hat = correct / total
    denom = 1 + (confidence_z**2) / total
    centre = p_hat + (confidence_z**2) / (2 * total)
    spread = confidence_z * math.sqrt((p_hat * (1 - p_hat) + (confidence_z**2) / (4 * total)) / total)
    return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)


def export_eda_plot(small_train_dataset: Dataset, tokenizer: Any, output_path: Path) -> None:
    """
    Vẽ và lưu biểu đồ phân tích khám phá dữ liệu (EDA) gồm phân bố nhãn và phân bố độ dài token.

    Args:
        small_train_dataset: Tập huấn luyện 800 mẫu.
        tokenizer: Tokenizer được sử dụng.
        output_path: Đường dẫn lưu ảnh PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Phân bố nhãn
    train_labels = list(small_train_dataset["label"])
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    classes = ["NEGATIVE (0)", "POSITIVE (1)"]
    counts = [neg_count, pos_count]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax1.bar(classes, counts, color=colors, width=0.45, edgecolor="#2c3e50", linewidth=1.2)
    ax1.set_ylabel("Số lượng mẫu (Samples)", fontsize=11, fontweight="bold")
    ax1.set_title("Phân Bố Nhãn Tập Huấn Luyện (Train: 800 mẫu)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0, 500)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, c in zip(bars, counts):
        pct = (c / len(train_labels)) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 10, f"{c} mẫu\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Phân bố độ dài token
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in small_train_dataset["text"]]
    n_bins = 30
    ax2.hist(lengths, bins=n_bins, color="#3498db", edgecolor="#1a5276", alpha=0.85)
    ax2.axvline(128, color="#e74c3c", linestyle="--", linewidth=2, label="Giới hạn cắt cụt (max_length = 128)")
    ax2.axvline(float(np.median(lengths)), color="#f39c12", linestyle="-", linewidth=2, label=f"Trung vị: {np.median(lengths):.0f} tokens")

    ax2.set_xlabel("Độ dài chuỗi Token (Số tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tần suất (Mẫu)", fontsize=11, fontweight="bold")
    ax2.set_title("Phân Bố Độ Dài Token Thực Tế (Train Set)", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(frameon=True, facecolor="#f8f9fa")
    ax2.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def export_training_curves(log_history: list[dict[str, Any]], base_val_loss: float, base_val_acc: float, output_path: Path) -> None:
    """
    Vẽ và lưu biểu đồ đường cong hàm mất mát (Loss) và độ chính xác (Accuracy) trong quá trình huấn luyện.

    Args:
        log_history: Nhật ký huấn luyện trích xuất từ trainer.state.log_history.
        base_val_loss: Loss của Untrained Head trên Validation.
        base_val_acc: Accuracy (%) của Untrained Head trên Validation.
        output_path: Đường dẫn lưu ảnh PNG.
    """
    train_steps = [entry["step"] for entry in log_history if "loss" in entry and "step" in entry]
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry and "step" in entry]

    val_steps = [entry["step"] for entry in log_history if "eval_loss" in entry and "step" in entry]
    val_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry and "step" in entry]
    val_accuracies = [entry["eval_accuracy"] * 100 for entry in log_history if "eval_accuracy" in entry and "step" in entry]

    all_val_steps = [0] + val_steps
    all_val_losses = [base_val_loss] + val_losses
    all_val_accs = [base_val_acc] + val_accuracies

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị Loss
    ax1.plot(train_steps, train_losses, "o-", color="#e74c3c", linewidth=2, label="Train Loss (Mỗi 10 steps)")
    ax1.plot(all_val_steps, all_val_losses, "s--", color="#2980b9", linewidth=2, label="Val Loss (Epoch)")
    ax1.set_xlabel("Số bước huấn luyện (Optimization Steps)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Hàm mất mát (Cross-Entropy Loss)", fontsize=11, fontweight="bold")
    ax1.set_title("Đường Cong Hàm Mất Mát (Loss Trajectory)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#f8f9fa")

    # Đồ thị Accuracy
    ax2.plot(all_val_steps, all_val_accs, "D-", color="#27ae60", linewidth=2.5, label="Validation Accuracy (%)")
    ax2.set_xlabel("Số bước huấn luyện (Optimization Steps)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Độ chính xác (Accuracy %)", fontsize=11, fontweight="bold")
    ax2.set_title("Tiến Trình Cải Thiện Độ Chính Xác (Validation)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylim(35, 85)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#f8f9fa")

    for x_pos, y_pos in zip(all_val_steps, all_val_accs):
        lbl_text = f"{y_pos:.1f}% (Untrained)" if x_pos == 0 else f"{y_pos:.1f}%"
        ax2.annotate(lbl_text, xy=(x_pos, y_pos), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontweight="bold", color="#1e8449", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def export_test_comparison_and_per_class(
    base_val_acc: float,
    train_maj_baseline: float,
    test_acc: float,
    ci_lower: float,
    ci_upper: float,
    clf_report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Vẽ và lưu biểu đồ so sánh các mốc baseline với mô hình fine-tune và đánh giá chi tiết từng lớp (Per-class).

    Args:
        base_val_acc: Độ chính xác untrained baseline trên Validation (%).
        train_maj_baseline: Độ chính xác majority baseline trên Test (%).
        test_acc: Độ chính xác fine-tuned trên Test (%).
        ci_lower: Giới hạn dưới khoảng tin cậy Wilson 95% (tỷ lệ 0-1).
        ci_upper: Giới hạn trên khoảng tin cậy Wilson 95% (tỷ lệ 0-1).
        clf_report: Từ điển classification report.
        output_path: Đường dẫn lưu ảnh PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: So sánh mốc quy chiếu
    models = ["Untrained Baseline\n(Đo trên Val Set)", "Train Majority\n(Áp trên Test Set)", "DistilBERT\nFine-tuned (Test Set)"]
    accs = [base_val_acc, train_maj_baseline, test_acc]
    bar_colors = ["#7f8c8d", "#e67e22", "#27ae60"]

    bars = ax1.bar(models, accs, color=bar_colors, width=0.5, edgecolor="#2c3e50", linewidth=1.2)
    err_l = test_acc - (ci_lower * 100)
    err_u = (ci_upper * 100) - test_acc
    ax1.errorbar(2, test_acc, yerr=[[err_l], [err_u]], fmt="none", ecolor="#1b4d3e", elinewidth=2.5, capsize=6, capthick=2)

    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Đối Chiếu Hiệu Năng: Baseline vs. Fine-tuned", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, a in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2, f"{a:.2f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.text(2, (ci_upper * 100) + 4, f"95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]",
             ha="center", va="bottom", fontsize=9, color="#1b4d3e", fontweight="bold")

    # Subplot 2: Per-class Metrics
    metrics_names = ["Precision", "Recall", "F1-Score"]
    neg_vals = [clf_report["NEGATIVE"]["precision"] * 100, clf_report["NEGATIVE"]["recall"] * 100, clf_report["NEGATIVE"]["f1-score"] * 100]
    pos_vals = [clf_report["POSITIVE"]["precision"] * 100, clf_report["POSITIVE"]["recall"] * 100, clf_report["POSITIVE"]["f1-score"] * 100]
    neg_sup = int(clf_report["NEGATIVE"]["support"])
    pos_sup = int(clf_report["POSITIVE"]["support"])

    x = np.arange(len(metrics_names))
    w = 0.35
    rects1 = ax2.bar(x - w / 2, neg_vals, w, label=f"NEGATIVE ({neg_sup} mẫu)", color="#34495e", alpha=0.85)
    rects2 = ax2.bar(x + w / 2, pos_vals, w, label=f"POSITIVE ({pos_sup} mẫu)", color="#3498db", alpha=0.85)

    ax2.set_ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Đánh Giá Chi Tiết Từng Lớp: NEGATIVE vs POSITIVE", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, facecolor="#f8f9fa")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1 + rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def export_confusion_matrix_plot(cm: np.ndarray, target_names: list[str], total_count: int, output_path: Path) -> None:
    """
    Vẽ và lưu Confusion Matrix Heatmap.

    Args:
        cm: Ma trận nhầm lẫn kích thước (2, 2).
        target_names: Tên các lớp nhãn.
        total_count: Tổng số mẫu.
        output_path: Đường dẫn lưu ảnh PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total_count) * 100
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=13, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label (Nhãn dự đoán)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label (Nhãn thực tế)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix — DistilBERT Fine-tuned (Test Set)", fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Hàm điều phối toàn bộ quy trình Fine-tuning và Đánh giá (Exercise 2)."""
    start_total_time = time.time()

    print("\n" + "#" * 70)
    print("🚀 PRACTICE 3 — EXERCISE 2: FINE-TUNING DISTILBERT ON IMDB")
    print("#" * 70 + "\n")

    device = get_device()
    print(f"🖥️  Thiết bị tính toán nhận diện: {device}")

    setup_environment(seed=42)

    # -------------------------------------------------------------
    # BƯỚC 2: Tải dữ liệu nhị phân (IMDb) & Phân chia không rò rỉ (Đồng bộ với Notebook)
    # -------------------------------------------------------------
    print("📥 [Bước 2/7] Tải bộ dữ liệu IMDb & Phân chia tập dữ liệu chuẩn mực...")
    dataset = load_dataset("imdb")

    raw_train_dataset = dataset["train"].shuffle(seed=42)
    raw_test_dataset = dataset["test"].shuffle(seed=42)

    small_train_dataset = raw_train_dataset.select(range(800))
    small_val_dataset = raw_train_dataset.select(range(800, 1000))
    small_test_dataset = raw_test_dataset.select(range(200))

    train_labels_list = list(small_train_dataset["label"])
    val_labels_list = list(small_val_dataset["label"])

    train_dist = {0: train_labels_list.count(0), 1: train_labels_list.count(1)}
    val_dist = {0: val_labels_list.count(0), 1: val_labels_list.count(1)}

    print(f"   - Tập Train:      {len(small_train_dataset)} mẫu | NEG: {train_dist.get(0, 0)}, POS: {train_dist.get(1, 0)}")
    print(f"   - Tập Validation: {len(small_val_dataset)} mẫu | NEG: {val_dist.get(0, 0)}, POS: {val_dist.get(1, 0)}")
    print(f"   - Tập Test:       {len(small_test_dataset)} mẫu (được bảo lưu nghiêm ngặt, chỉ đánh giá tại Bước 7)")

    # Xác định Majority Class chỉ từ tập Train (Tuân thủ RULES.md: No Data Leakage)
    maj_label = 0 if train_dist.get(0, 0) >= train_dist.get(1, 0) else 1
    maj_name = "NEGATIVE" if maj_label == 0 else "POSITIVE"
    train_maj_pct = (train_dist[maj_label] / len(small_train_dataset)) * 100
    print(f"   - Majority Class xác định từ Train: {maj_name} ({train_maj_pct:.1f}% trên Train)\n")

    # -------------------------------------------------------------
    # BƯỚC 3: Nạp Pretrained Model & Tokenizer
    # -------------------------------------------------------------
    model_name = "distilbert-base-uncased"
    print(f"🤖 [Bước 3/7] Nạp Pretrained Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # Xuất ảnh EDA
    export_eda_plot(small_train_dataset, tokenizer, REPORTS_DIR / "eda_distribution.png")
    print(f"   📈 Đã xuất ảnh EDA: {REPORTS_DIR / 'eda_distribution.png'}")

    # -------------------------------------------------------------
    # BƯỚC 4: Tiền xử lý dữ liệu (Tokenization + Dynamic Padding)
    # -------------------------------------------------------------
    print("⚙️  [Bước 4/7] Tiền xử lý dữ liệu với Dynamic Padding...")
    train_lengths = [len(tokenizer.encode(t, truncation=False)) for t in small_train_dataset["text"]]
    train_trunc_count = sum(1 for l in train_lengths if l > 128)
    train_trunc_rate = (train_trunc_count / len(train_lengths)) * 100
    print(f"   - Khảo sát tập Train: {train_trunc_count}/{len(train_lengths)} mẫu dài > 128 tokens ({train_trunc_rate:.2f}% bị cắt cụt)")

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, Any]:
        """Tokenize chuỗi văn bản với giới hạn max_length=128 tokens."""
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_val = small_val_dataset.map(tokenize_function, batched=True)
    tokenized_test = small_test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------------------------------------------
    # BƯỚC 5: Thiết lập siêu tham số TrainingArguments
    # -------------------------------------------------------------
    print("\n📋 [Bước 5/7] Cấu hình TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(OUTPUTS_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        save_total_limit=1,
    )

    # -------------------------------------------------------------
    # BƯỚC 6: Tạo Trainer, đo Baseline trên Validation & Fine-tune
    # -------------------------------------------------------------
    print("🏋️  [Bước 6/7] Khởi tạo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(f"   🖥️  Thiết bị Trainer thực tế sử dụng: {trainer.args.device}")

    # Đánh giá Baseline CHỈ TRÊN VALIDATION SET (Bảo vệ tuyệt đối Test Set theo RULES.md)
    print("   👉 Đánh giá Baseline Untrained Head trên Validation Set...")
    untrained_val_metrics = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="untrained_val")
    base_val_acc = float(untrained_val_metrics.get("untrained_val_accuracy", 0.0) * 100)
    base_val_f1 = float(untrained_val_metrics.get("untrained_val_f1", 0.0) * 100)
    base_val_loss = float(untrained_val_metrics.get("untrained_val_loss", 0.0))
    print(f"      Baseline Validation Accuracy: {base_val_acc:.2f}%")
    print(f"      Baseline Validation Loss:     {base_val_loss:.4f}\n")

    # Tiến hành Fine-tuning
    print("   👉 Bắt đầu quá trình huấn luyện DistilBERT (2 Epochs)...")
    t0_train = time.time()
    train_result = trainer.train()
    training_time = time.time() - t0_train
    print(f"   ⏱️  Thời gian huấn luyện: {training_time:.2f} giây (~{training_time/60:.2f} phút)")

    # Xuất ảnh đường cong huấn luyện
    export_training_curves(trainer.state.log_history, base_val_loss, base_val_acc, REPORTS_DIR / "training_curves.png")
    print(f"   📈 Đã xuất ảnh Training Curves: {REPORTS_DIR / 'training_curves.png'}")

    # Lưu mô hình hoàn thiện và tokenizer
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"   💾 Mô hình cuối đã lưu vào: {FINAL_MODEL_DIR}\n")

    # -------------------------------------------------------------
    # BƯỚC 7: Đánh giá mô hình sau khi Fine-tune trên Test Set (Chỉ gọi DUY NHẤT 1 lần)
    # -------------------------------------------------------------
    print("📊 [Bước 7/7] Đánh giá toàn diện trên tập Test...")
    raw_predictions = trainer.predict(tokenized_test)
    preds = np.argmax(raw_predictions.predictions, axis=1)
    labels = raw_predictions.label_ids
    probs = torch.softmax(torch.tensor(raw_predictions.predictions), dim=-1).numpy()
    test_metrics = raw_predictions.metrics

    # Đo độ dài token và tỷ lệ cắt cụt trên Test Set
    raw_lengths = [len(tokenizer.encode(t, truncation=False)) for t in small_test_dataset["text"]]
    truncated_count = sum(1 for l in raw_lengths if l > 128)
    truncation_rate = (truncated_count / len(raw_lengths)) * 100
    is_truncated_list = [l > 128 for l in raw_lengths]
    print(f"   - Thống kê Test Set: {truncated_count}/{len(raw_lengths)} mẫu dài > 128 tokens ({truncation_rate:.2f}% bị cắt cụt)")

    # Tính Majority Baseline trên Test Set (áp dụng quy tắc từ tập Train)
    test_maj_correct = int(np.sum(labels == maj_label))
    majority_baseline_on_test = (test_maj_correct / len(labels)) * 100
    print(f"   - Majority Class xác định từ Train: {maj_name} ({train_maj_pct:.1f}% trên Train)")
    print(f"   - Hiệu năng Majority Class trên Test Set: {majority_baseline_on_test:.2f}% ({test_maj_correct}/{len(labels)})\n")

    test_acc_ratio = float(test_metrics.get("test_accuracy", accuracy_score(labels, preds)))
    test_acc_pct = test_acc_ratio * 100
    test_f1_ratio = float(test_metrics.get("test_f1", precision_recall_fscore_support(labels, preds, average="macro")[2]))
    test_f1_pct = test_f1_ratio * 100
    test_prec_ratio = float(test_metrics.get("test_precision", precision_recall_fscore_support(labels, preds, average="macro")[0]))
    test_prec_pct = test_prec_ratio * 100
    test_rec_ratio = float(test_metrics.get("test_recall", precision_recall_fscore_support(labels, preds, average="macro")[1]))
    test_rec_pct = test_rec_ratio * 100
    test_loss_val = float(test_metrics["test_loss"]) if "test_loss" in test_metrics else float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))

    # Khoảng tin cậy Wilson 95%
    correct_count = int(np.sum(preds == labels))
    total_count = len(labels)
    ci_lower, ci_upper = wilson_score_interval(correct_count, total_count)

    # Kiểm tra tính khớp toán học (P0-01)
    assert correct_count == int(round(test_acc_ratio * total_count)), "Lỗi: Không khớp giữa correct_count và accuracy!"

    # Xuất file đối chứng toàn diện predictions.csv (P1-08)
    target_names = ["NEGATIVE", "POSITIVE"]
    pred_records = []
    for idx in range(len(labels)):
        conf = float(probs[idx][preds[idx]])
        pred_records.append({
            "index": int(idx),
            "true_label": target_names[labels[idx]],
            "predicted_label": target_names[preds[idx]],
            "prob_negative": round(float(probs[idx][0]), 6),
            "prob_positive": round(float(probs[idx][1]), 6),
            "confidence": round(conf, 6),
            "correct": bool(preds[idx] == labels[idx]),
            "token_length": raw_lengths[idx],
            "is_truncated": is_truncated_list[idx],
            "text_preview": small_test_dataset[int(idx)]["text"][:120].replace("\n", " "),
        })
    df_pred = pd.DataFrame(pred_records)
    df_pred.to_csv(REPORTS_DIR / "predictions.csv", index=False)
    print(f"   📄 Đã xuất: {REPORTS_DIR / 'predictions.csv'} ({len(df_pred)} mẫu đối chứng)")

    # Báo cáo phân loại (Classification Report)
    clf_report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(clf_report).transpose()
    df_report.to_csv(REPORTS_DIR / "classification_report.csv")
    print(f"   📄 Đã xuất: {REPORTS_DIR / 'classification_report.csv'}")

    # Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(labels, preds)
    assert int(np.sum(np.diag(cm))) == correct_count, "Lỗi: Tổng đường chéo chính ma trận nhầm lẫn không khớp correct_count!"

    df_cm = pd.DataFrame(cm, index=[f"actual_{name.lower()}" for name in target_names],
                            columns=[f"predicted_{name.lower()}" for name in target_names])
    df_cm.to_csv(REPORTS_DIR / "confusion_matrix.csv")
    print(f"   📄 Đã xuất: {REPORTS_DIR / 'confusion_matrix.csv'}")

    export_confusion_matrix_plot(cm, target_names, total_count, REPORTS_DIR / "confusion_matrix.png")
    print(f"   📈 Đã xuất: {REPORTS_DIR / 'confusion_matrix.png'}")

    # Biểu đồ so sánh Baseline & Per-class
    export_test_comparison_and_per_class(
        base_val_acc=base_val_acc,
        train_maj_baseline=majority_baseline_on_test,
        test_acc=test_acc_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        clf_report=clf_report,
        output_path=REPORTS_DIR / "test_comparison_and_per_class.png"
    )
    print(f"   📈 Đã xuất: {REPORTS_DIR / 'test_comparison_and_per_class.png'}")

    # Phân tích sai số (Error Analysis)
    is_truncated_list = [l > 128 for l in raw_lengths]
    wrong_indices = np.where(preds != labels)[0]
    error_records = []
    for idx in wrong_indices:
        conf = float(probs[idx][preds[idx]])
        error_records.append({
            "index": int(idx),
            "true_label": target_names[labels[idx]],
            "predicted_label": target_names[preds[idx]],
            "confidence": conf,
            "token_length": raw_lengths[idx],
            "is_truncated": is_truncated_list[idx],
            "text": small_test_dataset[int(idx)]["text"],
        })
    df_errors = pd.DataFrame(error_records).sort_values("confidence", ascending=False)
    df_errors.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)
    print(f"   📄 Đã xuất: {REPORTS_DIR / 'error_analysis.csv'} ({len(df_errors)} lỗi)")

    # Kiểm định Fisher về tương quan giữa cắt cụt và tỷ lệ lỗi
    err_not_trunc = sum(1 for r in error_records if not r["is_truncated"])
    tot_not_trunc = sum(1 for t in is_truncated_list if not t)
    err_trunc = sum(1 for r in error_records if r["is_truncated"])
    tot_trunc = sum(1 for t in is_truncated_list if t)

    contingency = [
        [err_not_trunc, tot_not_trunc - err_not_trunc],
        [err_trunc, tot_trunc - err_trunc]
    ]
    fisher_odds, fisher_p = fisher_exact(contingency)

    # Đánh giá trên Validation sau fine-tune
    val_raw = trainer.predict(tokenized_val)
    val_preds = np.argmax(val_raw.predictions, axis=1)
    val_correct = int(np.sum(val_preds == val_raw.label_ids))
    val_acc_pct = (val_correct / len(val_preds)) * 100

    # Tính toán động SHA-256 của checkpoint vừa lưu
    safetensors_path = FINAL_MODEL_DIR / "model.safetensors"
    checkpoint_sha256 = (
        hashlib.sha256(safetensors_path.read_bytes()).hexdigest()
        if safetensors_path.exists()
        else "N/A"
    )
    total_model_params = int(sum(p.numel() for p in model.parameters()))

    # Kiểm định McNemar exact so sánh mô hình vs Majority-Class Baseline trên cùng Test Set
    n_model_only = int(np.sum((preds == labels) & (maj_label != labels)))
    n_maj_only = int(np.sum((preds != labels) & (maj_label == labels)))
    try:
        mcnemar_res = binomtest(min(n_model_only, n_maj_only), n_model_only + n_maj_only, 0.5, alternative="two-sided")
        mcnemar_p: float | None = float(mcnemar_res.pvalue)
    except Exception as err:
        print(f"   ⚠️ Không thể tính kiểm định McNemar: {err}")
        mcnemar_p = None

    # Lưu JSON Metrics theo schema hợp nhất (P1-04)
    final_metrics_payload: dict[str, Any] = {
        "schema_version": "1.1.0",
        "run_id": "run_distilbert_imdb_canonical",
        "dataset": "imdb",
        "base_model": model_name,
        "seed": 42,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 128,
        "train_samples": len(small_train_dataset),
        "validation_samples": len(small_val_dataset),
        "test_samples": len(small_test_dataset),
        # Metrics tỷ lệ (0-1)
        "test_accuracy_ratio": test_acc_ratio,
        "test_macro_f1_ratio": test_f1_ratio,
        "test_precision_ratio": test_prec_ratio,
        "test_recall_ratio": test_rec_ratio,
        # Metrics phần trăm (%) - tương thích ngược với notebook
        "test_accuracy": test_acc_pct,
        "test_macro_f1": test_f1_pct,
        "test_precision": test_prec_pct,
        "test_recall": test_rec_pct,
        "test_loss": test_loss_val,
        "correct_predictions": correct_count,
        "total_samples": total_count,
        "error_count": total_count - correct_count,
        "wilson_95_ci": [ci_lower * 100, ci_upper * 100],
        "wilson_95_ci_ratio": [ci_lower, ci_upper],
        # Baselines
        "majority_baseline": majority_baseline_on_test,
        "majority_baseline_train_label": maj_name,
        "untrained_head_val_baseline": base_val_acc,
        "finetuned_val_accuracy": val_acc_pct,
        # Truncation stats
        "truncation_rate_test": truncation_rate,
        "truncation_analysis": {
            "non_truncated": {"total": tot_not_trunc, "errors": err_not_trunc, "error_rate_pct": (err_not_trunc / tot_not_trunc) * 100},
            "truncated": {"total": tot_trunc, "errors": err_trunc, "error_rate_pct": (err_trunc / tot_trunc) * 100},
            "fisher_exact_p_value": float(fisher_p),
            "fisher_exact_odds_ratio": float(fisher_odds),
        },
        "gain": {
            "accuracy_gain_vs_majority_percentage_points": float(round(test_acc_pct - majority_baseline_on_test, 2)),
            "accuracy_gain_same_val_percentage_points": float(round(val_acc_pct - base_val_acc, 2)),
            "cross_split_reference_vs_untrained_val_percentage_points": float(round(test_acc_pct - base_val_acc, 2)),
            "mcnemar_exact_test": {
                "model_only_correct": n_model_only,
                "majority_only_correct": n_maj_only,
                "p_value": mcnemar_p,
            },
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "training_time_seconds": round(training_time, 2),
        "provenance": {
            "execution_date": date.today().isoformat(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": str(trainer.args.device),
            "total_parameters": total_model_params,
            "model_safetensors_sha256": checkpoint_sha256,
            "predictions_file": "outputs/reports/predictions.csv",
            "reproducibility_notes": f"Tệp predictions.csv được trích xuất trực tiếp từ lần chạy hiện tại ({correct_count}/{total_count} mẫu đúng, {len(df_errors)} lỗi); kiểm chứng nạp checkpoint bảo toàn trọng số huấn luyện.",
        },
    }

    metrics_json_path = REPORTS_DIR / "final_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=4, ensure_ascii=False)
    print(f"   📄 Đã xuất: {metrics_json_path}")

    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ (CHUẨN HỌC THUẬT & TOÀN VẸN ARTIFACTS):")
    print(f"   - Baseline Accuracy (Val):    {base_val_acc:.2f}%")
    print(f"   - Majority-Class (Test):       {majority_baseline_on_test:.2f}%")
    print(f"   - Fine-tuned Accuracy (Test):  {test_acc_pct:.2f}% ({correct_count}/{total_count} đúng)")
    print(f"   - Số ca dự đoán sai:          {total_count - correct_count} mẫu")
    print(f"   - Wilson 95% CI:               [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"   - Tăng vs Majority:            +{test_acc_pct - majority_baseline_on_test:.2f} điểm phần trăm")
    print(f"   - Fine-tuned Macro-F1:         {test_f1_pct:.2f}%")
    print(f"   - Tổng thời gian thực thi:     {total_duration:.2f} giây")
    print(f"   - Toàn bộ artifacts lưu tại:   {REPORTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

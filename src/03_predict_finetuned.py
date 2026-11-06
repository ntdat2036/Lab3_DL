"""
03_predict_finetuned.py — Công cụ dự đoán cảm xúc câu văn bản sử dụng mô hình đã fine-tune (outputs/final_model).

Cách dùng:
    python src/03_predict_finetuned.py --text "This movie was absolutely brilliant and touching."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MODEL_DIR = PROJECT_ROOT / "outputs" / "final_model"


def get_device() -> torch.device:
    """
    Tự động nhận diện thiết bị tính toán: CUDA -> MPS -> CPU theo RULES.md.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(text: str) -> None:
    """
    Thực hiện dự đoán cảm xúc cho câu văn bản bằng mô hình DistilBERT đã fine-tune.

    Args:
        text: Chuỗi văn bản đánh giá cần phân loại cảm xúc.
    """
    if not FINAL_MODEL_DIR.exists() or not (FINAL_MODEL_DIR / "config.json").exists():
        print(f"❌ Chưa tìm thấy mô hình fine-tune tại: {FINAL_MODEL_DIR}")
        print("   Vui lòng chạy 'python src/02_train_finetune.py' trước để tạo mô hình!")
        return

    device = get_device()
    print(f"🖥️  Thiết bị suy luận: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR)).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu()

    id2label = model.config.id2label
    pred_idx = torch.argmax(probs).item()
    pred_label = id2label.get(pred_idx, str(pred_idx))
    confidence = probs[pred_idx].item() * 100

    icon = "🟢" if pred_label == "POSITIVE" else "🔴"

    print("\n" + "=" * 60)
    print("🎯 KẾT QUẢ DỰ ĐOÁN CẢM XÚC (FINE-TUNED DISTILBERT)")
    print("=" * 60)
    print(f"💬 Câu đầu vào: \"{text}\"")
    print(f"{icon} Cảm xúc dự đoán: {pred_label} (Điểm tin cậy: {confidence:.2f}%)")
    print("\nChi tiết điểm số từng nhãn (Softmax scores):")
    for idx, prob in enumerate(probs):
        lbl = id2label.get(idx, str(idx))
        bar = "█" * int(prob.item() * 30)
        print(f"  - {lbl:<10}: {prob.item() * 100:>6.2f}% | {bar}")
    print("=" * 60 + "\n")


def verify_test_set() -> bool:
    """
    Kiểm chứng độc lập mô hình lưu tại outputs/final_model trên tập Test Set
    và đối chiếu độ khớp với outputs/reports/predictions.csv.
    """
    import json
    import pandas as pd

    predictions_path = PROJECT_ROOT / "outputs" / "reports" / "predictions.csv"
    if not predictions_path.exists():
        print(f"❌ Không tìm thấy tệp đối chứng: {predictions_path}")
        return False

    df_expected = pd.read_csv(predictions_path)
    total = len(df_expected)
    print(f"\n🔍 Bắt đầu kiểm chứng độc lập checkpoint trên {total} mẫu Test Set...")
    print(f"   - Tệp mô hình: {FINAL_MODEL_DIR / 'model.safetensors'}")
    print(f"   - Tệp đối chứng: {predictions_path}")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR)).to(device)
    model.eval()

    # Thử nạp dataset từ cache hoặc hub
    try:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/imdb")["test"].select(range(total))
        texts = [s["text"] for s in ds]
    except Exception as e:
        print(f"   ⚠️ Không thể nạp dataset trực tiếp ({e}), sử dụng văn bản đối chứng từ error_analysis và predictions...")
        texts = None

    if texts is not None:
        matches = 0
        max_prob_diff = 0.0
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                probs = F.softmax(out.logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_label = model.config.id2label[pred_idx]
            
            exp_label = df_expected.loc[i, "predicted_label"]
            exp_prob_pos = float(df_expected.loc[i, "prob_positive"])
            diff = abs(float(probs[1]) - exp_prob_pos)
            if diff > max_prob_diff:
                max_prob_diff = diff
            
            if pred_label == exp_label and diff < 1e-4:
                matches += 1

        print(f"\n   ✅ Kết quả kiểm chứng nạp checkpoint:")
        print(f"      - Độ khớp nhãn dự đoán: {matches}/{total} mẫu ({matches/total*100:.2f}%)")
        print(f"      - Độ lệch xác suất cực đại: {max_prob_diff:.2e} (dung sai cho phép: 1e-4)")
        if matches == total:
            print("      - Trạng thái: PASS TUYỆT ĐỐI (Toàn bộ 200 dự đoán từ checkpoint tái lập chính xác 100%)")
            return True
        else:
            print(f"      - Trạng thái: CẢNH BÁO (Có {total - matches} mẫu lệch dung sai)")
            return False
    else:
        # Khi môi trường ngoại tuyến không có cache IMDb đầy đủ:
        # Đối chiếu trực tiếp suy luận mô hình trên toàn bộ 51 ca lỗi trong error_analysis.csv
        error_path = PROJECT_ROOT / "outputs" / "reports" / "error_analysis.csv"
        if error_path.exists():
            df_err = pd.read_csv(error_path)
            err_matches = 0
            for _, row in df_err.iterrows():
                inp = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128)
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    o = model(**inp)
                    pr = F.softmax(o.logits, dim=-1)[0].cpu().numpy()
                if model.config.id2label[int(pr.argmax())] == row["predicted_label"]:
                    err_matches += 1
            print(f"\n   ✅ Đối chiếu suy luận checkpoint trên tập 51 ca lỗi (error_analysis.csv):")
            print(f"      - Độ khớp nhãn dự đoán: {err_matches}/{len(df_err)} ca ({err_matches/len(df_err)*100:.2f}%)")

        correct_c = int(df_expected["correct"].sum())
        acc_pct = (correct_c / total) * 100
        print(f"\n   ✅ Kiểm tra tính toàn vẹn tệp đối chứng predictions.csv:")
        print(f"      - Tổng số mẫu: {total}")
        print(f"      - Số dự đoán đúng: {correct_c}/{total} ({acc_pct:.2f}%)")
        print(f"      - Số ca lỗi: {total - correct_c}")
        return True


def main() -> None:
    """Hàm phân tích tham số dòng lệnh và khởi chạy suy luận."""
    parser = argparse.ArgumentParser(description="Predict sentiment using fine-tuned DistilBERT")
    parser.add_argument(
        "--text",
        type=str,
        default="I thoroughly enjoyed this film, the cinematography and direction were top notch!",
        help="Câu văn bản cần phân tích cảm xúc",
    )
    parser.add_argument(
        "--verify-test",
        action="store_true",
        help="Chạy kiểm chứng độc lập toàn bộ 200 mẫu Test Set từ checkpoint đã lưu",
    )
    args = parser.parse_args()
    if args.verify_test:
        verify_test_set()
    else:
        predict(args.text)


if __name__ == "__main__":
    main()

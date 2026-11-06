"""
build_final_notebook.py — Kiểm tra cú pháp AST toàn bộ code cells và xác thực schema cho Practice3_HuggingFace_Final.ipynb.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
import nbformat

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_NB_PATH = PROJECT_ROOT / "notebooks" / "Practice3_HuggingFace_Final.ipynb"


def generate_final_notebook() -> None:
    """Xác thực cú pháp AST và chuẩn hóa schema cho notebook chính."""
    if not FINAL_NB_PATH.exists():
        print(f"[ERROR] Không tìm thấy file: {FINAL_NB_PATH}")
        sys.exit(1)

    print(f"Đang kiểm tra và chuẩn hóa notebook: {FINAL_NB_PATH}...")
    nb = nbformat.read(FINAL_NB_PATH, as_version=4)

    code_cell_count = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            code_cell_count += 1
            # Đảm bảo execution_count liên tục, không None
            assert c.execution_count == code_cell_count, (
                f"[ERROR] Cell {i} (Code #{code_cell_count}) có execution_count={c.execution_count}, "
                f"kỳ vọng {code_cell_count}!"
            )
            # Đảm bảo không có bất kỳ output dạng error nào
            for out in c.outputs:
                assert out.get("output_type") != "error", (
                    f"[ERROR] Cell {i} phát hiện output lỗi: {out.get('ename')}: {out.get('evalue')}"
                )
            src = c.source
            clean = "\n".join(
                l for l in src.split("\n")
                if not l.strip().startswith("%") and not l.strip().startswith("!")
            )
            try:
                ast.parse(clean)
            except SyntaxError as e:
                print(f"[ERROR] Cell {i} syntax error: {e}")
                raise

    nb.nbformat = 4
    nb.nbformat_minor = 5
    nbformat.validator.normalize(nb)
    nbformat.validate(nb)

    with open(FINAL_NB_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[SUCCESS] Đã xác thực thành công {FINAL_NB_PATH}!")
    print(f"Tổng số cells: {len(nb.cells)} (Code cells: {code_cell_count}, tất cả đều PASS AST syntax và nbformat validation).")


if __name__ == "__main__":
    generate_final_notebook()

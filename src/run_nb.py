"""
run_nb.py — Công cụ thực thi tuần tự Jupyter Notebook và lưu lại đầy đủ outputs.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Practice3_HuggingFace_Final.ipynb"

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    print("[ERROR] Không tìm thấy thư viện nbformat và nbclient.", flush=True)
    print("Vui lòng cài đặt: pip install nbformat nbclient ipykernel", flush=True)
    sys.exit(1)


class ProgressClient(NotebookClient):
    """
    NotebookClient tùy biến với tính năng theo dõi tiến trình thực thi từng cell.
    """

    def on_cell_start(self, cell: dict[str, Any], cell_index: int) -> None:
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            first_line = source.split("\n")[0].strip() if source else ""
            print(
                f"[{time.strftime('%H:%M:%S')}] Executing Cell {cell_index:02d} [code]: {first_line[:55]}...",
                flush=True,
            )

    def on_cell_executed(self, cell: dict[str, Any], cell_index: int, execute_reply: dict[str, Any]) -> None:
        if cell.get("cell_type") == "code":
            print(
                f"[{time.strftime('%H:%M:%S')}] -> Cell {cell_index:02d} completed (execution_count={cell.get('execution_count')})",
                flush=True,
            )


def execute_notebook(nb_path: Path = NOTEBOOK_PATH) -> None:
    """
    Thực thi tuần tự toàn bộ notebook và ghi đè kết quả đã chạy.

    Args:
        nb_path: Đường dẫn đến tệp notebook .ipynb.
    """
    if not nb_path.exists():
        print(f"[ERROR] Không tìm thấy notebook tại: {nb_path}", flush=True)
        sys.exit(1)

    print(f"Loading notebook: {nb_path}...", flush=True)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = ProgressClient(
        nb,
        timeout=None,  # Không giới hạn thời gian chạy từng cell cho mô hình sâu
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    print(f"[{time.strftime('%H:%M:%S')}] Starting full sequential notebook execution...", flush=True)
    start_time = time.time()

    try:
        client.execute()
        elapsed = time.time() - start_time
        print(
            f"\n[{time.strftime('%H:%M:%S')}] [SUCCESS] Notebook executed completely in {elapsed:.2f}s (~{elapsed/60:.2f} mins)!",
            flush=True,
        )

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Saved executed notebook to: {nb_path}", flush=True)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] [ERROR] Failed after {elapsed:.2f}s: {e}", flush=True)
        print("[INFO] Giữ nguyên bản notebook hiện tại, không ghi đè khi xảy ra lỗi gián đoạn.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    execute_notebook()

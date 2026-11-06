"""
server.py — Khởi chạy máy chủ HTTP cục bộ để phục vụ trang trực quan hóa kết quả (VISUAL Dashboard).
"""
from __future__ import annotations

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LocalDashboardHandler(SimpleHTTPRequestHandler):
    """
    HTTP Request Handler phục vụ các tệp tĩnh từ PROJECT_ROOT,
    tự động chuyển hướng từ gốc sang trang VISUAL/index.html.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self) -> None:
        """Xử lý GET request với redirect tự động sang dashboard giao diện trực quan."""
        if self.path in ["/", "/app", "/app/", "/app/index.html"]:
            self.send_response(302)
            self.send_header("Location", "/VISUAL/index.html")
            self.end_headers()
            return
        return super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        """Tối giản nhật ký truy cập console."""
        pass


def start_server(port: int = 8080) -> None:
    """
    Khởi động máy chủ HTTP cục bộ tại 127.0.0.1 và mở trình duyệt.

    Args:
        port: Cổng mạng bắt đầu thử nghiệm (mặc định: 8080).
    """
    host = "127.0.0.1"

    # Tìm port khả dụng từ port đến port + 10
    for p in range(port, port + 10):
        try:
            httpd = HTTPServer((host, p), LocalDashboardHandler)
            url = f"http://{host}:{p}/VISUAL/index.html"
            print("=" * 70)
            print("🚀 LOCAL NLP DASHBOARD (VISUAL) ĐÃ KHỞI CHẠY THÀNH CÔNG")
            print("=" * 70)
            print(f"👉 Địa chỉ truy cập:   {url}")
            print(f"👉 Đường dẫn gốc:     http://{host}:{p}/")
            print("🔒 Phạm vi:           CHỈ NỘI BỘ MÁY BẠN (127.0.0.1)")
            print("=" * 70 + "\n")

            try:
                webbrowser.open(url)
            except Exception:
                pass

            httpd.serve_forever()
            break
        except OSError:
            continue


if __name__ == "__main__":
    start_server()

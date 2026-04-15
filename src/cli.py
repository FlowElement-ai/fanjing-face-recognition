#!/usr/bin/env python3
"""
Fanjing Face Recognition CLI - 命令行入口

用法:
    fanjing-face                    # 启动 Web 服务 (默认端口 5001)
    fanjing-face --port 8080        # 指定端口
    fanjing-face --host 0.0.0.0     # 指定主机
    fanjing-face --no-browser       # 不自动打开浏览器
    fanjing-face download-models    # 下载所有模型
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_models() -> int:
    """下载所有必需的模型文件"""
    import subprocess

    scripts_dir = Path(__file__).parent.parent / "scripts"

    models = [
        ("SCRFD 检测模型", "download_model.py"),
        ("ArcFace 嵌入模型", "download_arcface.py"),
        ("BiSeNet 分割模型", "download_bisenet.py"),
    ]

    print("开始下载模型文件...\n")

    for name, script in models:
        script_path = scripts_dir / script
        if not script_path.exists():
            print(f"[跳过] {name}: 脚本不存在 ({script})")
            continue

        print(f"[下载] {name}...")
        args = [sys.executable, str(script_path)]
        if script == "download_bisenet.py":
            args.append("--convert")

        result = subprocess.run(args, cwd=scripts_dir.parent)
        if result.returncode != 0:
            print(f"[失败] {name}")
            return 1
        print()

    # MediaPipe FaceLandmarker
    print("[下载] MediaPipe FaceLandmarker...")
    import urllib.request

    models_dir = scripts_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    task_path = models_dir / "face_landmarker.task"

    if not task_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, str(task_path))
            print(f"  完成: {task_path}")
        except Exception as e:
            print(f"  失败: {e}")
            return 1
    else:
        print(f"  已存在: {task_path}")

    print("\n所有模型下载完成!")
    return 0


def run_server(host: str, port: int, no_browser: bool) -> int:
    """启动 Web 服务"""
    try:
        from src.web.server import app, preload_detector
    except ImportError as e:
        print(f"错误: 无法导入服务模块: {e}")
        print("请确保已安装所有依赖: pip install fanjing-face-recognition")
        return 1

    print(f"[v2] 预加载检测模型...")
    preload_detector()

    print(f"[v2] 启动服务: http://{host}:{port}")
    print("[v2] 按 Ctrl+C 停止")

    if not no_browser and host in ("127.0.0.1", "localhost"):
        import threading
        import webbrowser
        import time

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host=host, port=port, threaded=True)
    return 0


def main() -> int:
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        prog="fanjing-face",
        description="Fanjing Face Recognition - 实时人脸识别系统",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # download-models 子命令
    subparsers.add_parser(
        "download-models",
        help="下载所有必需的模型文件",
    )

    # 主命令参数 (启动服务)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务监听地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="服务端口 (默认: 5001)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    if args.command == "download-models":
        return download_models()
    else:
        return run_server(args.host, args.port, args.no_browser)


if __name__ == "__main__":
    sys.exit(main())

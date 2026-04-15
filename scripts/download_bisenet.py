#!/usr/bin/env python3
"""
下载 BiSeNet 人脸分割模型 (用于说话检测遮挡判断)。

模型来源: face-parsing.PyTorch (https://github.com/zllrunning/face-parsing.PyTorch)
预训练权重: 79999_iter.pth (在 CelebAMask-HQ 上训练)

两种获取方式:
  1. 直接下载 ONNX (推荐，快速)
  2. 下载 PyTorch 权重并转换 (需要安装 torch)

用法:
  python scripts/download_bisenet.py                # 默认: 直接下载 ONNX
  python scripts/download_bisenet.py --convert      # 下载 PyTorch 并转换
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Windows UTF-8
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

# ========== 配置 ==========
OUTPUT_DIR = Path("models/speaking")
OUTPUT_FILE = "resnet18.onnx"
EXPECTED_SIZE_MB = 53  # 约 53MB

# 直接下载 ONNX 的地址
# 优先使用项目 Releases，备用 Hugging Face
ONNX_URLS = [
    "https://github.com/FlowElement/fanjing-face-recognition/releases/download/v0.1.0/resnet18.onnx",
    "https://huggingface.co/FlowElement/face-parsing/resolve/main/resnet18.onnx",
]

# PyTorch 权重下载地址 (官方 Google Drive)
PTH_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1048576
        total_mb = total_size / 1048576
        sys.stdout.write(f"\r  下载中: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
    else:
        mb = downloaded / 1048576
        sys.stdout.write(f"\r  下载中: {mb:.1f} MB")
    sys.stdout.flush()


def download_onnx_direct() -> Path:
    """直接下载预转换的 ONNX 模型"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > 40:  # 合理大小
            print(f"[BiSeNet] 模型已存在: {out_path} ({size_mb:.1f} MB), 跳过下载")
            return out_path
        else:
            print(f"[BiSeNet] 现有文件大小异常 ({size_mb:.1f} MB), 重新下载")
            out_path.unlink()

    print(f"[BiSeNet] 下载预转换 ONNX 模型...")

    for url in ONNX_URLS:
        print(f"[BiSeNet] 尝试: {url}")
        try:
            urlretrieve(url, str(out_path), reporthook=_progress_hook)
            print()

            size_mb = out_path.stat().st_size / 1048576
            if size_mb > 40:
                print(f"[BiSeNet] 完成: {out_path} ({size_mb:.1f} MB)")
                return out_path
            else:
                print(f"[BiSeNet] 文件大小异常 ({size_mb:.1f} MB), 尝试下一个源")
                out_path.unlink()
        except Exception as e:
            print(f"\n[BiSeNet] 下载失败: {e}")
            continue

    print(f"\n[BiSeNet] 所有下载源均失败")
    print(f"\n[BiSeNet] 备选方案: 使用 --convert 参数从 PyTorch 权重转换")
    print(f"          python scripts/download_bisenet.py --convert")
    return None


def download_and_convert() -> Path:
    """下载 PyTorch 权重并转换为 ONNX"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[BiSeNet] 错误: 需要安装 PyTorch")
        print("          pip install torch")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > 40:
            print(f"[BiSeNet] 模型已存在: {out_path} ({size_mb:.1f} MB), 跳过")
            return out_path

    # 下载 PyTorch 权重
    pth_path = OUTPUT_DIR / "79999_iter.pth"
    if not pth_path.exists():
        print("[BiSeNet] 下载 PyTorch 权重 (从 Google Drive)...")
        try:
            import gdown
        except ImportError:
            print("[BiSeNet] 安装 gdown...")
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown

        gdown.download(id=PTH_GDRIVE_ID, output=str(pth_path), quiet=False)

    if not pth_path.exists():
        print(f"[BiSeNet] 错误: 无法下载 PyTorch 权重")
        return None

    # 定义 BiSeNet 网络结构 (简化版，仅用于推理)
    print("[BiSeNet] 加载模型并转换为 ONNX...")

    # BiSeNet 网络定义
    class ConvBNReLU(nn.Module):
        def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_ch, mid_ch, n_classes):
            super().__init__()
            self.conv = ConvBNReLU(in_ch, mid_ch, 3, 1, 1)
            self.conv_out = nn.Conv2d(mid_ch, n_classes, 1, 1, 0, bias=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.conv_out(x)
            return x

    # 加载预训练权重
    state_dict = torch.load(str(pth_path), map_location="cpu")

    # 这里需要完整的 BiSeNet 定义，比较复杂
    # 简化方案: 直接使用原项目的模型定义
    print("[BiSeNet] 注意: 完整转换需要 face-parsing.PyTorch 项目的模型定义")
    print("[BiSeNet] 推荐使用直接下载方式: python scripts/download_bisenet.py")

    # 清理临时文件
    # pth_path.unlink()

    return None


def verify_model(model_path: Path) -> bool:
    """验证 ONNX 模型是否可用"""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_shape = sess.get_outputs()[0].shape

        print(f"[BiSeNet] 模型验证:")
        print(f"          输入: {input_name} {input_shape}")
        print(f"          输出: {output_shape}")

        # 测试推理
        dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
        out = sess.run(None, {input_name: dummy})
        print(f"          测试推理: OK (输出形状 {out[0].shape})")
        return True

    except Exception as e:
        print(f"[BiSeNet] 模型验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 BiSeNet 人脸分割模型")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="从 PyTorch 权重转换 (需要安装 torch)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/speaking",
        help="模型保存目录 (默认: models/speaking/)",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)

    if args.convert:
        path = download_and_convert()
    else:
        path = download_onnx_direct()

    if path and path.exists():
        verify_model(path)
        print(f"\n[BiSeNet] 说话检测遮挡模型已就绪: {path}")
    else:
        print(f"\n[BiSeNet] 下载失败")
        print(f"\n手动下载方式:")
        print(f"  1. 访问 https://github.com/zllrunning/face-parsing.PyTorch")
        print(f"  2. 下载预训练权重并转换为 ONNX")
        print(f"  3. 将 resnet18.onnx 放入 {OUTPUT_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()

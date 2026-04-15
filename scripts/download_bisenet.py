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
        import torch.nn.functional as F
        from torchvision import models
    except ImportError:
        print("[BiSeNet] 错误: 需要安装 PyTorch")
        print("          pip install torch torchvision")
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
    if not pth_path.exists() or pth_path.stat().st_size < 40 * 1048576:
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

    print("[BiSeNet] 定义模型结构...")

    # BiSeNet 网络定义 (完整版)
    class ConvBNReLU(nn.Module):
        def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                                  stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_chan, mid_chan, n_classes):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
            self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        def forward(self, x):
            return self.conv_out(self.conv(x))

    class AttentionRefinementModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            self.sigmoid_atten = nn.Sigmoid()
        def forward(self, x):
            feat = self.conv(x)
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
            return torch.mul(feat, atten)

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=None)
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        def forward(self, x):
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            feat8 = self.resnet.layer1(x)
            feat8 = self.resnet.layer2(feat8)
            feat16 = self.resnet.layer3(feat8)
            feat32 = self.resnet.layer4(feat16)
            H8, W8 = feat8.size()[2:]
            H16, W16 = feat16.size()[2:]
            H32, W32 = feat32.size()[2:]
            avg = F.adaptive_avg_pool2d(feat32, 1)
            avg = self.conv_avg(avg)
            avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
            feat32_arm = self.arm32(feat32)
            feat32_sum = feat32_arm + avg_up
            feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
            feat32_up = self.conv_head32(feat32_up)
            feat16_arm = self.arm16(feat16)
            feat16_sum = feat16_arm + feat32_up
            feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
            feat16_up = self.conv_head16(feat16_up)
            return feat8, feat16_up, feat32_up

    class FeatureFusionModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()
        def forward(self, fsp, fcp):
            fcat = torch.cat([fsp, fcp], dim=1)
            feat = self.convblk(fcat)
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
            return torch.mul(feat, atten) + feat

    class BiSeNet(nn.Module):
        def __init__(self, n_classes=19):
            super().__init__()
            self.cp = ContextPath()
            self.ffm = FeatureFusionModule(256, 256)
            self.conv_out = BiSeNetOutput(256, 256, n_classes)
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        def forward(self, x):
            H, W = x.size()[2:]
            feat8, feat16_up, feat32_up = self.cp(x)
            feat_fuse = self.ffm(feat8, feat16_up)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
            return feat_out

    print("[BiSeNet] 加载权重...")
    model = BiSeNet(n_classes=19)
    state_dict = torch.load(str(pth_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("[BiSeNet] 导出 ONNX...")
    dummy_input = torch.randn(1, 3, 512, 512)
    temp_path = OUTPUT_DIR / "temp.onnx"

    torch.onnx.export(
        model, dummy_input, str(temp_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11, do_constant_folding=True,
    )

    # 合并为单一文件 (新版 PyTorch 可能会分离权重)
    data_path = Path(str(temp_path) + ".data")
    if data_path.exists():
        print("[BiSeNet] 合并为单一 ONNX 文件...")
        import onnx
        onnx_model = onnx.load(str(temp_path), load_external_data=True)
        onnx.save_model(onnx_model, str(out_path), save_as_external_data=False)
        temp_path.unlink()
        data_path.unlink()
    else:
        temp_path.rename(out_path)

    size_mb = out_path.stat().st_size / 1048576
    print(f"[BiSeNet] 完成: {out_path} ({size_mb:.1f} MB)")
    return out_path


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

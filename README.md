# Fanjing Face Recognition

> [FlowElement](https://github.com/FlowElement) 开源生态组件 — 实时人脸识别与身份管理系统

支持多人跟踪、自动注册、跨 session 身份持久化、可选说话检测。可独立使用，也可作为 [M-Flow](https://github.com/FlowElement/m_flow) Playground 的视觉感知服务。

## 安装

### 方式一：PyPI 安装（推荐）

```bash
# 安装
pip install fanjing-face-recognition

# 下载模型
fanjing-face download-models

# 启动服务
fanjing-face
```

### 方式二：从源码安装

```bash
# 克隆仓库
git clone https://github.com/FlowElement-ai/fanjing-face-recognition.git
cd fanjing-face-recognition

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 下载模型文件（见下方"模型下载"章节）

# 启动服务
python run_web_v2.py

# 浏览器自动打开 http://localhost:5001
```

### CLI 命令

```bash
fanjing-face                      # 启动服务 (默认 localhost:5001)
fanjing-face --host 0.0.0.0       # 监听所有网络接口
fanjing-face --port 8080          # 指定端口
fanjing-face --no-browser         # 不自动打开浏览器
fanjing-face download-models      # 下载所有模型
```

## Docker 部署

### 使用 Docker Hub 镜像

```bash
# 拉取镜像
docker pull flowelement/fanjing-face-recognition:latest

# 创建模型目录并下载模型
mkdir -p models/speaking
python scripts/download_model.py
python scripts/download_arcface.py
python scripts/download_bisenet.py --convert
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 运行容器
docker run -d \
  --name fanjing-face \
  -p 5001:5001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/identities:/app/identities \
  flowelement/fanjing-face-recognition:latest
```

### 使用 Docker Compose

```bash
# 下载模型后，一键启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 本地构建镜像

```bash
# 构建镜像（不含模型）
docker build -t fanjing-face-recognition .

# 构建镜像（含模型，需要较长时间）
docker build --build-arg DOWNLOAD_MODELS=true -t fanjing-face-recognition .
```

## 模型下载

模型文件体积较大，不包含在仓库中。请将以下模型放入 `models/` 目录：

| 模型 | 用途 | 大小 | 路径 |
|------|------|------|------|
| SCRFD det_10g | 人脸检测 | ~16MB | `models/det_10g.onnx` |
| ArcFace w600k_r50 | 人脸嵌入 | ~174MB | `models/w600k_r50.onnx` |
| MediaPipe FaceLandmarker | 关键点检测 | ~4MB | `models/face_landmarker.task` |
| BiSeNet ResNet18 | 面部分割 | ~53MB | `models/speaking/resnet18.onnx` |

### 使用下载脚本

```bash
# 下载 SCRFD 检测模型
python scripts/download_model.py

# 下载 ArcFace 嵌入模型
python scripts/download_arcface.py

# 下载 BiSeNet 面部分割模型 (说话检测需要)
python scripts/download_bisenet.py

# 下载 MediaPipe FaceLandmarker
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

说话检测模型 (`models/speaking/speaking_model.json` + `speaking_meta.json`) 可通过训练生成，详见"说话检测训练"章节。

## 项目结构

```
├── src/
│   ├── ingestion/              # Module 0: 视频采集
│   │   ├── frame.py            # Frame 数据结构
│   │   ├── camera_source.py    # 摄像头源 (生产者-消费者)
│   │   └── video_source.py     # 视频文件源
│   ├── detectors/              # Module 1: 人脸检测
│   │   ├── scrfd_detector.py   # SCRFD 检测器 (ONNX Runtime)
│   │   └── detection.py        # 检测结果数据结构
│   ├── tracking/               # Module 2: 多目标跟踪
│   │   ├── bot_sort.py         # BoT-SORT 跟踪器 (四阶段匹配)
│   │   ├── track.py            # STrack 轨迹定义
│   │   └── draw.py             # 画面绘制 (框/标签/HUD)
│   ├── alignment/              # Module 3: 对齐 + 质量 + 采样
│   │   ├── aligner.py          # 仿射对齐 (Umeyama, 112x112)
│   │   ├── quality.py          # Quality Gate (四维评估)
│   │   └── track_sampler.py    # Best-K 样本管理
│   ├── embedding/              # Module 4-5: 嵌入 + 匹配 + 身份
│   │   ├── embedder.py         # ArcFace R50 嵌入器
│   │   ├── track_template.py   # 模板聚合
│   │   ├── person_registry.py  # Person 匹配
│   │   ├── identity_state.py   # 三态判定 + RegisteredPersonDB
│   │   └── candidate_pool.py   # 候选池
│   ├── speaking/               # 说话检测 (可选模块)
│   │   ├── speaking_analyzer.py  # XGBoost + BiSeNet
│   │   ├── mesh_detector.py      # MediaPipe FaceLandmarker
│   │   └── mouth_worker.py       # 异步 worker 线程
│   └── web/
│       ├── server.py           # Flask 后端 + 主循环
│       └── templates/
│           ├── index_v2.html   # 前端页面 (默认)
│           └── index.html      # 旧版前端 (/legacy)
├── models/                     # 模型文件 (需单独下载)
├── docs/                       # 文档
├── run_web_v2.py               # Web 服务入口
├── record_speaking_data.py     # 说话检测数据录制工具
├── train_speaking_model.py     # 说话检测模型训练
├── requirements.txt            # 核心依赖
├── requirements-training.txt   # 训练附加依赖
└── LICENSE                     # MIT License
```

## 系统架构

```
主线程: 读帧 → SCRFD检测 → BoT-SORT跟踪 → 画框推流 (保证FPS)
                               │
                ┌──────────────┼──────────────┐
                ▼                              ▼
        IdentityWorker                  MouthWorker (可选)
   对齐→质量→采样→embedding         MediaPipe→BiSeNet→XGBoost
   →匹配→身份判定→自动注册          →说话/遮挡检测
```

主线程不等待异步结果，只读缓存标签显示。新人入画时 FPS 不波动。

## 功能模块

| 模块 | 功能 | 开关 |
|------|------|------|
| 检测+跟踪 | SCRFD 人脸检测 + BoT-SORT 跟踪 | 始终开启 |
| 对齐+采样 | 人脸标准化 + 质量评估 + 样本管理 | 前端 checkbox |
| 人员匹配 | ArcFace embedding + Person 匹配 | 前端 checkbox |
| 信用门控 | 防止低质量帧进入 embedding | 前端 checkbox |
| 身份判定 | KS/AMB/US 三态 + 自动注册 + 持久化 | 前端 checkbox |
| 说话检测 | 判断是否在说话 + 遮挡检测 | 前端 checkbox |

## API 接口

所有 POST 接口需要在 Header 中携带 `X-API-Key`（启动时控制台打印），GET 接口无需认证。

| 接口 | 方法 | 认证 | 功能 |
|------|------|------|------|
| `/` | GET | 否 | 前端页面（自动注入 API Key） |
| `/video_feed` | GET | 签名URL | MJPEG 视频流 |
| `/api/start` | POST | API Key | 启动流水线 |
| `/api/stop` | POST | API Key | 停止流水线 |
| `/api/stats` | GET | 否 | 实时统计 |
| `/api/persons` | GET | 否 | 人员列表 |
| `/api/person/rename` | POST | API Key | 重命名人员 |
| `/api/upload_video` | POST | API Key | 上传视频 |

## 环境要求

- Python >= 3.12
- 摄像头（用于实时检测）或视频文件
- 核心依赖安装：`pip install -r requirements.txt`
- 训练附加依赖：`pip install -r requirements-training.txt`

## 说话检测训练

```bash
# 安装训练依赖
pip install -r requirements-training.txt

# 1. 录制数据 (交互式, 按场景独立录制)
python record_speaking_data.py

# 2. 训练模型
python train_speaking_model.py

# 3. 独立测试
python test_full_mouth.py
```

## 安全说明

- 服务默认绑定 `127.0.0.1`，仅本地可访问
- 所有管理类 API 通过 API Key 保护（启动时自动生成）
- 视频流通过签名 URL 保护（5 分钟时效）
- 如需局域网访问，使用 `--host 0.0.0.0` 并确保网络安全
- 可通过环境变量 `FACE_API_KEY` 指定固定 Key
- 视频文件默认只允许从 `uploads/` 目录加载，可通过 `ALLOWED_VIDEO_DIRS` 环境变量扩展

## 隐私与合规

本项目涉及人脸生物特征数据处理，使用前请注意：

- **所有数据均在本地处理**，不上传至任何外部服务器
- 已注册身份数据保存在 `output/registered_db/`，请妥善管理
- `data/recordings/` 中的训练数据不包含在仓库中
- 请遵守当地关于人脸识别的法律法规（如中国《个人信息保护法》、欧盟 GDPR 等）
- 部署到公共场所前，请确保取得相关人员的知情同意

## 详细文档

完整系统逻辑、参数说明、数据流图见 [`docs/系统逻辑文档.md`](docs/系统逻辑文档.md)。

## 贡献

欢迎提交 Issue 和 Pull Request！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

安全漏洞请通过 [SECURITY.md](SECURITY.md) 中的方式私密报告。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。

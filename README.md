# Fanjing Face Recognition

> [FlowElement](https://github.com/FlowElement) Open Source Ecosystem Component — Real-time Face Recognition and Identity Management System

Supports multi-person tracking, automatic registration, cross-session identity persistence, and optional speaking detection. Can be used standalone or as a visual perception service for [M-Flow](https://github.com/FlowElement/m_flow) Playground.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model files (see "Model Download" section below)

# 4. Start Web service
python run_web_v2.py

# 5. Browser automatically opens http://localhost:5001
#    API Key is printed in console at startup, automatically injected into frontend
```

## Docker Deployment

### Using Docker Hub Image

```bash
# Pull image
docker pull flowelement/fanjing-face-recognition:latest

# Create model directory and download models
mkdir -p models/speaking
python scripts/download_model.py
python scripts/download_arcface.py
python scripts/download_bisenet.py --convert
python scripts/download_silero_vad.py
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Run container
docker run -d \
  --name fanjing-face \
  -p 5001:5001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/identities:/app/identities \
  flowelement/fanjing-face-recognition:latest
```

### Using Docker Compose

```bash
# After downloading models, one-click start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Local Image Build

```bash
# Build image (without models)
docker build -t fanjing-face-recognition .

# Build image (with models, takes longer)
docker build --build-arg DOWNLOAD_MODELS=true -t fanjing-face-recognition .
```

## Model Download

Model files are large and not included in the repository. Please place the following models in the `models/` directory:

| Model | Purpose | Size | Path |
|-------|---------|------|------|
| SCRFD det_10g | Face Detection | ~16MB | `models/det_10g.onnx` |
| ArcFace w600k_r50 | Face Embedding | ~174MB | `models/w600k_r50.onnx` |
| MediaPipe FaceLandmarker | Keypoint Detection | ~4MB | `models/face_landmarker.task` |
| BiSeNet ResNet18 | Face Parsing | ~53MB | `models/speaking/resnet18.onnx` |
| Silero VAD | Voice Activity Detection | ~1.2MB | `models/speaking/silero_vad_half.onnx` |

### Using Download Scripts

```bash
# Download SCRFD detection model
python scripts/download_model.py

# Download ArcFace embedding model
python scripts/download_arcface.py

# Download BiSeNet face parsing model (required for speaking detection)
python scripts/download_bisenet.py

# Download Silero VAD model (for ambient voice detection)
python scripts/download_silero_vad.py

# Download MediaPipe FaceLandmarker
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

Speaking detection model (`models/speaking/speaking_model.json` + `speaking_meta.json`) can be generated through training, see "Speaking Detection Training" section.

## Project Structure

```
├── src/
│   ├── ingestion/              # Module 0: Video Capture
│   │   ├── frame.py            # Frame data structure
│   │   ├── camera_source.py    # Camera source (producer-consumer)
│   │   └── video_source.py     # Video file source
│   ├── detectors/              # Module 1: Face Detection
│   │   ├── scrfd_detector.py   # SCRFD detector (ONNX Runtime)
│   │   └── detection.py        # Detection result data structure
│   ├── tracking/               # Module 2: Multi-Object Tracking
│   │   ├── bot_sort.py         # BoT-SORT tracker (4-stage matching)
│   │   ├── track.py            # STrack trajectory definition
│   │   └── draw.py             # Frame drawing (boxes/labels/HUD)
│   ├── alignment/              # Module 3: Alignment + Quality + Sampling
│   │   ├── aligner.py          # Affine alignment (Umeyama, 112x112)
│   │   ├── quality.py          # Quality Gate (4-dimension evaluation)
│   │   └── track_sampler.py    # Best-K sample management
│   ├── embedding/              # Module 4-5: Embedding + Matching + Identity
│   │   ├── embedder.py         # ArcFace R50 embedder
│   │   ├── track_template.py   # Template aggregation
│   │   ├── person_registry.py  # Person matching
│   │   ├── identity_state.py   # Tri-state judgment + RegisteredPersonDB
│   │   └── candidate_pool.py   # Candidate pool
│   ├── speaking/               # Speaking Detection (optional module)
│   │   ├── speaking_analyzer.py  # XGBoost + BiSeNet
│   │   ├── mesh_detector.py      # MediaPipe FaceLandmarker
│   │   └── mouth_worker.py       # Async worker thread
│   └── web/
│       ├── server.py           # Flask backend + main loop
│       └── templates/
│           ├── index_v2.html   # Frontend page (default)
│           └── index.html      # Legacy frontend (/legacy)
├── models/                     # Model files (download separately)
├── docs/                       # Documentation
├── run_web_v2.py               # Web service entry point
├── record_speaking_data.py     # Speaking detection data recording tool
├── train_speaking_model.py     # Speaking detection model training
├── requirements.txt            # Core dependencies
├── requirements-training.txt   # Training additional dependencies
└── LICENSE                     # MIT License
```

## System Architecture

```
Main thread: Read frame → SCRFD detection → BoT-SORT tracking → Draw boxes & stream (ensure FPS)
                               │
                ┌──────────────┼──────────────┐
                ▼                              ▼
        IdentityWorker                  MouthWorker (optional)
   Align→Quality→Sample→Embedding      MediaPipe→BiSeNet→XGBoost
   →Match→Identity judgment→Auto-reg   →Speaking/occlusion detection
```

Main thread doesn't wait for async results, only reads cached labels for display. FPS doesn't fluctuate when new person enters frame.

## Feature Modules

| Module | Function | Toggle |
|--------|----------|--------|
| Detection+Tracking | SCRFD face detection + BoT-SORT tracking | Always on |
| Alignment+Sampling | Face normalization + quality assessment + sample management | Frontend checkbox |
| Person Matching | ArcFace embedding + Person matching | Frontend checkbox |
| Credit Gate | Prevent low-quality frames from entering embedding | Frontend checkbox |
| Identity Judgment | KS/AMB/US tri-state + auto registration + persistence | Frontend checkbox |
| Speaking Detection | Determine if speaking + occlusion detection | Frontend checkbox |

## API Endpoints

All POST endpoints require `X-API-Key` header (printed in console at startup), GET endpoints don't require authentication.

| Endpoint | Method | Auth | Function |
|----------|--------|------|----------|
| `/` | GET | No | Frontend page (auto-injects API Key) |
| `/video_feed` | GET | Signed URL | MJPEG video stream |
| `/api/start` | POST | API Key | Start pipeline |
| `/api/stop` | POST | API Key | Stop pipeline |
| `/api/stats` | GET | No | Real-time statistics |
| `/api/persons` | GET | No | Person list |
| `/api/person/rename` | POST | API Key | Rename person |
| `/api/upload_video` | POST | API Key | Upload video |

## Requirements

- Python >= 3.12
- Camera (for real-time detection) or video file
- Core dependencies: `pip install -r requirements.txt`
- Training dependencies: `pip install -r requirements-training.txt`

## Speaking Detection Training

```bash
# Install training dependencies
pip install -r requirements-training.txt

# 1. Record data (interactive, record separately per scenario)
python record_speaking_data.py

# 2. Train model
python train_speaking_model.py

# 3. Standalone test
python test_full_mouth.py
```

## Security Notes

- Service binds to `127.0.0.1` by default, only accessible locally
- All management APIs protected by API Key (auto-generated at startup)
- Video stream protected by signed URL (5-minute expiration)
- For LAN access, use `--host 0.0.0.0` and ensure network security
- Can specify fixed Key via `FACE_API_KEY` environment variable
- Video files only allowed from `uploads/` directory by default, expandable via `ALLOWED_VIDEO_DIRS` environment variable

## Privacy and Compliance

This project involves facial biometric data processing. Please note before use:

- **All data processed locally**, not uploaded to any external servers
- Registered identity data saved in `output/registered_db/`, please manage carefully
- Training data in `data/recordings/` not included in repository
- Please comply with local facial recognition laws (e.g., China's Personal Information Protection Law, EU GDPR)
- Before deploying in public places, ensure informed consent from relevant individuals

## Detailed Documentation

Complete system logic, parameter descriptions, and data flow diagrams available in [`docs/system_logic.md`](docs/system_logic.md).

## Contributing

Issues and Pull Requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Security vulnerabilities should be reported privately via [SECURITY.md](SECURITY.md).

## License

This project is open source under [Apache License 2.0](LICENSE).

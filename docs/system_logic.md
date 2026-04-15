# FlowElement Face Recognition System — Complete Logic Documentation

> Version: Module 6 (Auto Registration + Speaking Detection)  
> Updated: 2026-04-10  
> This document describes the complete system processing flow in natural language, without code.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module 0: Video Input Layer](#2-module-0-video-input-layer)
3. [Module 1: Face Detection](#3-module-1-face-detection)
4. [Module 2: Multi-Object Tracking](#4-module-2-multi-object-tracking)
5. [Module 3: Face Alignment + Quality Assessment + Sampling](#5-module-3-face-alignment--quality-assessment--sampling)
6. [Module 4: Face Embedding + Template Aggregation + Person Matching](#6-module-4-face-embedding--template-aggregation--person-matching)
7. [Module 5: Tri-State Identity Judgment + Auto Registration](#7-module-5-tri-state-identity-judgment--auto-registration)
8. [Credit Gate: Credit Score Gating](#8-credit-gate-credit-score-gating)
9. [Speaking Detection Module](#9-speaking-detection-module)
10. [Performance Protection Mechanisms](#10-performance-protection-mechanisms)
11. [Web Frontend + Main Loop](#11-web-frontend--main-loop)
12. [Complete Data Flow Summary](#12-complete-data-flow-summary)

---

## 1. System Overview

This system is a **real-time face recognition and identity management pipeline** that reads frames from cameras or video files and processes them frame by frame.

**Dual-Thread Architecture:**

```
Main Thread: Read frame → Detection → Tracking → Draw boxes → JPEG streaming (ensure smooth FPS)
Async Thread 1 (IdentityWorker): Alignment → Quality → Sampling → Embedding → Matching → Identity judgment
Async Thread 2 (MouthWorker): MediaPipe → BiSeNet occlusion → XGBoost speaking detection (optional)
```

The main thread **does not wait** for any identity/speaking results, only reads the latest cached labels for display. This ensures FPS doesn't fluctuate when new people enter the frame or during re-identification.

**Core Design Principles:**
- **Fail-open**: Any downstream module error doesn't affect the video stream
- **Progressive layers**: Each module only triggers when the previous module produces valid results
- **Async decoupling**: Identity computation and frame rendering are completely separated

---

## 2. Module 0: Video Input Layer

**Responsibility**: Unify camera and video file inputs, providing a standard "Frame" object externally.

### 2.1 Frame Object

Each frame contains:
- **Image data**: Pixel matrix in BGR format
- **Timestamp**: Millisecond precision. Camera uses system monotonic clock, video uses container playback position
- **Frame ID**: Incrementing from 0
- **Source identifier**: e.g., `camera:0` or `file:demo.mp4`
- **Width/Height**, **Dropped frames count**

### 2.2 Camera Source

Uses **producer-consumer model** with frame buffer size of 1, keeping only the latest frame. Main thread always gets the latest frame without delay accumulation.

### 2.3 Video File Source

Reads frames sequentially, timestamps from video container. Provides `realtime` switch to control playback speed limiting.

---

## 3. Module 1: Face Detection

**Responsibility**: Find all face positions, confidence scores, and keypoints in each frame.

### 3.1 Detector: SCRFD

Uses **SCRFD** model (`det_10g.onnx`) with ONNX Runtime inference.

- **Input**: Original image scaled proportionally to specified size (e.g., 640×640) with letterbox padding
- **Output**: For each face: bbox [x1,y1,x2,y2], score (0~1), 5 keypoints (left eye, right eye, nose tip, left mouth corner, right mouth corner)
- **Thread limit**: `intra_op_num_threads = max(cpu_count - 2, 4)`, reserving CPU cores for async thread's ArcFace inference, eliminating cold-start FPS fluctuation

### 3.2 Four-Threshold Separation Architecture

| Threshold | Name | Default | Purpose |
|-----------|------|---------|---------|
| **A** | `detector_emit_min_det` | 0.15 | Minimum score for detector to emit to tracker |
| **B** | `track_new_high_thres` | 0.50 | Only high-score boxes can create new tracks |
| **C** | `track_update_low_thres` | 0.15 | Low-score boxes can update existing/lost tracks |
| **D** | `sample_min_det` | 0.60 | Minimum score for alignment/embedding/identity |

Weak detection boxes (0.15~0.50) are used by tracker to maintain tracking continuity, but cannot create new tracks or enter embedding pipeline.

---

## 4. Module 2: Multi-Object Tracking

**Responsibility**: Associate frame-by-frame detections into continuous trajectories, assigning stable `track_id`.

### 4.1 Algorithm: BoT-SORT (Four-Stage Matching)

1. **Prediction**: Kalman filter predicts all track positions
2. **Stage 1**: High-score detections vs active tracks (IoU matching)
3. **Stage 2**: Low-score weak boxes vs unmatched active tracks (weak boxes maintain existing tracks)
4. **Stage 3**: Unmatched high-score detections vs lost tracks (recovery)
5. **Stage 4**: Low-score weak boxes vs lost tracks (weak box revival)
6. **Creation**: Still unmatched high-score detections (score ≥ B) create new tracks. Low-score boxes never create new tracks

### 4.2 Track Lifecycle

```
New detection → TENTATIVE → CONFIRMED → LOST → REMOVED
```

- TENTATIVE → CONFIRMED: Matched for `min_hits` (default 3) consecutive frames
- CONFIRMED → LOST: Not matched in detection frame
- LOST → CONFIRMED: Re-matched (revival)
- LOST → REMOVED: Not matched for more than `max_age` (default 30) detection frames

### 4.3 Non-Detection Frame Handling

When `det_every_n > 1`, non-detection frames only perform Kalman prediction to estimate positions, significantly improving FPS.

---

## 5. Module 3: Face Alignment + Quality Assessment + Sampling

**Responsibility**: Normalize faces, assess quality, retain high-quality samples. Executed in **IdentityWorker async thread**.

### 5.1 Face Alignment

Uses 5 keypoints + Umeyama algorithm to map faces at any angle to standard **112×112** frontal face image. Supports rotated/tilted faces (e.g., lying down usage).

### 5.2 Quality Assessment (Quality Gate)

| Dimension | Description | Default Threshold |
|-----------|-------------|-------------------|
| **Detection confidence** | SCRFD raw score | ≥ 0.60 |
| **Face area** | Original bbox pixel area | ≥ 900 px² (30×30) |
| **Sharpness** | Laplacian variance of aligned image | ≥ 15.0 |
| **Keypoint validity** | Eye distance ≥ 20px | Pass check |

All four dimensions must pass to qualify. Also calculates weighted composite score for sorting.

### 5.3 Sample Management

Each track maintains best-K sample cache (default max 10 images), replacing the worst when new sample has higher quality.

---

## 6. Module 4: Face Embedding + Template Aggregation + Person Matching

**Responsibility**: Convert face images to vectors, aggregate templates, session identity matching. Executed in **IdentityWorker async thread**.

### 6.1 Embedding Extraction (ArcFace)

- Model: `w600k_r50.onnx`, outputs 512-dimensional normalized vector
- Thread limit: `intra_op_num_threads=2`, not competing with SCRFD
- Cosine similarity between two faces: Same person typically > 0.5, different people typically < 0.3

### 6.2 Template Aggregation

Collects multiple embeddings from same track, aggregates with quality-weighted average. Generates template only after at least 3 embeddings.

### 6.3 Person Matching (Person Registry)

New track template computes cosine similarity with known person templates:
- top1 ≥ 0.4 and margin ≥ 0.1 → Match successful, bind to existing person
- Otherwise → Create new person

After matching, person template updates with sliding average.

---

## 7. Module 5: Tri-State Identity Judgment + Auto Registration

**Responsibility**: Determine if each person is known identity or stranger, auto-register and persist across sessions.

### 7.1 Tri-State Identity Judgment

For each session person, compare against **RegisteredPersonDB (registered identity database)**:

| State | Abbrev | Trigger Condition |
|-------|--------|-------------------|
| **KNOWN_STRONG** | KS | top1 similarity ≥ 0.55 and margin ≥ 0.10 |
| **AMBIGUOUS** | AMB | Similarity in [0.35, 0.55) or insufficient margin |
| **UNKNOWN_STRONG** | US | Similarity < 0.35, or registered DB is empty |

### 7.2 Auto Registration

- **UNKNOWN_STRONG and register_ready** → Auto-register as R#y (new registered identity), recognized as KS in subsequent sessions
- **register_ready composite score**: Weighted score of sample count, consistency, average quality, duration ≥ threshold
- **AMBIGUOUS timeout registration**: Stays in AMB over 15 seconds and register_ready → Force registration

### 7.3 KNOWN_STRONG Template Update

KS person conservatively updates its registered identity template (weight 0.2), making known database more accurate with use.

### 7.4 Auto Merge

After template update, triggers `check_and_merge()`: If two registered identities have similarity ≥ 0.75, merge newer into older, update all mappings.

### 7.5 Persistence

- `RegisteredPersonDB` auto-saves on pipeline stop, auto-loads on startup
- Storage path: `data/registered_db/`

---

## 8. Credit Gate: Credit Score Gating

**Responsibility**: Prevent frames that occasionally pass Quality Gate from directly entering embedding. Does not delete any tracks.

Each track maintains `face_valid_credit` (float, initial 0):

- Quality passes → credit += 1.0
- Quality fails → credit -= 0.5
- Floor 0, ceiling 10.0
- credit ≥ 3.0 allows embedding

Typical scenario: 3 consecutive good frames → credit 3.0 reaches threshold; walking from distance → credit accumulates naturally → auto-qualifies.

---

## 9. Speaking Detection Module

**Responsibility**: Determine if each tracked person is speaking. **Optional feature**, frontend has toggle, off by default.

### 9.1 Architecture

Runs in separate async threads, doesn't affect main thread FPS:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MouthWorker Thread (Visual Detection):                                       │
│   face_crop → MediaPipe (blendshapes + yaw) → BiSeNet (occlusion)            │
│            → XGBoost (speaking) → Hysteresis (debounce)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AudioVAD Thread (Ambient Voice Detection):                                   │
│   microphone → Silero VAD (ONNX) → has_voice (True/False)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Result Merge:                                                                │
│   if has_voice == False:                                                     │
│       override all "speaking" → "not_speaking" (environment is silent)       │
│   else:                                                                      │
│       use visual detection results as-is                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 AudioVAD (Ambient Voice Activity Detection)

**Purpose**: Prevent false positive speaking detections when environment is silent.

When the visual model detects mouth movement (e.g., chewing, yawning), but no voice is present in the environment, AudioVAD overrides the result to "not_speaking".

**Technical Details:**
- **Model**: Silero VAD half-precision ONNX (~1.2MB)
- **Input**: 512 samples at 16kHz (32ms per frame) from system default microphone
- **Output**: Speech probability 0.0~1.0 per frame
- **Threshold**: 0.3 (any of last 6 frames > 0.3 → has_voice = True)
- **Fallback**: If microphone unavailable, VAD silently degrades and doesn't interfere with visual detection

### 9.3 Visual Three-Layer Judgment

**L1 Occlusion Judgment:**
- |yaw| > 60° → `SELF_OCCLUDED` (extreme side face)
- BiSeNet face parsing detects lip pixel ratio in mouth region (runs once per 5 calls per track, reuses in between)
  - lip_ratio < 0.10 → `OCCLUDED`
  - lip_ratio 0.10~0.25 → PARTIAL (prob reduced by 60%)

**L2 Speaking Probability:**
- MediaPipe FaceLandmarker extracts 27 mouth blendshapes + yaw/pitch/roll
- Maintains ring buffer of last 12 frames
- Computes sliding window features (current/mean/std/range/diff_mean/min/max = 210 dimensions)
- XGBoost model outputs speaking_prob (0~1)
- Model trained on 52000 recorded frames, F1 = 0.948

**L3 Hysteresis State Machine (data-driven parameters):**
- prob ≥ 0.70 for 3 consecutive frames → `SPEAKING`
- prob < 0.30 for 3 consecutive frames → `NOT_SPEAKING`
- At 45°+ side face, adaptively increases confirmation frames (add 1 frame per 5° beyond)

### 9.3 Output

| State | Label | Meaning |
|-------|-------|---------|
| speaking | SPK | Currently speaking |
| not_speaking | (no label) | Not speaking |
| occluded | OCC | Mouth occluded |
| self_occluded | S-OCC | Extreme side face |
| unobservable | ? | Cannot detect face |

### 9.4 Performance Impact

- When off: Zero overhead
- When on: Runs in async thread, doesn't block main loop. BiSeNet runs every 5 frames (~60ms/call), XGBoost <0.1ms

---

## 10. Performance Protection Mechanisms

### 10.1 Three-Layer Embedding Rate Limiting

| Layer | Mechanism | Default |
|-------|-----------|---------|
| Per-frame limit | Max N embeddings per frame | 5 per frame |
| Per-track total limit | Stop after enough for template | 5 per track |
| Time cooldown | Minimum interval between embeddings for same track | 200ms |

### 10.2 Async Identity Computation (IdentityWorker)

Alignment, quality assessment, sampling, embedding, template aggregation, person matching, identity judgment all moved from main loop to async thread. Main thread only reads cached labels. FPS doesn't fluctuate when new person enters.

### 10.3 CPU Core Allocation

- SCRFD detector: `cpu_count - 2` cores
- ArcFace embedder: 2 cores
- They don't compete with each other, eliminating cold-start FPS drop

### 10.4 Fail-Open

Each step has independent exception handling, any error only logs without affecting video stream.

---

## 11. Web Frontend + Main Loop

### 11.1 Architecture

- Flask web framework, REST API + MJPEG video stream
- Single-page HTML frontend, AJAX interaction

### 11.2 Main Loop (Per Frame)

```
1. Read frame
2. Determine if detection frame
3. Detection frame: SCRFD detection + BoT-SORT tracking
   Non-detection frame: Kalman prediction only
4. Async submit: IdentityWorker (identity) + MouthWorker (speaking)
5. Read async result cache
6. Draw tracking boxes + labels + HUD
7. JPEG encode → MJPEG streaming
```

### 11.3 Label Meanings

```
T#102 P#1 KS R#1 C3 SPK (0.87)
```

| Part | Meaning |
|------|---------|
| `T#102` | Track ID |
| `P#1` | Person ID |
| `KS` | Identity state (KS=Known, AMB=Ambiguous, US=Unknown) |
| `R#1` | Registered identity ID |
| `C3` | Credit score |
| `SPK` | Speaking state (SPK=Speaking, OCC=Occluded, S-OCC=Self-occluded) |
| `(0.87)` | Cosine similarity |

### 11.4 Frontend Toggles

| Toggle | Controls |
|--------|----------|
| Enable face alignment sampling | Module 3 |
| Enable person matching (T#→P#) | Module 4 |
| Enable credit score system | Credit Gate |
| Enable candidate pool (C-pool) | Module 5 |
| Enable speaking detection (SPK/OCC) | Speaking detection module |

---

## 12. Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│  Camera / Video File                                                 │
│     │                                                                │
│     ▼                                                                │
│  Frame {image, timestamp_ms, frame_id}                              │
│     │                                                                │
│     ▼                                                                │
│  [Main Thread] SCRFD Detection (every N frames) → BoT-SORT Tracking │
│     │                              │                                 │
│     │         ┌────────────────────┤                                 │
│     │         │                    │                                 │
│     │         ▼                    ▼                                 │
│     │  [Async] IdentityWorker  [Async] MouthWorker (optional)       │
│     │    Align → Quality → Sample   MediaPipe blendshapes           │
│     │    → Credit Gate              → BiSeNet occlusion             │
│     │    → ArcFace embedding        → XGBoost speaking              │
│     │    → Template aggregation     → Hysteresis                    │
│     │    → Person matching          → status cache                  │
│     │    → M5 identity judgment                                     │
│     │    → Auto register/merge                                      │
│     │    → Result cache                                             │
│     │         │                    │                                 │
│     │         └────────┬───────────┘                                 │
│     │                  │                                             │
│     ▼                  ▼                                             │
│  [Main Thread] Read cached labels → Draw frame → JPEG → Browser    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Configurable Parameters Quick Reference

### Four-Threshold Separation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector_emit_min_det` | 0.15 | Minimum score for detector to emit to tracker |
| `track_new_high_thres` | 0.50 | Minimum score for creating new tracks |
| `track_update_low_thres` | 0.15 | Minimum score for weak boxes to update existing tracks |
| `sample_min_det` | 0.60 | Minimum score for alignment/embedding |

### Detection / Tracking
| Parameter | Default | Description |
|-----------|---------|-------------|
| `det_size` | 640 | Detection input size |
| `det_every_n` | 1 | Detect every N frames |
| `max_age` | 30 | Lost track survival frames |
| `min_hits` | 3 | Consecutive matches needed for confirmation |

### Quality / Sampling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_bbox_area` | 900 | Minimum face area (30×30 px²) |
| `min_blur_score` | 15.0 | Minimum sharpness |
| `max_samples` | 10 | Maximum samples per track |

### Embedding
| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.4 | Person matching threshold |
| `embed_cooldown_ms` | 200 | Minimum embedding interval for same track |
| `max_embed_per_frame` | 5 | Maximum embeddings per frame |
| `max_embed_per_track` | 5 | Maximum embeddings per track |

### Credit Gate
| Parameter | Default | Description |
|-----------|---------|-------------|
| `credit_increment` | 1.0 | Score added when quality passes |
| `credit_decrement` | 0.5 | Score deducted when quality fails |
| `credit_threshold` | 3.0 | Minimum credit for embedding |
| `credit_max` | 10.0 | Credit ceiling |

### Module 5 Identity Judgment
| Parameter | Default | Description |
|-----------|---------|-------------|
| `known_threshold` | 0.55 | Minimum similarity for KS judgment |
| `band_threshold` | 0.35 | Ambiguous zone lower bound |
| `amb_timeout_sec` | 15 | AMBIGUOUS timeout registration seconds |
| `merge_threshold` | 0.75 | Registered identity auto-merge threshold |

### Speaking Detection (XGBoost, data-driven parameters)
| Parameter | Value | Source |
|-----------|-------|--------|
| `on_thresh` | 0.70 | Grid search optimal |
| `off_thresh` | 0.30 | Grid search optimal |
| `on_frames` | 3 | Grid search optimal |
| `off_frames` | 3 | Grid search optimal |
| `window_size` | 12 frames | Speed/accuracy tradeoff |
| `bisenet_every_n` | 5 | Performance optimization |

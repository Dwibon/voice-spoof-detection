# False Shepherd — Real-Time AI Voice Spoof Detection

> **When the voice isn't who it claims to be.**

A real-time AI voice spoof detection system designed to identify AI-generated, cloned, and manipulated speech and convert detection signals into an actionable risk assessment.

False Shepherd combines self-supervised speech representations, lightweight classification, temporal aggregation, prosodic analysis, speaker consistency checks, explainable risk scoring, and a privacy-aware API/UI layer.

---

## Overview

Voice cloning has made voice alone an increasingly unreliable trust signal.

False Shepherd analyzes suspicious speech and produces a structured security assessment:

- **Spoof probability**
- **Risk score**
- **Risk level — LOW / MEDIUM / HIGH**
- **Decision confidence**
- **Window-level detection information**
- **Explanation of detected signals**
- **Recommended action**
- **Optional speaker-consistency verification**

The system is designed for security-sensitive scenarios such as fraud prevention, call-center verification, banking workflows, and identity-sensitive voice interactions.

---

## Key Capabilities

### AI Voice Spoof Detection

The core detector uses **Wav2Vec2 speech representations** with a lightweight binary classifier to distinguish bonafide speech from spoofed speech.

### Real-Time Analysis

Audio is analyzed using overlapping short windows:

```text
4-second analysis window
        ↓
1-second step
        ↓
Window-level spoof probability
        ↓
3-window rolling aggregation
        ↓
Current risk assessment
```

This enables continuously updated risk assessment rather than relying only on a single score for an entire recording.

### Indian-Accent Domain Adaptation

The detection model includes targeted domain adaptation using Indian-accented speech, improving robustness for the intended regional deployment context. The adapted model is evaluated on speaker-disjoint held-out Indian speech.

### Prosodic Analysis

Lightweight prosodic features provide an additional signal based on characteristics such as:

- Pitch statistics
- Voicing behaviour
- Pause patterns
- Speech-rate characteristics

The prosodic component acts as an auxiliary detection signal.

### Speaker Consistency Verification

An optional enrolled-speaker reference can be supplied. The system uses an ECAPA-TDNN speaker embedding model and cosine similarity to determine whether the analyzed recording is consistent with the enrolled speaker. This provides an additional identity-consistency signal alongside spoof detection.

### Explainable Risk Assessment

Raw model outputs are converted into an actionable risk assessment:

| Risk Score | Risk Level | Recommended Action |
|---|---|---|
| < 30 | LOW | PROCEED |
| 30–69 | MEDIUM | VERIFY |
| ≥ 70 | HIGH | ESCALATE |

The interface also provides explanation tags describing why the risk assessment was raised.

### Privacy-Aware Processing

Uploaded audio is processed using temporary files.

```text
Upload
  ↓
Temporary storage
  ↓
Inference
  ↓
Result returned
  ↓
Temporary audio deleted
```

Raw audio is not intentionally persisted by the detection API.

---

## System Architecture

```text
                    AUDIO INPUT
                         │
                         ▼
              ┌─────────────────────┐
              │ Audio Preprocessing  │
              │ 16 kHz / mono        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Windowing            │
              │ 4 s / 1 s step       │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Wav2Vec2          │    │ Prosodic         │
    │ Speech Features   │    │ Features         │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Spoof Classifier  │    │ Auxiliary        │
    │                   │    │ Prosody Signal   │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
               ┌──────────────────┐
               │ Rolling Temporal │
               │ Aggregation      │
               └────────┬─────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Risk Engine       │   │ Speaker          │
    │ 0–100 score       │   │ Consistency      │
    │ LOW/MED/HIGH      │   │ ECAPA-TDNN       │
    └────────┬─────────┘   └────────┬─────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ Explanation &         │
             │ Recommendation        │
             └──────────┬───────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Web UI / REST API  │
              └────────────────────┘
```

---

## Evaluation

The system has been evaluated across benchmark, Indian-accent, and cross-dataset speech conditions.

**Final Detection Model**

| Evaluation Set | Accuracy | ROC-AUC | EER |
|---|---|---|---|
| ASVspoof2019 LA | 92.15% | 0.9851 | 4.93% |
| Indian held-out test | 96.88% | 0.9932 | 3.12% |
| In-The-Wild | 55.28% | 0.5755 | 42.17% |

The Indian held-out evaluation contains 64 recordings from 8 unseen speakers, with speaker-disjoint train, validation, and test splits. The ASVspoof2019 LA evaluation provides the primary benchmark assessment, while In-The-Wild provides an additional cross-dataset generalization evaluation.

---

## Model

The primary detection pipeline uses:

```text
Audio
  ↓
16 kHz preprocessing
  ↓
Wav2Vec2
  ↓
768-dimensional speech representation
  ↓
Lightweight classifier head
  ↓
Spoof probability
  ↓
Rolling temporal aggregation
  ↓
Risk assessment
```

The classifier head is initialized from the ASVspoof-trained model and adapted for the target Indian-accent domain.

**Primary Model:** `models/classifier_head_indian_adapted.pt`

---

## Web Interface

False Shepherd provides a browser-based interface for interactive analysis.

The dashboard provides:

- Audio upload
- Optional enrolled-speaker reference
- Real-time detection status
- Risk level
- Risk score
- Spoof probability
- Decision confidence
- Number of analyzed windows
- Explanation
- Recommended action
- Privacy status

The application also provides navigation for:

- API Docs
- Health
- Webhook Logs

---

## REST API

The backend is implemented using FastAPI.

**Health**
```
GET /health
```
Returns service health information.

**Detect Audio**
```
POST /detect
```
Accepts an audio file and returns the complete detection result.

**Detect With Speaker Reference**
```
POST /detect-with-reference
```
Accepts:
- Target audio
- Enrolled speaker reference

...and performs spoof detection together with speaker-consistency analysis.

**Mock Webhook**
```
POST /mock-webhook
```
Provides a demonstration integration endpoint for receiving high-risk voice spoof alerts.

**Webhook Logs**
```
GET /mock-webhook/logs
```
Returns the webhook events recorded by the demonstration integration.

**Interactive API Documentation**

When the server is running, FastAPI automatically provides interactive API documentation.

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/Dwibon/voice-spoof-detection.git
cd voice-spoof-detection
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the application**
```bash
uvicorn src.api:app --reload
```

Open: `http://localhost:8000`

---

## Command-Line Detection

The detector can also be used directly from the command line.

**Audio only**
```bash
python3 src/realtime_detect.py <audio_file>
```

**Audio with speaker reference**
```bash
python3 src/realtime_detect.py <audio_file> <reference_audio>
```

---

## Project Structure

```text
voice-spoof-detection/
│
├── data/
│   └── datasets and local evaluation data
│
├── docs/
│   └── project documentation
│
├── models/
│   ├── classifier_head_baseline.pt
│   ├── classifier_head_best.pt
│   ├── classifier_head_indian_adapted.pt
│   ├── prosody_classifier.npz
│   └── risk_calibration.npz
│
├── notebooks/
│   └── research and analysis notebooks
│
├── outputs/
│   └── generated evaluation outputs
│
├── scripts/
│   └── utility scripts
│
├── src/
│   ├── api.py
│   ├── dataset.py
│   ├── model.py
│   ├── realtime_detect.py
│   ├── risk_engine.py
│   ├── speaker_check.py
│   ├── explanation_engine.py
│   ├── prosody_features.py
│   └── evaluation/training utilities
│
├── ui/
│   └── web interface
│
├── requirements.txt
└── README.md
```

---

## Technology Stack

- Python
- PyTorch
- Wav2Vec2
- SpeechBrain / ECAPA-TDNN
- librosa
- scikit-learn
- FastAPI
- Uvicorn
- HTML / CSS / JavaScript
- NumPy
- SoundFile

---

## Security & Privacy

The system follows a privacy-aware processing model:

- Raw uploaded audio is handled temporarily.
- Temporary files are removed after inference.
- Detection responses contain derived analysis rather than raw audio.
- Speaker-reference files are also cleaned after processing.
- High-risk results can be routed to the demonstration webhook layer for downstream action.

---

## Intended Applications

False Shepherd is designed as a detection and decision-support layer for scenarios including:

- Banking and financial fraud prevention
- Call-center security
- Voice-based identity verification
- Social engineering defense
- Remote customer verification
- High-risk voice interactions
- AI-generated voice screening

---

## Project Status

False Shepherd currently provides an integrated voice-spoof detection pipeline consisting of:

- AI voice spoof classification
- Targeted Indian-accent domain adaptation
- Windowed and rolling detection
- Prosodic analysis
- Speaker consistency verification
- Risk scoring
- Explainable alerts
- Privacy-aware audio processing
- Browser-based analysis interface
- FastAPI REST API
- Interactive API documentation
- Mock webhook integration

The architecture is designed so that additional live audio sources, communication channels, and external integrations can be connected without changing the core detection pipeline.

---

## Team

**Team Touch Grass**
Smart India Hackathon 2026
Problem Statement: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

---

## License

This project is developed as a research and hackathon prototype.
```
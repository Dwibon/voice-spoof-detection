# SIH25104 — Voice Spoof Detection

Real-Time AI Voice Cloning / Impersonation Detection.

## What is already built
- O1: Wav2Vec2 embeddings + lightweight bonafide/spoof classifier on ASVspoof2019 LA.
- O2: Frozen cross-dataset evaluation workflow.
- O3: 4-second windows, 1-second step, rolling 3-window aggregation.
- O4: Low/Medium/High risk engine + numeric decision confidence.
- O5: Rule-based explanations.
- O8: Lightweight prosodic features/classifier.
- O9: ECAPA speaker-consistency check using cosine similarity.
- O10: Privacy-safe result filtering and temporary-file cleanup utilities.
- O11: Risk-tier recommendations.
- Remaining integration: O6 web UI and O7 FastAPI/OpenAPI/SDK/mock webhook.

## Running the Application

Start the FastAPI server which also serves the Web UI on port 8000:
```bash
uvicorn src.api:app --reload
```
Then open `http://localhost:8000/` in your browser.

## Main detector (CLI)
```bash
python3 src/realtime_detect.py <audio_file>
python3 src/realtime_detect.py <audio_file> <reference_audio>
```

Primary model: `models/classifier_head_best.pt`

Pipeline:
Audio → 16 kHz preprocessing → overlapping windows → Wav2Vec2 → 768-D embedding → classifier → rolling aggregation → risk score → explanation → recommendation.

Optional reference audio adds ECAPA speaker similarity.

## Risk rules
- `< 0.30` → LOW → PROCEED
- `0.30–<0.70` → MEDIUM → VERIFY
- `>= 0.70` → HIGH → ESCALATE

The displayed confidence is a decision-confidence measure, not a calibrated posterior probability.

## API/UI handoff
Build a thin integration layer around the existing detector. Do not retrain or replace the primary model.

Suggested REST endpoints:
- `GET /health`
- `POST /detect`
- `POST /detect-with-reference`

The API should return spoof probability, risk score/level, confidence, explanation, recommendation, window information, and optional speaker consistency. Never return raw audio.

The UI should support upload/record → analyze → risk report. Show the risk tier, score, explanation, recommendation, optional speaker similarity, and privacy status.

The proposal calls for FastAPI + OpenAPI + generated Python/JS SDK stubs + one mock webhook. gRPC and real banking/telecom/contact-center integrations are future work.

## Privacy
Upload → temporary file → inference → response → DELETE temporary file.

Never persist raw audio or transcripts.

## Verified results
- ASVspoof2019 LA eval: 95.46% accuracy, 4.74% EER.
- In-The-Wild cross-dataset run: 37.86% accuracy, 37.52% EER.
- Prosodic auxiliary classifier: 83.62% eval accuracy, ROC-AUC 0.8966.
- Speaker-pair calibration: threshold 0.2132, accuracy 81.40%.

**Important:** the current repository's cross-dataset script/result is In-The-Wild. Do not call that result IndieFake unless IndieFake is actually rerun and verified.

## Scope boundaries
Do not claim the demo solves full live streaming stability, telephony/VoIP robustness, broad Indic-language/fairness coverage, neural-codec attacks, multimodal verification, gRPC/production integrations, live banking/CRM enrichment, configurable enterprise workflows, real SMS/email dispatch, formal compliance certification, or true edge/on-device deployment.

See `AI_HANDOFF.md` and the project proposal in `docs/` for the complete scope.

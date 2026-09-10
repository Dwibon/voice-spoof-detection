# AI HANDOFF — SIH25104

## Read this first
Read this file and the project proposal in `docs/` before changing code. Then inspect the repository.

**The ML pipeline is already built.** Your main task is integration: UI + FastAPI REST API + OpenAPI/SDK generation + mock webhook + privacy-safe upload lifecycle.

Do NOT retrain the primary model, replace the detector architecture, invent metrics, or present future work as implemented.

## Objectives
O1 classifier on ASVspoof2019 LA using Wav2Vec2 + lightweight head.
O2 frozen cross-dataset evaluation.
O3 simulated real-time window inference + rolling aggregation.
O4 Low/Medium/High risk score + numeric confidence.
O5 rule-based explanations.
O6 upload/record demo UI.
O7 FastAPI REST + OpenAPI + SDK stubs + mock webhook.
O8 lightweight prosodic features.
O9 enrolled-reference speaker consistency.
O10 privacy-by-design.
O11 alert/recommendation mapping.

## Existing files
- `src/realtime_detect.py` — primary end-to-end detector/CLI.
- `src/model.py` — Wav2Vec2 classifier definition.
- `src/risk_engine.py` — risk/recommendation logic.
- `src/explanation_engine.py` — explanations.
- `src/prosody_features.py` — six prosodic features.
- `src/speaker_check.py` — SpeechBrain ECAPA + cosine similarity.
- `src/privacy.py` — privacy policy, cleanup and result validation.
- `models/classifier_head_best.pt` — primary trained model.
- `models/prosody_classifier.npz` — auxiliary prosody model.
- `models/risk_calibration.npz` — experimental calibration artifact; currently NOT used.

## Detector behavior
Default window = 4 seconds.
Step = 1 second.
Rolling aggregation = 3 windows.
Audio target sample rate = 16 kHz.
Primary embedding = mean-pooled Wav2Vec2, 768 dimensions.

Typical commands:
```bash
python3 src/realtime_detect.py AUDIO
python3 src/realtime_detect.py AUDIO REFERENCE_AUDIO
```

## Output contract
Return derived information such as:
```json
{
  "spoof_probability": 0.87,
  "spoof_percentage": 87.0,
  "risk_score": 87.0,
  "risk_level": "HIGH",
  "confidence": 74.0,
  "speaker_similarity": 0.34,
  "speaker_consistency": "MISMATCH",
  "explanation": {"summary": "...", "evidence": ["..."]},
  "recommendation": {"action": "ESCALATE", "message": "..."},
  "windows_analyzed": 7,
  "rolling_window_size": 3
}
```
Follow the actual Python implementation for exact fields. Never include audio bytes, waveform, raw-audio paths, or transcripts.

## Risk
`<0.30` LOW → PROCEED.
`0.30–<0.70` MEDIUM → VERIFY.
`>=0.70` HIGH → ESCALATE.

Confidence is decision confidence, not a statistically calibrated probability.

## Speaker consistency
ECAPA cosine-similarity threshold = 0.2132.
This indicates consistency with an enrolled voice; it is NOT proof that speech is AI-generated.

## Prosody
Features:
- pitch mean
- pitch variance
- voiced ratio
- pause ratio
- pause count
- speech-rate proxy

Treat prosody as an auxiliary/ablation stream, not a replacement for Wav2Vec2 without evaluation.

## Privacy
`src/privacy.py` validates that raw-audio fields are absent from returned results.

The API/UI MUST complete:
`upload → temporary storage → inference → response → delete temporary file`

Never persist recordings or transcripts.

## O6 — UI
Build:
1. Upload or microphone recording.
2. Optional enrolled reference.
3. Analyze button.
4. Risk Level, Risk Score, Spoof Probability, Confidence.
5. Explanation and evidence.
6. Speaker similarity/consistency when reference exists.
7. Recommended action.
8. Privacy indicator.

## O7 — API
Use FastAPI.

Minimum:
- `GET /health`
- `POST /detect`
- `POST /detect-with-reference`

Provide request/response schemas, OpenAPI documentation, generated SDK stubs (Python + JS if feasible), and a mock webhook demonstrating external consumption.

Do NOT claim real integrations with banks, telecom carriers, Genesys, Five9, etc. Those are deferred.

## Evaluation — never invent
Verified:
- ASVspoof2019 LA eval: 95.46% accuracy, 4.74% EER.
- In-The-Wild cross-dataset: 37.86% accuracy, 37.52% EER.
- Prosody eval: 83.62% accuracy, ROC-AUC 0.8966.
- Speaker-pair calibration: threshold 0.2132, accuracy 81.40%.

The proposal describes IndieFake as the intended Indian-speaker cross-test, but the current repository result is In-The-Wild. Do not relabel it.

## Future work — do not claim as built
Full streaming endpointing/stability; telephony/VoIP codec robustness; privacy-preserving research representations; broad Indic-language coverage/fairness; neural-audio-codec attacks; multimodal audio+video; gRPC; production integrations; live banking/CRM context; configurable enterprise workflows; real SMS/email dispatch; formal compliance certification; true edge/on-device inference.

## Judge demo
Upload suspicious/genuine audio → API → overlapping-window analysis → risk tier → explanation → recommendation → optional speaker consistency → privacy confirmation → mock webhook.

## Acceptance criteria
- Existing detector still runs.
- UI submits audio and displays structured results.
- FastAPI endpoints work.
- OpenAPI is available.
- Mock webhook works.
- Temporary uploaded audio is deleted.
- No raw audio is returned/logged.
- No metrics are fabricated.
- README explains how to run the system.

Prefer small, reliable integration over refactoring the working ML pipeline.

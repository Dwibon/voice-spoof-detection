import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil

import sys
sys.path.append(os.path.dirname(__file__))

# Import the core logic
from realtime_detect import detect

app = FastAPI(
    title="Voice Spoof Detection API",
    description="Real-Time AI Voice Cloning / Impersonation Detection API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory webhook target for demo
webhook_payloads = []

class WebhookPayload(BaseModel):
    event: str
    risk_level: str
    risk_score: float
    recommendation: str

# ---------------------------------------------------------
# O7: Mock Webhook Endpoint
# ---------------------------------------------------------
@app.post("/mock-webhook")
async def mock_webhook(payload: WebhookPayload):
    """
    Mock external webhook that receives alerts for high-risk voice spoofing.
    """
    webhook_payloads.append(payload.model_dump())
    return {"status": "received", "payload": payload}

@app.get("/mock-webhook/logs")
async def get_webhook_logs():
    return {"logs": webhook_payloads}


# ---------------------------------------------------------
# O7: API Endpoints
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "voice-spoof-detector"}

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        if not suffix:
            suffix = ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
        return path
    finally:
        upload_file.file.close()

def remove_file(path: str):
    if path and os.path.exists(path):
        os.remove(path)

async def dispatch_webhook(result: dict):
    # Only dispatch if HIGH risk
    if result.get("risk_level") == "HIGH":
        # Simulate network delay for webhook
        await asyncio.sleep(0.5)
        # In a real app we'd make an HTTP request using httpx to an external URL.
        # Here we just route it to our mock endpoint directly for the demo.
        payload = WebhookPayload(
            event="voice_spoof_detected",
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            recommendation=result["recommendation"]["action"]
        )
        await mock_webhook(payload)

@app.post("/detect")
async def api_detect(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...)
):
    """
    Detect spoofing on a single audio file.
    O10 Privacy: Upload -> temp storage -> inference -> response -> DELETE
    """
    audio_path = save_upload_file_tmp(audio)
    
    try:
        # Run detection
        result = detect(audio_path=audio_path, reference_audio=None)
        
        # Dispatch webhook if necessary
        background_tasks.add_task(dispatch_webhook, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Guarantee privacy safe cleanup
        background_tasks.add_task(remove_file, audio_path)

@app.post("/detect-with-reference")
async def api_detect_with_reference(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    reference: UploadFile = File(...)
):
    """
    Detect spoofing with an optional enrolled speaker reference.
    """
    audio_path = save_upload_file_tmp(audio)
    ref_path = save_upload_file_tmp(reference)
    
    try:
        result = detect(audio_path=audio_path, reference_audio=ref_path)
        
        background_tasks.add_task(dispatch_webhook, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Privacy cleanup
        background_tasks.add_task(remove_file, audio_path)
        background_tasks.add_task(remove_file, ref_path)

# Serve the UI (if exists)
ui_path = os.path.join(os.path.dirname(__file__), '..', 'ui')
if not os.path.exists(ui_path):
    os.makedirs(ui_path, exist_ok=True)
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

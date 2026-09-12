import os
import tempfile
import asyncio

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import shutil
import numpy as np
import torch

from src.risk_engine import calculate_risk

import sys
sys.path.append(os.path.dirname(__file__))

# Import the core logic
from realtime_detect import (
    detect,
    load_model,
    extract_embedding,
    TARGET_SR,
    ROLLING_WINDOW,
)


app = FastAPI(
    title="Voice Spoof Detection API",
    description="Real-Time AI Voice Cloning / Impersonation Detection API",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# In-memory webhook target for demo
# ---------------------------------------------------------

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
    Mock external webhook that receives alerts for high-risk
    voice spoofing.
    """

    webhook_payloads.append(
        payload.model_dump()
    )

    return {
        "status": "received",
        "payload": payload,
    }


@app.get("/mock-webhook/logs")
async def get_webhook_logs():
    return {
        "logs": webhook_payloads
    }


# ---------------------------------------------------------
# O7: API Endpoints
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "voice-spoof-detector",
    }


def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(
            upload_file.filename
        )[1]

        if not suffix:
            suffix = ".wav"

        fd, path = tempfile.mkstemp(
            suffix=suffix
        )

        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(
                upload_file.file,
                f,
            )

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

        # In a real app we'd make an HTTP request
        # using httpx to an external URL.
        # Here we route it to our mock endpoint.

        payload = WebhookPayload(
            event="voice_spoof_detected",
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            recommendation=result["recommendation"]["action"],
        )

        await mock_webhook(payload)


# ---------------------------------------------------------
# File Detection
# ---------------------------------------------------------

@app.post("/detect")
async def api_detect(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
):
    """
    Detect spoofing on a single audio file.

    O10 Privacy:
    Upload -> temporary storage -> inference -> response -> DELETE
    """

    audio_path = save_upload_file_tmp(audio)

    try:

        result = detect(
            audio_path=audio_path,
            reference_audio=None,
        )

        background_tasks.add_task(
            dispatch_webhook,
            result,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        background_tasks.add_task(
            remove_file,
            audio_path,
        )


@app.post("/detect-with-reference")
async def api_detect_with_reference(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    """
    Detect spoofing with an optional enrolled speaker reference.
    """

    audio_path = save_upload_file_tmp(audio)
    ref_path = save_upload_file_tmp(reference)

    try:

        result = detect(
            audio_path=audio_path,
            reference_audio=ref_path,
        )

        background_tasks.add_task(
            dispatch_webhook,
            result,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        background_tasks.add_task(
            remove_file,
            audio_path,
        )

        background_tasks.add_task(
            remove_file,
            ref_path,
        )


# ---------------------------------------------------------
# Real-Time WebSocket Detection
# ---------------------------------------------------------

@app.websocket("/ws/live-detect")
async def live_detect(websocket: WebSocket):

    await websocket.accept()

    feature_extractor, backbone, classifier = load_model(None)

    device = next(
        classifier.parameters()
    ).device

    # -----------------------------------------------------
    # Detection buffer
    #
    # 4-second window
    # -----------------------------------------------------

    audio_buffer = bytearray()

    window_bytes = (
        TARGET_SR * 4 * 2
    )

    # 1-second step
    step_bytes = (
        TARGET_SR * 2
    )

    probabilities = []

    # -----------------------------------------------------
    # DEBUG RECORDING
    #
    # IMPORTANT:
    # This stores the ACTUAL incoming microphone stream.
    #
    # It does NOT store overlapping 4-second windows.
    # Therefore a 20-second recording produces ~20 seconds
    # of debug audio rather than ~80 seconds.
    # -----------------------------------------------------

    debug_audio_chunks = []

    try:

        while True:

            # Receive the actual microphone PCM chunk
            chunk = await websocket.receive_bytes()

            # -------------------------------------------------
            # Save the actual incoming stream for offline
            # reproduction.
            # -------------------------------------------------

            debug_audio_chunks.append(
                np.frombuffer(
                    chunk,
                    dtype=np.int16,
                ).copy()
            )

            # -------------------------------------------------
            # Add chunk to detection buffer
            # -------------------------------------------------

            audio_buffer.extend(chunk)

            # -------------------------------------------------
            # Process every available 4-second window
            # -------------------------------------------------

            while len(audio_buffer) >= window_bytes:

                window_bytes_data = bytes(
                    audio_buffer[:window_bytes]
                )

                pcm = np.frombuffer(
                    window_bytes_data,
                    dtype=np.int16,
                )

                waveform = torch.from_numpy(
                    pcm.astype(np.float32) / 32768.0
                )

                # -------------------------------------------------
                # Model inference
                # -------------------------------------------------

                with torch.no_grad():

                    embedding = extract_embedding(
                        waveform,
                        feature_extractor,
                        backbone,
                        device,
                    )

                    logit = classifier(
                        embedding
                    ).squeeze()

                    probability = torch.sigmoid(
                        logit
                    ).item()

                # -------------------------------------------------
                # Rolling probability
                # -------------------------------------------------

                probabilities.append(
                    probability
                )

                if len(probabilities) > ROLLING_WINDOW:
                    probabilities.pop(0)

                final_probability = float(
                    np.mean(probabilities)
                )

                # -------------------------------------------------
                # Risk calculation
                # -------------------------------------------------

                risk = calculate_risk(
                    final_probability
                )

                # -------------------------------------------------
                # Diagnostics
                # -------------------------------------------------

                rms = float(
                    torch.sqrt(
                        torch.mean(
                            waveform ** 2
                        )
                    )
                )

                peak = float(
                    torch.max(
                        torch.abs(waveform)
                    )
                )

                print(
                    f"[LIVE] "
                    f"window={probability:.4f} "
                    f"rolling={final_probability:.4f} "
                    f"risk={risk['risk_level']} "
                    f"rms={rms:.5f} "
                    f"peak={peak:.5f}"
                )

                # -------------------------------------------------
                # Send result to frontend
                # -------------------------------------------------

                await websocket.send_json({

                    "spoof_probability":
                        final_probability,

                    "spoof_percentage":
                        round(
                            final_probability * 100,
                            2,
                        ),

                    "risk_score":
                        risk["risk_score"],

                    "risk_level":
                        risk["risk_level"],

                    "confidence":
                        risk["confidence"],

                    "recommendation":
                        risk["recommendation"],

                    "windows_analyzed":
                        len(probabilities),
                })

                # -------------------------------------------------
                # Move forward by exactly 1 second
                # -------------------------------------------------

                del audio_buffer[
                    :step_bytes
                ]

    except WebSocketDisconnect:

        # -----------------------------------------------------
        # Save the ACTUAL microphone stream
        # -----------------------------------------------------

        if debug_audio_chunks:

            try:

                import soundfile as sf

                # Convert all incoming int16 chunks into one
                # continuous audio stream.

                debug_pcm = np.concatenate(
                    debug_audio_chunks
                )

                debug_audio = (
                    debug_pcm.astype(
                        np.float32
                    ) / 32768.0
                )

                os.makedirs(
                    "demo_audio",
                    exist_ok=True,
                )

                debug_path = (
                    "demo_audio/"
                    "live_debug_actual.wav"
                )

                sf.write(
                    debug_path,
                    debug_audio,
                    TARGET_SR,
                )

                print(
                    f"[LIVE DEBUG] saved "
                    f"{len(debug_audio) / TARGET_SR:.2f}s "
                    f"of actual microphone audio "
                    f"to {debug_path}"
                )

            except Exception as debug_exc:

                print(
                    f"[LIVE DEBUG] failed to save: "
                    f"{debug_exc}"
                )

    except Exception as exc:

        print(
            f"[LIVE ERROR] {exc}"
        )

        try:

            await websocket.send_json({
                "error": str(exc)
            })

        except Exception:
            pass


# ---------------------------------------------------------
# Serve the UI
# ---------------------------------------------------------

ui_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ui",
)

if not os.path.exists(ui_path):

    os.makedirs(
        ui_path,
        exist_ok=True,
    )

app.mount(
    "/",
    StaticFiles(
        directory=ui_path,
        html=True,
    ),
    name="ui",
)
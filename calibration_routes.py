#/Users/laraibnoorien/deception_detection_project/calibration_routes.py
import asyncio
import json
import uuid
import time
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

# Import your existing calibration engine and profile
from multimodal_calibration import (
    CalibrationEngine,
    CalibrationConfig,
    CalibrationProgress,
    run_calibration_for_subject,
)
from subject_behavior_profile import SubjectBehaviorProfile

router = APIRouter()
print("✅ Calibration router created")

class CalibrationStartRequest(BaseModel):
    subject_id: str

class CalibrationStartResponse(BaseModel):
    session_id: str
    stimulus_videos: List[str]
    questions: List[str]

# ---------------------------------------------------------------------------
# In‑memory session store (replace with Redis/DB for persistence)
# ---------------------------------------------------------------------------
active_sessions: Dict[str, asyncio.Task] = {}
session_progress: Dict[str, CalibrationProgress] = {}

# Default list of stimulus videos (adjust paths as needed)
DEFAULT_STIMULUS_DIR = Path("./stimuli/casme2_videos")
DEFAULT_QUESTIONS = [
    "Please state your full name and current occupation.",
    "Describe a recent event that made you happy.",
    "Tell me about a time you had to make a difficult decision.",
    "What are your plans for next year?",
    "Can you explain what you had for breakfast today?",
]

def get_stimulus_videos() -> list[str]:
    """List video files from the CASME2 directory."""
    if not DEFAULT_STIMULUS_DIR.exists():
        return []
    return sorted([
        f.name for f in DEFAULT_STIMULUS_DIR.glob("*.mp4")
    ])



# WebSocket endpoint: /ws/calibration/{session_id}
@router.websocket("/ws/calibration/{session_id}")
async def calibration_websocket(websocket: WebSocket, session_id: str):
    """WebSocket bridge for a calibration session.

    Expects binary image frames from the frontend and will forward them to the
    FRAME_QUEUES[session_id] queue created by backend_app.start_calibration.
    Sends progress updates and stage_complete messages to the connected client
    by reading CALIBRATIONS[session_id] state.
    """
    await websocket.accept()

    subject_id = websocket.query_params.get("subject_id", "UNKNOWN")

    # Access shared stores from backend_app module directly to avoid import-order issues
    try:
        import backend_app as backend
        frame_queues = getattr(backend, 'FRAME_QUEUES', None)
        calibrations = getattr(backend, 'CALIBRATIONS', None)
    except Exception:
        frame_queues = globals().get('FRAME_QUEUES')
        calibrations = globals().get('CALIBRATIONS')

    if frame_queues is None or calibrations is None:
        await websocket.send_json({"type": "error", "payload": {"message": "Server calibration store unavailable"}})
        await websocket.close()
        return

    q = frame_queues.get(session_id)
    if q is None:
        await websocket.send_json({"type": "error", "payload": {"message": "Calibration session not found or not started"}})
        await websocket.close()
        return

    last_progress = None

    async def sender_loop():
        nonlocal last_progress
        try:
            while True:
                cal = calibrations.get(session_id)
                if cal is None:
                    await asyncio.sleep(0.2)
                    continue
                prog = cal.get('progress')
                status = cal.get('status')
                # Send progress changes
                if prog and prog != last_progress:
                    try:
                        await websocket.send_json({"type": "progress", "payload": prog})
                    except Exception:
                        # If send fails (client disconnected), stop sender loop
                        break
                    last_progress = prog
                # When calibration completes, send stage_complete with baseline
                if status == 'completed' and cal.get('result'):
                    try:
                        await websocket.send_json({"type": "stage_complete", "payload": {"stage": "COMPLETED", "baseline": cal.get('result')}})
                    except Exception:
                        pass
                    break
                if status == 'failed':
                    try:
                        await websocket.send_json({"type": "error", "payload": {"message": cal.get('error', 'Calibration failed')}})
                    except Exception:
                        pass
                    break
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return

    sender_task = asyncio.create_task(sender_loop())

    try:
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                # Client disconnected; stop receiving
                break
            except Exception:
                # Other receive errors — stop receiving to avoid busy-loop
                break

            if 'bytes' in data and data['bytes']:
                # forward raw bytes into the server-side frame queue for the engine to consume
                try:
                    # Only put into queue if it still exists
                    if q is not None:
                        await q.put(data['bytes'])
                except Exception:
                    pass
            elif 'text' in data and data['text']:
                try:
                    msg = json.loads(data['text'])
                    # optional: handle control messages like 'ready' from client
                except Exception:
                    pass
    finally:
        # Stop sender task if still running
        if not sender_task.done():
            sender_task.cancel()
        # Cleanup frame queue and mark calibration as client_disconnected if present
        try:
            frame_queues.pop(session_id, None)
        except Exception:
            pass
        try:
            if session_id in calibrations:
                calibrations[session_id]['status'] = calibrations[session_id].get('status', 'unknown')
                # do not delete calibration result; only mark disconnection
                calibrations[session_id]['client_disconnected_at'] = time.time()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

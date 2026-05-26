"""Simple FastAPI backend for Microsense frontend integration.
Provides REST endpoints and WebSocket support for real-time multimodal telemetry.
"""

import os

# OpenMP duplicate-runtime handling controlled via env var ENABLE_KMP_DUPLICATE_LIB_OK (dev only)
if os.getenv("ENABLE_KMP_DUPLICATE_LIB_OK", "false").lower() == "true":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from multimodal_calibration import run_calibration_for_subject

# PM hardware configuration
try:
    from O2_PhysiologicalModality.hardware_config import check_pm_hardware, get_pm_hardware_config
    print("✅ PM hardware config imported")
except Exception as e:
    print(f"⚠️ PM hardware config import failed: {e}")
    def check_pm_hardware():
        return {"sensors_enabled": {}, "hardware_ready": False}
    def get_pm_hardware_config():
        class _Dummy:
            config = {}
            def save_config(self):
                return False
        return _Dummy()

# Audio preprocessing
try:
    from O3_SpeechModality.audio_preprocessing import get_audio_preprocessor
    print("✅ Audio preprocessing imported")
except Exception as e:
    print(f"⚠️ Audio preprocessing import failed: {e}")
    def get_audio_preprocessor(*args, **kwargs):
        return None


# FASTAPI APP INITIALIZATION

app = FastAPI(title="Microsense Backend Bridge")


# CALIBRATION ROUTES IMPORT

try:
    from calibration_routes import router as calibration_router, calibration_websocket
    print("✅ Calibration router created")
except Exception as e:
    print(f"⚠️ Calibration router import failed: {e}")
    calibration_router = None
    calibration_websocket = None

# Mount calibration router
if calibration_router is not None:
    try:
        app.include_router(calibration_router, prefix="/api")
        print("✅ Calibration router mounted")
    except Exception as e:
        print(f"⚠️ Failed to mount calibration router: {e}")

# Calibration websocket endpoint (delegates to calibration_routes handler)
if calibration_websocket is not None:
    @app.websocket("/ws/calibration/{session_id}")
    async def ws_calibration_endpoint(websocket: WebSocket, session_id: str):
        await calibration_websocket(websocket, session_id)


# FRONTEND STATIC FILES

_frontend_dist = Path(__file__).parent / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_frontend_dist), html=True),
        name="frontend",
    )

    @app.get("/")
    async def frontend_index():
        index_path = _frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {
            "success": True,
            "message": "Microsense backend bridge running",
        }


# CORS - custom middleware to safely allow local frontend origins

_allowed = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:5173",
)
origins = [o.strip() for o in _allowed.split(",") if o.strip()]

class LocalhostCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        origin = request.headers.get('origin', '')
        
        def origin_allowed(o: str) -> bool:
            if not o:
                return False
            if o in origins:
                return True
            # allow any localhost:PORT origins
            if o.startswith('http://localhost') or o.startswith('https://localhost'):
                return True
            if o.startswith('http://127.0.0.1') or o.startswith('https://127.0.0.1'):
                return True
            return False

        # Handle preflight
        if request.method == 'OPTIONS':
            cors_headers = {}
            if origin_allowed(origin):
                cors_headers['Access-Control-Allow-Origin'] = origin
                cors_headers['Access-Control-Allow-Credentials'] = 'true'
                cors_headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
                cors_headers['Access-Control-Allow-Headers'] = 'Authorization,Content-Type,Accept'
                cors_headers['Vary'] = 'Origin'
                cors_headers['Access-Control-Max-Age'] = '3600'
            # 204 must not contain a body — return an empty Response to avoid corrupting downstream middleware
            return Response(status_code=204, headers=cors_headers)

        # Process actual request
        try:
            response = await call_next(request)
            
            # Add CORS headers to response
            if origin_allowed(origin):
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Credentials'] = 'true'
                response.headers['Vary'] = 'Origin'
            
            return response
        except Exception as e:
            # Ensure error responses also get CORS headers
            print(f"Middleware error: {e}")
            error_response = JSONResponse(
                status_code=500, 
                content={"error": "Internal server error"}
            )
            if origin_allowed(origin):
                error_response.headers['Access-Control-Allow-Origin'] = origin
                error_response.headers['Access-Control-Allow-Credentials'] = 'true'
            return error_response

# Replace generic CORSMiddleware with our custom one
app.add_middleware(LocalhostCORSMiddleware)


# OPTIONAL API KEY SECURITY

API_KEY = os.getenv("MICROSENSE_API_KEY")
if API_KEY:

    class APIKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if (
                request.url.path.startswith("/docs")
                or request.url.path.startswith("/openapi.json")
                or request.url.path.startswith("/assets")
            ):
                return await call_next(request)

            key = (
                request.headers.get("x-api-key")
                or request.query_params.get("api_key")
            )
            if not key or key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Unauthorized",
                    },
                )
            return await call_next(request)

    app.add_middleware(APIKeyMiddleware)


# MEMORY STORES

SESSIONS = {}
METRICS_HISTORY = []
FRAME_QUEUES = {}
CALIBRATIONS = {}
SESSION_LOCKS = {}  # Track active sessions to prevent duplicates
SESSION_LOCK_TIMEOUT = 300  # 5 minutes timeout
# Additionally track subject-level locks to prevent race conditions on start
SUBJECT_LOCKS = {}  # subject_id -> timestamp

# If calibration_routes was imported earlier, inject shared stores so it can
# forward frames and read progress without circular imports.
try:
    import calibration_routes as _cal_routes
    _cal_routes.FRAME_QUEUES = FRAME_QUEUES
    _cal_routes.CALIBRATIONS = CALIBRATIONS
except Exception:
    pass


# HELPERS

def now_ts():
    return int(time.time() * 1000)


# WEBSOCKET CONNECTION MANAGER

class ConnectionManager:
    def __init__(self):
        self.active = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        living = []
        for connection in list(self.active):
            try:
                await connection.send_json(message)
                living.append(connection)
            except Exception:
                try:
                    await connection.close()
                except Exception:
                    pass
        self.active = living

manager = ConnectionManager()


# SESSION ROUTES

@app.get("/api/sessions")
async def list_sessions():
    return {
        "success": True,
        "data": list(SESSIONS.values()),
        "timestamp": now_ts(),
    }

@app.post("/api/sessions/start")
async def start_session(payload: dict):
    subject_id = (
        payload.get("subjectId")
        or payload.get("subject_id")
        or "unknown"
    )
    session_id = str(uuid.uuid4())
    sess = {
        "id": session_id,
        "subjectId": subject_id,
        "startedAt": now_ts(),
        "status": "running",
    }
    SESSIONS[session_id] = sess
    return {
        "success": True,
        "data": sess,
        "timestamp": now_ts(),
    }

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Session not found",
            },
        )
    sess["endedAt"] = now_ts()
    sess["status"] = "ended"
    return {
        "success": True,
        "data": sess,
        "timestamp": now_ts(),
    }


# METRICS ROUTES

@app.get("/api/system/status")
async def system_status():
    return {
        "success": True,
        "data": {
            "status": "ok",
            "uptime_ms": 0,
        },
        "timestamp": now_ts(),
    }

@app.get("/api/metrics/current")
async def get_current_metrics():
    if METRICS_HISTORY:
        return {
            "success": True,
            "data": METRICS_HISTORY[-1],
            "timestamp": now_ts(),
        }
    return JSONResponse(
        status_code=204,
        content={
            "success": False,
            "error": "No metrics available",
        },
    )

@app.post("/api/metrics/push")
async def push_metrics(payload: dict):
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid payload",
            },
        )

    METRICS_HISTORY.append(payload)
    if len(METRICS_HISTORY) > 1000:
        METRICS_HISTORY.pop(0)

    try:
        await manager.broadcast({
            "type": "metrics",
            "payload": payload,
            "timestamp": now_ts(),
        })
    except Exception:
        pass

    return {
        "success": True,
        "data": payload,
        "timestamp": now_ts(),
    }


# PM HARDWARE CONFIG API

@app.get('/api/pm/config')
async def get_pm_config():
    try:
        cfg = get_pm_hardware_config().config
    except Exception:
        cfg = {}
    return {
        "success": True,
        "data": cfg,
        "timestamp": now_ts(),
    }

@app.post('/api/pm/config')
async def set_pm_config(payload: dict):
    try:
        cfgobj = get_pm_hardware_config()
        # Merge provided settings shallowly into existing config
        if isinstance(payload, dict):
            cfgobj.config.update(payload)
            saved = cfgobj.save_config()
        else:
            saved = False
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    return {
        "success": bool(saved),
        "data": cfgobj.config,
        "timestamp": now_ts(),
    }


# MAIN WEBSOCKET

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                txt = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(0.05)
                continue

            try:
                msg = json.loads(txt)
            except Exception:
                continue

            mtype = msg.get("type")
            if mtype == "ping":
                try:
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": now_ts(),
                    })
                except Exception:
                    pass
                continue

            if mtype == "frame":
                cal_id = msg.get("calibrationId")
                payload = msg.get("payload")
                if not cal_id or payload is None:
                    continue
                q = FRAME_QUEUES.get(cal_id)
                if q is None:
                    continue
                try:
                    await q.put(payload)
                except Exception:
                    pass
    finally:
        manager.disconnect(websocket)


# CALIBRATION ROUTES

def get_stimulus_videos() -> list:
    """List video files from the stimuli directory, or return empty list if not found."""
    base_dir = Path(__file__).parent
    stimuli_dir = base_dir / "stimuli" / "casme2_videos"
    if stimuli_dir.exists():
        return sorted([f.name for f in stimuli_dir.glob("*.mp4")])
    return []

def get_calibration_questions() -> list:
    """Return default calibration questions."""
    return [
        "Please state your full name and current occupation.",
        "Describe a recent event that made you happy.",
        "Tell me about a time you had to make a difficult decision.",
        "What are your plans for next year?",
        "Can you explain what you had for breakfast today?",
    ]

@app.post("/api/calibration/start")
async def start_calibration(payload: dict):
    subject_id = (
        payload.get("subjectId")
        or payload.get("subject_id")
        or "unknown"
    )
    
    # Debug: Log current state
    print(f"\n🔍 DEBUG: start_calibration called for subject_id={subject_id}")
    print(f"   Current CALIBRATIONS keys: {list(CALIBRATIONS.keys())}")
    print(f"   Current CALIBRATIONS: {[(cal['subjectId'], cal['status']) for cal in CALIBRATIONS.values()]}")

    # Subject-level lock to prevent race conditions from concurrent starts
    now = now_ts()
    lock_ts = SUBJECT_LOCKS.get(subject_id)
    if lock_ts and (now - lock_ts) < SESSION_LOCK_TIMEOUT:
        print(f"   ❌ Returning 409 Conflict due to subject lock for {subject_id}")
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Subject already has active calibration session (subject-level lock)",
                "subject_id": subject_id,
            },
        )

    # Check both active status AND recent SESSION_LOCKS
    existing_active = [
        cal for cal in CALIBRATIONS.values()
        if cal["subjectId"] == subject_id and cal["status"] in ["queued", "running"]
    ]
    
    # Also check for recent locks (within SESSION_LOCK_TIMEOUT)
    recent_locks = [
        (cal_id, lock_ts) for cal_id, lock_ts in SESSION_LOCKS.items()
        if (now - lock_ts) < SESSION_LOCK_TIMEOUT
    ]
    
    # Find if any recent lock belongs to this subject
    for lock_cal_id, lock_ts in recent_locks:
        if lock_cal_id in CALIBRATIONS and CALIBRATIONS[lock_cal_id]["subjectId"] == subject_id:
            existing_active.append(CALIBRATIONS[lock_cal_id])
    
    print(f"   Found existing_active: {len(existing_active)} sessions")
    
    if existing_active:
        print(f"   ❌ Returning 409 Conflict")
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Subject already has active calibration session",
                "subject_id": subject_id,
                "existing_session_id": existing_active[0]["id"]
            }
        )

    base_dir = Path(__file__).parent
    required_files = [
        "O4_MicroexpressionModality/macro_v7.py",
        "O2_PhysiologicalModality/physio_v3.py",
        "O3_SpeechModality/speech_v3.py",
        "O1_TextModality/text_v4.py",
    ]
    missing = [
        f for f in required_files
        if not (base_dir / f).exists()
    ]
    if missing:
        # Missing capture modules are a hard error in deployment — raise loudly
        raise HTTPException(status_code=500, detail={
            "error": "Capture modules missing",
            "missing": missing,
            "timestamp": now_ts(),
        })
    
    # Check PM hardware status
    pm_status = check_pm_hardware()
    if not pm_status.get("hardware_ready", False):
        # Return warning but proceed (hardware may be optional in dev/testing)
        print(f"⚠️ PM hardware not fully ready: {pm_status}")

    cal_id = str(uuid.uuid4())
    # Lock both by cal_id and by subject to avoid races
    SESSION_LOCKS[cal_id] = now_ts()  # Lock this session
    SUBJECT_LOCKS[subject_id] = now_ts()
    
    CALIBRATIONS[cal_id] = {
        "id": cal_id,
        "sessionId": cal_id,
        "subjectId": subject_id,
        "status": "queued",
        "startedAt": now_ts(),
        "progress": None,
        "result": None,
        "stimulus_videos": get_stimulus_videos(),
        "questions": get_calibration_questions(),
        "hardware_status": pm_status,
    }

    FRAME_QUEUES[cal_id] = asyncio.Queue(maxsize=120)

    async def progress_callback(progress):
        CALIBRATIONS[cal_id]["progress"] = progress.to_dict()
        CALIBRATIONS[cal_id]["status"] = progress.stage
        try:
            await manager.broadcast({
                "type": "calibration_progress",
                "calibrationId": cal_id,
                "payload": progress.to_dict(),
                "timestamp": now_ts(),
            })
        except Exception:
            pass

    async def run_cal():
        try:
            print(f"   📊 [run_cal] Starting calibration for cal_id={cal_id}, subject_id={subject_id}")
            CALIBRATIONS[cal_id]["status"] = "running"
            print(f"   📊 [run_cal] Status changed to 'running'")
            profile = await run_calibration_for_subject(
                subject_id,
                progress_callback=progress_callback,
                external_frame_queue=FRAME_QUEUES.get(cal_id),
            )
            CALIBRATIONS[cal_id]["status"] = "completed"
            print(f"   📊 [run_cal] Status changed to 'completed'")
            try:
                CALIBRATIONS[cal_id]["result"] = profile.to_dict()
            except Exception:
                CALIBRATIONS[cal_id]["result"] = profile
            CALIBRATIONS[cal_id]["completedAt"] = now_ts()
        except Exception as e:
            print(f"   ❌ [run_cal] Calibration failed: {e}")
            CALIBRATIONS[cal_id]["status"] = "failed"
            CALIBRATIONS[cal_id]["error"] = str(e)
        finally:
            FRAME_QUEUES.pop(cal_id, None)
            # Release locks on completion/failure
            SESSION_LOCKS.pop(cal_id, None)
            SUBJECT_LOCKS.pop(subject_id, None)

    asyncio.create_task(run_cal())
    return {
        "success": True,
        "data": CALIBRATIONS[cal_id],
        "timestamp": now_ts(),
    }

@app.post("/api/demo/start")
async def start_demo(payload: dict = {}):
    """Start a demo calibration session with synthetic, synchronized modality data.
    This endpoint is demo-only and preserves the real websocket flow (broadcasts "progress" and "stage_complete").
    """
    subject_id = payload.get("subject_id") or payload.get("subjectId") or "demo-subject"
    cal_id = str(uuid.uuid4())
    now = now_ts()
    SESSION_LOCKS[cal_id] = now
    SUBJECT_LOCKS[subject_id] = now
    FRAME_QUEUES[cal_id] = asyncio.Queue(maxsize=120)
    CALIBRATIONS[cal_id] = {
        "id": cal_id,
        "sessionId": cal_id,
        "subjectId": subject_id,
        "status": "queued",
        "startedAt": now,
        "progress": None,
        "result": None,
        "stimulus_videos": get_stimulus_videos(),
        "questions": get_calibration_questions(),
        "hardware_status": check_pm_hardware(),
    }

    async def demo_runner():
        try:
            CALIBRATIONS[cal_id]["status"] = "running"
            # Simulate a sequence of emotional stimuli (MM->PM), then speech (SM)
            stages = [
                ("mm_pm", 8, "Video"),
                ("pm", 1, "Physio"),
                ("sm", 5, "Question"),
            ]
            total = sum(max(1,c) for _,c,_ in stages)
            step = 0
            for stage, count, label in stages:
                for i in range(max(1,count)):
                    step += 1
                    prog = {
                        "stage": stage,
                        "overall_progress": step / float(total),
                        "stage_progress": (i+1)/float(max(1,count)),
                        "current_item": f"{label} {i+1}/{max(1,count)}: demo",
                        "modality_status": {"mm":"capturing","pm":"capturing","sm":"pending","tm":"pending"},
                        "modality_confidence": {"mm":0.8 - 0.05*(i%3), "pm":0.75 + 0.01*(i%2), "sm":0.6, "tm":0.5},
                        "estimated_time_remaining": float(total - step) * 1.0,
                        "errors": [],
                    }
                    CALIBRATIONS[cal_id]["progress"] = prog
                    try:
                        await manager.broadcast({"type":"progress","calibrationId":cal_id,"payload":prog,"timestamp": now_ts()})
                    except Exception:
                        pass
                    # Also push a small synthetic frame payload into FRAME_QUEUES so consumers see input
                    try:
                        q = FRAME_QUEUES.get(cal_id)
                        if q is not None and not q.full():
                            await q.put(b"demo_frame")
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
            # Simulate speech questions processing
            # For SM stage, push several progress updates and then completion
            for qidx in range(1,6):
                step += 1
                prog = {
                    "stage": "sm",
                    "overall_progress": min(0.95, float(step)/float(total+5)),
                    "stage_progress": qidx/5.0,
                    "current_item": f"Question {qidx}/5: demo",
                    "modality_status": {"mm":"capturing","pm":"capturing","sm":"capturing","tm":"pending"},
                    "modality_confidence": {"mm":0.75, "pm":0.72, "sm":0.82, "tm":0.55},
                    "estimated_time_remaining": float(6-qidx),
                    "errors": [],
                }
                CALIBRATIONS[cal_id]["progress"] = prog
                try:
                    await manager.broadcast({"type":"progress","calibrationId":cal_id,"payload":prog,"timestamp": now_ts()})
                except Exception:
                    pass
                await asyncio.sleep(1.2)

            # Finalize
            CALIBRATIONS[cal_id]["status"] = "completed"
            result = {"deception_probability": 0.78, "confidence": 0.89, "modalities": {"mm":0.28,"pm":0.22,"sm":0.38,"tm":0.12}}
            CALIBRATIONS[cal_id]["result"] = result
            CALIBRATIONS[cal_id]["completedAt"] = now_ts()
            try:
                await manager.broadcast({"type":"stage_complete","calibrationId":cal_id,"payload":{"stage":"COMPLETED","baseline":result},"timestamp": now_ts()})
            except Exception:
                pass
        finally:
            FRAME_QUEUES.pop(cal_id, None)
            SESSION_LOCKS.pop(cal_id, None)
            SUBJECT_LOCKS.pop(subject_id, None)

    asyncio.create_task(demo_runner())
    return {
        "success": True,
        "data": CALIBRATIONS[cal_id],
        "timestamp": now_ts(),
    }

@app.get("/api/calibration/{cal_id}/status")
async def calibration_status(cal_id: str):
    c = CALIBRATIONS.get(cal_id)
    if not c:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Calibration not found",
            },
        )
    return {
        "success": True,
        "data": c,
        "timestamp": now_ts(),
    }


# ROOT HEALTHCHECK

@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "healthy",
        "timestamp": now_ts(),
    }


# Shutdown endpoint (best-effort): allows main.py to request backend shutdown
@app.post("/shutdown")
async def shutdown(request: Request):
    try:
        loop = asyncio.get_event_loop()
        # Schedule loop.stop on the loop thread
        def _stop_loop():
            try:
                loop.stop()
            except Exception:
                pass
        loop.call_soon_threadsafe(_stop_loop)
        return {"success": True, "message": "Shutdown scheduled"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
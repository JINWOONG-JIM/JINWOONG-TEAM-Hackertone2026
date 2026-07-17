from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .accent_estimator import estimate_accent
from .audio_features import extract_features
from .tts_detector import detect_synthetic

app = FastAPI(title="VoiceOrigin AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 15 * 1024 * 1024
MAX_DURATION_SEC = 30.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large (max 15MB)")

    try:
        features = extract_features(data)
    except Exception as exc:
        raise HTTPException(400, f"could not process audio: {exc}") from exc

    if features.duration_sec > MAX_DURATION_SEC:
        raise HTTPException(400, f"clip too long (max {int(MAX_DURATION_SEC)}s)")

    step1 = detect_synthetic(features)
    result = {"step1": step1, "step2": None}

    if step1["label"] == "synthetic":
        result["verdict"] = "Synthetic (TTS) voice suspected — accent estimation skipped since it isn't a human voice"
        return result

    if step1["label"] == "unknown":
        result["verdict"] = "Unable to determine — insufficient voiced speech"
        return result

    step2 = estimate_accent(features)
    result["step2"] = step2

    if step2["status"] == "insufficient_data":
        result["verdict"] = "Human voice / accent estimation withheld due to insufficient speech length or voiced content"
    else:
        top = step2["distribution"][0]
        result["verdict"] = f"Human voice / likely {top['label']} ({top['probability']}%)"

    return result

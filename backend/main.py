from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from leaf_engine import GROQ_MODEL, analyze_leaf
from livestock_engine import cow_status, generate_frames

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Haran AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/diagnose")
def diagnose(
    file: UploadFile = File(...),
    lat: float = Form(36.6742),
    lng: float = Form(10.2854),
):
    image_bytes = file.file.read()
    return analyze_leaf(image_bytes, lat, lng)


@app.get("/api/cow-stream")
def cow_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cow-status")
def cow_watch_status():
    return cow_status()


@app.get("/api/health")
def health():
    return {"status": "online", "model": GROQ_MODEL}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

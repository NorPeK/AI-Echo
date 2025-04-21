import os, base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Ai_Echo_model import (
    TMP_AUDIO,
    prepare_local_audio, download_youtube_audio,
    isolate_voice, transcribe_arabic,
    polish_commentary, translate_to_english,
    tts_for_line, TTS_OUT
)
from dotenv import load_dotenv

load_dotenv()

if not all(os.getenv(k) for k in ("LALAL_API_KEY","OPENAI_API_KEY","ELEVENLABS_API_KEY")):
    raise RuntimeError("Set LALAL_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY in .env")

app = FastAPI(title="AI‑Echo TTS Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    youtube_url: str | None = None

class ProcessResponse(BaseModel):
    arabic_polished: str
    english: str
    tts_audio_base64: str

@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if not req.youtube_url:
        raise HTTPException(400, "youtube_url is required")
    try:
        raw = download_youtube_audio(req.youtube_url, TMP_AUDIO)
        clean = isolate_voice(raw)
        arabic = transcribe_arabic(clean)
        polished = polish_commentary(arabic)
        english = translate_to_english(polished)
        tts_for_line(english, TTS_OUT)

        with open(TTS_OUT, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return ProcessResponse(
            arabic_polished=polished,
            english=english,
            tts_audio_base64=b64
        )
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)

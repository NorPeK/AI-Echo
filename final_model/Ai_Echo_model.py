"""
football_tts_pipeline.py
────────────────────────
End‑to‑end Arabic‑to‑English football‑commentary → ElevenLabs speech.

Install:
    pip install yt_dlp requests whisper openai ffmpeg-python
    pip install elevenlabs python-dotenv sentence-transformers scikit-learn joblib

Env:
    LALAL_API_KEY      – for lalal.ai
    OPENAI_API_KEY     – for Whisper / GPT
    ELEVENLABS_API_KEY – for TTS
"""

import os, json, time, subprocess, shutil, requests, yt_dlp, whisper, joblib, numpy as np
from openai import OpenAI
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ─────────────────────────────  CONFIG  ──────────────────────────────
# Input clip (choose ONE)
YOUTUBE_URL    = "https://www.youtube.com/shorts/6f4fftURCbQ"   # "" if using LOCAL_FILE
LOCAL_FILE     = None                                           # Path("clip.mp4") / "voice.mp3"

# Pipeline settings
TMP_AUDIO      = Path("downloaded_audio.mp3")
VOICE_BASENAME = "commentator_voice"
CONVERT_TO_WAV = True
WHISPER_SIZE   = "large"                                        # "medium" | "large"
OPENAI_MODEL   = "gpt-4o-mini-2024-07-18"
MAX_DURATION   = 60                                             # seconds
POLL_DELAY     = 5                                              # seconds

# ElevenLabs / ML
VOICE_ID       = "3mAgWMVqMBHLhX3ZqwNJ"                         # your custom voice
MODEL_FILE     = Path("param_model.joblib")
TTS_OUT        = "commentary_tts.mp3"

# ---------------------------------------------------------------------#
# 0. Keys & clients
# ---------------------------------------------------------------------#
load_dotenv()
LICENSE  = os.getenv("LALAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

if not (LICENSE and OPENAI_API_KEY and ELEVEN_KEY):
    raise RuntimeError("Set LALAL_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client = ElevenLabs(api_key=ELEVEN_KEY)

HEADERS = {"Authorization": f"license {LICENSE}"}
API     = "https://www.lalal.ai/api"

# ───────────────────────  AUDIO HELPERS  ────────────────────────────
def ensure_max_duration(f: Path, limit: int = MAX_DURATION) -> Path:
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(f)], text=True))
    if dur <= limit + 0.01:
        return f
    trimmed = f.with_stem(f.stem + "_trim")
    subprocess.run(["ffmpeg", "-y", "-t", str(limit), "-i", str(f),
                    "-acodec", "copy", str(trimmed)],
                   check=True, capture_output=True)
    return trimmed

def prepare_local_audio(src: Path | str, out_path: Path) -> Path:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() in {".mp3", ".wav"}:
        shutil.copy(src, out_path)
    else:  # video → extract audio
        subprocess.run(["ffmpeg", "-y", "-i", str(src),
                        "-vn", "-acodec", "mp3", str(out_path)],
                       check=True, capture_output=True)
    return ensure_max_duration(out_path)

def download_youtube_audio(url: str, out_path: Path) -> Path:
    if out_path.exists():
        out_path.unlink()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_path.with_suffix("")),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192"}],
        "quiet": True,
        "overwrites": True,
    }
    print("🔻  Downloading audio …")
    yt_dlp.YoutubeDL(opts).download([url])
    return ensure_max_duration(out_path)

# ───────────────────────  LALAL.AI HELPERS  ─────────────────────────
def api_upload(file_path: Path) -> str:
    with file_path.open("rb") as f:
        hdrs = HEADERS | {"Content-Disposition": f'attachment; filename="{file_path.name}"'}
        r = requests.post(f"{API}/upload/", headers=hdrs, data=f, timeout=300)
    r.raise_for_status()
    if r.json()["status"] != "success":
        raise RuntimeError(f"Upload error: {r.json()}")
    return r.json()["id"]

def api_split(file_id: str, noise_lvl: int = 1) -> None:
    params = json.dumps([{
        "id": file_id,
        "stem": "voice",
        "noise_cancelling_level": noise_lvl,
        "neural_network": "perseus"
    }])
    r = requests.post(f"{API}/split/", headers=HEADERS,
                      data={"params": params}, timeout=30)
    r.raise_for_status()
    if r.json()["status"] != "success":
        raise RuntimeError(f"/split/ error: {r.json()}")
    print("🪄  File queued for splitting …")

def api_poll(file_id: str) -> dict:
    while True:
        r = requests.post(f"{API}/check/", headers=HEADERS,
                          data={"id": file_id}, timeout=30)
        r.raise_for_status()
        res   = r.json()["result"][file_id]
        task  = res.get("task", {})
        state = task.get("state", "success")
        if state == "success" and res.get("split"):
            print("✅  Split finished.")
            return res["split"]
        if state == "error":
            raise RuntimeError(f"LALAL error: {task.get('error')}")
        print(f"   … {task.get('progress',0):3}% complete", end="\r")
        time.sleep(POLL_DELAY)

def isolate_voice(in_file: Path, noise_level: int = 1) -> Path:
    file_id    = api_upload(in_file)
    api_split(file_id, noise_level)
    split_info = api_poll(file_id)

    voice_url  = split_info["stem_track"]
    ext        = Path(urlparse(voice_url).path).suffix or ".wav"
    voice_file = Path(f"{VOICE_BASENAME}{ext}")

    print(f"⬇️   Downloading clean voice ({ext.lstrip('.')}) …")
    voice_file.write_bytes(requests.get(voice_url, timeout=300).content)

    if CONVERT_TO_WAV and voice_file.suffix.lower() != ".wav":
        wav_path = voice_file.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(voice_file),
                        "-ac", "1", "-ar", "16000", str(wav_path)],
                       capture_output=True, check=True)
        voice_file = wav_path
    return voice_file

# ─────────────────────────  WHISPER & GPT  ──────────────────────────
def transcribe_arabic(audio_path: Path) -> str:
    print(f"🧠  Loading Whisper ({WHISPER_SIZE}) …")
    model = whisper.load_model(WHISPER_SIZE)
    print("📝  Transcribing …")
    return model.transcribe(str(audio_path), language="ar")["text"]

def polish_commentary(raw_text: str) -> str:
    prompt = (
        "You are a football‑commentary editor. "
        "Players name might be wrong so correct them. "
        "Correct player names and spelling in this Arabic text ONLY. "
        "Do not add or remove words:\n\n" + raw_text
    )
    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Arabic sports‑language specialist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def translate_to_english(arabic_text: str) -> str:
    prompt = (
        "You are a football-commentary translator. "
        "Translate the following Arabic text to English. "
        "Be accurate and use English football slogans if it is used in Arabic.\n\n" + arabic_text
    )
    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Expert Arabic-English translator specialized in football commentary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ─────────────────────  MINI ML MODEL FOR TTS SLIDERS  ──────────────
train_samples = [
    ("GOAL! A stunning strike from the edge of the box!",  [0.15, 0.90, 1.10]),
    ("Yellow card shown for dissent; the referee had no choice.", [0.60, 0.85, 0.95]),
    ("The teams are lining up for the national anthem.",   [0.40, 0.80, 0.90]),
    ("Penalty saved! The keeper guessed the right way!",    [0.20, 0.88, 1.08]),
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
if MODEL_FILE.exists():
    model = joblib.load(MODEL_FILE)
else:
    X = embedder.encode([t for t, _ in train_samples])
    y = np.array([vals for _, vals in train_samples])
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Trained & saved new slider‑prediction model →", MODEL_FILE)

def tts_for_line(text: str, out_path: str = TTS_OUT) -> None:
    sliders = model.predict(embedder.encode([text]))[0]
    stability, similarity, speed = sliders
    settings = VoiceSettings(
        stability=float(np.clip(stability, 0.0, 1.0)),
        similarity_boost=float(np.clip(similarity, 0.0, 1.0)),
        speed=float(np.clip(speed, 0.7, 1.2)),
    )
    audio_iter = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=settings,
    )
    with open(out_path, "wb") as f:
        for chunk in audio_iter:
            if isinstance(chunk, bytes):
                f.write(chunk)
    print(
        f"🔊  Saved {out_path} | stability={settings.stability:.2f} "
        f"similarity={settings.similarity_boost:.2f} speed={settings.speed:.2f}"
    )

# ────────────────────────────  MAIN  ────────────────────────────────
def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found in PATH.")

    # 1‑2. Obtain & clean audio
    if LOCAL_FILE:
        raw_audio = prepare_local_audio(LOCAL_FILE, TMP_AUDIO)
    elif YOUTUBE_URL:
        raw_audio = download_youtube_audio(YOUTUBE_URL, TMP_AUDIO)
    else:
        raise SystemExit("Provide LOCAL_FILE or YOUTUBE_URL")

    clean_voice = isolate_voice(raw_audio)
    print(f"🎧  Voice track saved → {clean_voice}")

    # 3‑4‑5. ASR, polish, translate
    transcript       = transcribe_arabic(clean_voice)
    polished_comment = polish_commentary(transcript)
    english_comment  = translate_to_english(polished_comment)

    print("\n📄  Arabic (polished)\n" + "-"*40 + "\n" + polished_comment)
    print("\n📄  English\n" + "-"*40 + "\n" + english_comment)

    # 6. TTS
    tts_for_line(english_comment, TTS_OUT)

if __name__ == "__main__":
    main()

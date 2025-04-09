# tts_pipeline.py – End‑to‑end ElevenLabs demo with custom voice + ML slider prediction
"""
Install deps first:
    pip install elevenlabs python-dotenv sentence-transformers scikit-learn joblib

What this script does
---------------------
1. Load ELEVENLABS_API_KEY from a .env file.
2. Create (or reuse) a custom voice from the sample clip `Sample 1 comp (mp3cut.net).mp3`.
3. Train a tiny model that maps football‑scenario text ➜ [stability, similarity_boost, speed].
4. Generate speech for a demo line using the predicted sliders.
5. Stream the audio to `goal_of_season.mp3` without loading the whole file into RAM.

Replace the `train_samples` list with your own labelled data to improve the model.
"""

import os
from pathlib import Path
import joblib
import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# 0.  Authentication & client
# -----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ELEVENLABS_API_KEY in your environment or .env file")

client = ElevenLabs(api_key=API_KEY)

# -----------------------------------------------------------------------------
# 1.  Create or reuse a custom voice from a sample MP3
# -----------------------------------------------------------------------------
VOICE_CACHE = "voice_id.txt"               # stores the created voice ID
SAMPLE_FILE = "Sample 1 comp (mp3cut.net).mp3"  # your reference clip

def get_or_create_voice(name: str = "Football‑Caster") -> str:
    """Return an existing voice_id or create a new voice from SAMPLE_FILE."""
    cache_path = Path(VOICE_CACHE)
    if cache_path.exists():
        return cache_path.read_text().strip()

    sample_path = Path(SAMPLE_FILE).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    resp = client.voices.add(
        name=name,
        files=[str(sample_path)],          # *** LIST of file paths ***
        #remove_background_noise=True,
    )
    voice_id = resp.voice_id              # e.g. "Wg2U52ta8B8AKScD9PqE"
    cache_path.write_text(voice_id)
    print(f"Created new voice '{name}' with id {voice_id}")
    if resp.requires_verification:
        print("⚠️  Voice requires verification in the ElevenLabs dashboard before use.")
    return voice_id

VOICE_ID = get_or_create_voice()

# -----------------------------------------------------------------------------
# 2.  Minimal training data – replace with your own labelled scenarios
# -----------------------------------------------------------------------------
train_samples = [
    ("GOAL! A stunning strike from the edge of the box!",                   [0.15, 0.90, 1.10]),
    ("Yellow card shown for dissent; the referee had no choice.",           [0.60, 0.85, 0.95]),
    ("The teams are lining up for the national anthem.",                    [0.40, 0.80, 0.90]),
    ("Penalty saved! The keeper guessed the right way!",                    [0.20, 0.88, 1.08]),
]

# -----------------------------------------------------------------------------
# 3.  Fit or load a multi‑output regression model
# -----------------------------------------------------------------------------
MODEL_FILE = "param_model.joblib"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if Path(MODEL_FILE).exists():
    model = joblib.load(MODEL_FILE)
else:
    X = embedder.encode([text for text, _ in train_samples])
    y = np.array([vals for _, vals in train_samples])
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Trained and saved new parameter‑prediction model →", MODEL_FILE)

# -----------------------------------------------------------------------------
# 4.  Helper: generate speech for a given line of commentary
# -----------------------------------------------------------------------------

def tts_for_line(text: str, out_path: str = "speech.mp3") -> None:
    """Predict sliders, call TTS, stream to an MP3 file."""
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
        f"🔊  Saved {out_path}  |  stability={settings.stability:.2f}  "
        f"similarity={settings.similarity_boost:.2f}  speed={settings.speed:.2f}"
    )

# -----------------------------------------------------------------------------
# 5.  Quick demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo_line = "What a magnificent volley—surely a contender for goal of the season!"
    tts_for_line(demo_line, "goal_of_season.mp3")

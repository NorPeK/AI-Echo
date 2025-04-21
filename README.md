AI Echo – Real‑Time Arabic Football Commentary for Global Audiences
==================================================================

> **“Millions of football fans miss out on the excitement of Arabic play‑by‑play because translations lose the timing, tone, and raw emotion of legends like Fahd Al‑Otaibi.”**

**AI Echo** bridges that gap by **translating live Arabic commentary into other languages _and_ re‑synthesising it in the original commentator’s voice and style**.  
This repository contains everything you need to reproduce our latest prototype:

* **Voice isolation & cloning** (Lalal.ai → ElevenLabs custom voice)  
* **Real‑time Arabic → English pipeline** powered by Whisper + GPT‑4o  
* **Tiny ML model** that predicts optimal ElevenLabs “stability / similarity boost / speed” sliders from text using TF‑IDF + Random Forest  
* Python helpers for data collection, model retraining, and future expansion

---

Key Features
------------

| Feature | What it does |
|---------|--------------|
| **Real‑time pipeline** | Arabic → English (or any target language) with sub‑second TTS latency – suitable for live broadcasts. |
| **Voice preservation** | ElevenLabs custom‑voice keeps Fahd Al‑Otaibi’s timbre, energy, and pacing. |
| **Emotion‑aware synthesis** | TF‑IDF + Random Forest predicts `stability`, `similarity_boost`, and `speed` per line to match the moment’s intensity. |
| **Noise removal** | Lalal.ai “voice” stem extraction cleans raw broadcast audio before cloning or ASR. |
| **Data‑centric workflow** | Scripts to scrape, clean, and segment YouTube commentary plus aligned transcripts. |
| **Easily extensible** | Swap in new commentators, languages, or even other sports with minimal config changes. |

---

Repository Structure
--------------------

```
.
├─ football_tts_pipeline.py      # End‑to‑end demo (YouTube → clean voice → Whisper → GPT → ElevenLabs)
├─ example.py                    # One‑liner demo with hand‑tuned sliders
├─ data/                         # Place raw & processed audio / transcripts here
├─ models/
│   ├─ tfidf_vectorizer.pkl      # Saved TF‑IDF vocabulary
│   └─ text_prediction_model.pkl # Saved RandomForestRegressor weights
├─ .env.example                  # Template for all API keys
└─ README.md
```

---

Quick Start
-----------

1. **Clone & install deps**

   ```bash
   git clone https://github.com/<your‑org>/ai‑echo.git
   cd ai‑echo

   # Core libs
   pip install -r requirements.txt
   # Or, manually:
   pip install yt_dlp requests whisper openai ffmpeg-python                elevenlabs python-dotenv scikit-learn joblib
   ```

2. **Add your keys**

   ```bash
   cp .env.example .env
   # then edit .env and supply:
   #   LALAL_API_KEY=...
   #   OPENAI_API_KEY=...
   #   ELEVENLABS_API_KEY=...
   ```

3. **Run the minimal demo**

   ```bash
   python example.py
   # ⇒ speech_custom5.mp3 saved with Arabic‑style English commentary
   ```

4. **Full end‑to‑end pipeline**

   ```bash
   python football_tts_pipeline.py        --youtube "https://www.youtube.com/watch?v=<match‑clip>"
   # ⇒ commentary_tts.mp3 generated with ML‑predicted sliders
   ```

   *Alternatively*, point it at a local file by editing `LOCAL_FILE` in the script.

---

How It Works
------------

| Stage | Tooling | Purpose |
|-------|---------|---------|
| **1. Audio download / trim** | `yt_dlp`, `ffmpeg` | Grab best‑quality audio and cap to ≤ 60 s for quick iteration. |
| **2. Voice isolation** | `lalal.ai` Perseus model | Separate commentator’s voice track from stadium noise & crowd. |
| **3. Voice cloning** | **ElevenLabs** | Upload isolated sample once → receive persistent `voice_id`. |
| **4. ASR** | `openai‑whisper` | Transcribe Arabic speech. |
| **5. Polishing** | **GPT‑4o‑mini** | Fix typos / player names _in Arabic_. |
| **6. Translation** | **GPT‑4o‑mini** | Translate polished Arabic into idiomatic English commentary. |
| **7. Slider inference** | `scikit‑learn` (TF‑IDF + RF) | Map each English line → `[stability, similarity_boost, speed]`. |
| **8. TTS streaming** | **ElevenLabs** | Generate English speech in Fahd Al‑Otaibi’s voice. |

---

Retraining the Slider Model
---------------------------

```bash
python retrain_slider_model.py     --csv  training_sentences.csv     --out  models/
```

The script:

1. Fits a `TfidfVectorizer` on your English commentary corpus.  
2. Trains a `RandomForestRegressor` to predict `[stability, similarity_boost, speed]` from text.  
3. Saves `tfidf_vectorizer.pkl` and `text_prediction_model.pkl` into `models/`.

---

Data
----

| Type | Source | Usage |
|------|--------|-------|
| **Audio** | YouTube match clips (HD) | Clone voice; analyse tempo & pitch. |
| **Transcripts** | Whisper → manual clean‑up → time‑aligned | Train translation prompts; fine‑tune ASR if needed. |
| **Arabic jargon DB** | Curated TSV | Keep football terminology consistent. |

---

Vision & Impact
---------------

* **Enhance fan engagement** for non‑Arabic speakers and expatriates.  
* **Support Saudi Vision 2030** by globalising local sports content ahead of the 2034 FIFA World Cup.

---

Roadmap
-------

- [ ] Curate 10 h labelled Arabic commentary for ASR fine‑tuning  
- [ ] Live ASR → translation → TTS WebSocket loop (< 1 s total)  
- [ ] Add Spanish & French outputs  
- [ ] Deploy FastAPI micro‑service on GPU‑enabled edge nodes  
- [ ] Provide an OBS / vMix plugin for broadcasters  

---

Contributing
------------

Pull requests are welcome! Please open an issue first to discuss major changes.  
Run `pre‑commit install` and make sure all checks pass.

---

Team
----

| Name | Role |
|------|------|
| **Mohammed Hashyeshu** | ML & TTS Engineer |
| **Nour‑Allah Bek**     | Data & Backend |
| **Hamed Baageel**      | Product & Integration |

---

License
-------

MIT License – see `LICENSE` for full text.

---

> _“AI Echo brings the roar of Arabic football to every corner of the world—without losing a decibel of passion.”_

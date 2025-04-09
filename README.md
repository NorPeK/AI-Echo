AI Echo – Real‑Time Arabic Sports Commentary for Global Audiences
=================================================================

> **“Millions of football fans miss out on the excitement of Arabic play‑by‑play because translations lose the timing, tone, and raw emotion of legends like Fahd Al‑Otaibi.”**

AI Echo bridges that gap by **translating live Arabic commentary into other languages _and_ re‑synthesising it in the original commentator’s voice and style**.  
The repo contains everything you need to reproduce our prototype: Python code for voice cloning, real‑time TTS generation, a mini ML model that predicts optimal ElevenLabs “slider” parameters from text, and instructions for data collection & future expansion.

-----------------------------------------------------------------

Key Features
------------
* **Real‑time pipeline**: Arabic → English (or any target language) with latency suitable for live broadcasts.  
* **Voice preservation**: Uses ElevenLabs custom‑voice cloning to keep Fahd Al‑Otaibi’s timbre, energy, and pacing.  
* **Emotion‑aware synthesis**: Tiny ML model predicts _stability_, _similarity_boost_, and _speed_ per line to match the moment’s intensity.  
* **Data‑centric approach**: Scripted tools to scrape, clean, and segment YouTube commentary audio plus aligned transcripts.  
* **Scalable**: Swap in other commentators, target languages, or even sports with minimal changes.

-----------------------------------------------------------------

Repository Structure
--------------------
.
├─ example.py            # Minimal TTS call with hand‑tuned sliders
├─ tts_pipeline.py       # End‑to‑end demo: voice creation + ML slider prediction
├─ data/                 # Place raw & processed audio / transcripts here
├─ models/               # Saved Sentence‑Transformer & RandomForest weights
├─ .env.example          # Template for ELEVENLABS_API_KEY
└─ README.md

-----------------------------------------------------------------

Quick Start
-----------

1. **Clone & install deps**

   ```bash
   git clone https://github.com/<your‑org>/ai‑echo.git
   cd ai‑echo
   pip install -r requirements.txt
   ```

2. **Add your ElevenLabs key**

   ```bash
   cp .env.example .env
   # then edit .env and paste ELEVENLABS_API_KEY=sk‑xxxxx
   ```

3. **Run the minimal demo**

   ```bash
   python example.py
   # ⇒ speech_custom5.mp3 saved with Arabic‑style English commentary
   ```

4. **Full pipeline**

   ```bash
   python tts_pipeline.py
   # ⇒ goal_of_season.mp3 generated with ML‑predicted sliders
   ```

-----------------------------------------------------------------

How It Works
------------

| Stage | Tooling | Details |
|-------|---------|---------|
| **1. Voice cloning** | `ElevenLabs` | Upload a short Fahd Al‑Otaibi clip → receive `voice_id`. |
| **2. Text embedding** | `sentence-transformers` | `all-MiniLM-L6-v2` encodes each commentary line. |
| **3. Slider regression** | `scikit-learn` | `RandomForestRegressor` predicts `[stability, similarity_boost, speed]`. |
| **4. TTS streaming** | `ElevenLabs` streaming API | Convert text to audio iterator, write to disk chunk‑by‑chunk. |

-----------------------------------------------------------------

Data
----

| Type | Source | Usage |
|------|--------|-------|
| **Audio** | YouTube match clips | Train voice clone & analyse speech patterns. |
| **Transcripts** | Manually translated & time‑aligned | Fine‑tune translation model; feed TTS. |
| **Arabic terminology DB** | Curated | Ensure football jargon stays accurate. |

-----------------------------------------------------------------

Vision & Impact
---------------

* **Enhance fan engagement** for non‑Arabic speakers and expatriates in KSA.  
* **Aligns with Saudi Vision 2030** by globalising local sports content ahead of the 2034 FIFA World Cup.  

-----------------------------------------------------------------

Roadmap
-------

- [ ] Collect larger, labelled Arabic commentary corpus  
- [ ] Integrate live ASR → MT → TTS streaming loop  
- [ ] Add Spanish & French targets  
- [ ] Deploy as low‑latency micro‑service (FastAPI + WebSocket)  
- [ ] Real‑time broadcast plugin for OTT platforms  

-----------------------------------------------------------------

Contributing
------------

Pull requests are welcome! Please open an issue first to discuss major changes.  
Make sure to run `pre-commit` hooks and include relevant unit tests.

-----------------------------------------------------------------

Team
----

| Name | Role |
|------|------|
| Mohammed Hashyeshu | ML & TTS Engineer |
| Nour‑Allah Bek | Data & Backend |
| Hamed Baageel | Product & Integration |

-----------------------------------------------------------------

License
-------

This project is licensed under the MIT License — see `LICENSE` for details.

-----------------------------------------------------------------

_“AI Echo brings the roar of Arabic football to every corner of the world—without losing a decibel of passion.”_

# example.py  – customise stability / similarity / speed
"""
pip install elevenlabs python-dotenv
# optional: pip install "elevenlabs[pyaudio]"  (or have ffmpeg/mpv in PATH)
Put ELEVENLABS_API_KEY=sk-xxxxx... in a .env file next to this script.
"""

import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings, play   # <-- VoiceSettings lives here

# 1. auth --------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise RuntimeError("Set ELEVENLABS_API_KEY in your environment or .env")

client = ElevenLabs(api_key=api_key)

# 2. define per‑request voice settings ---------------------------------------
custom_settings = VoiceSettings(
    stability=0.25,         # expressive
    similarity_boost=0.80,  # stay close to the original timbre
    speed=0.85,             # 8 % slower than default
    # style and use_speaker_boost are optional too
)

# 3. convert text -> audio iterator ------------------------------------------
audio_iter = client.text_to_speech.convert(
    text=(
        "If they lose this one, it could be the end of their hopes for the season. Everything is on the line today!"  
        "Including the Copa del Rey—the stakes couldn’t be higher."  
        "What a fantastic piece of play from Sergio Roberto in the middle, keeping possession under pressure!"  
        "Barcelona breaking quickly, this is a dangerous counter-attack!"  
        "Five against three, Barcelona have the numbers—this could be crucial!"  
        "Luis Suárez with a chance to put this away—he’ll be looking to finish!"  
        "Now it’s with Gomes, he plays a brilliant pass forward."  
        "A stunning ball, perfectly weighted and inch-perfect."  
        "Oh my goodness, Messi lines it up—he shoots... and it’s in! Messi has done it again! What a sensational strike!"  
        "Messi, no fear, no hesitation! Barcelona fans are absolutely jubilant!"  
        "The all-time top scorer in El Clasico history, and he delivers once again!"  
        "He does it in the toughest of settings, under the brightest of lights."  
        "Right here, on his home turf, with the world watching."  
        "And in his favorite stadium—what a goal from Lionel Messi!"  
        "From Rosario to the world, Messi’s done it again!"  
        "Sergi Roberto’s run was crucial—what an incredible contribution!"  
        "The man behind the comeback—Sergi Roberto has been involved in everything today."  
        "And that assist from Jordi Alba—absolutely perfect, Messi couldn’t have asked for more!"  
        "This is Messi’s moment, his magic!"
    ),
    voice_id="Wg2U52ta8B8AKScD9PqE", # voice that we got from Elevenlabs API by using arabic commentary from Fahad Alotaibi
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
    voice_settings=custom_settings,       
)


# 4. stream chunks straight to disk ------------------------------------------
out_path = "speech_custom5.mp3"
with open(out_path, "wb") as f:
    for chunk in audio_iter:
        if isinstance(chunk, bytes):       # skip occasional JSON events
            f.write(chunk)

print(f"Saved to {out_path}")

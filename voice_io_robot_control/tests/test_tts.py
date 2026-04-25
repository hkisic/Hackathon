"""Test local TTS."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import TTS_RATE, TTS_VOLUME
from tts_engine import TTSEngine, list_voices

print("Available voices:")
list_voices()

print("Speaking test sentence...")
tts = TTSEngine(rate=TTS_RATE, volume=TTS_VOLUME)
tts.speak("Hello, I am your robot dog.")

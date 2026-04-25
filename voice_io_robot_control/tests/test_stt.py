"""Test microphone recording and local speech recognition."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MIC_DEVICE, SAMPLE_RATE, RECORD_SECONDS, WHISPER_MODEL, AUDIO_FILE
from stt_engine import STTEngine


stt = STTEngine(
    mic_device=MIC_DEVICE,
    sample_rate=SAMPLE_RATE,
    record_seconds=RECORD_SECONDS,
    model_name=WHISPER_MODEL,
    audio_file=AUDIO_FILE,
)

text = stt.listen_once()
print("Final STT result:", text)

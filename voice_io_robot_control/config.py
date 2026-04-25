"""
Project configuration for the robot dog AI system.
Edit this file first when your COM port, microphone, or model changes.
"""

# Serial / robot settings
ROBOT_PORT = "COM3"
ROBOT_BAUD = 115200
SERIAL_TIMEOUT = 0.5

# Microphone / STT settings
MIC_DEVICE = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 4
AUDIO_FILE = "test.wav"
WHISPER_MODEL = "base"  # use "base.en" if you only speak English

# TTS settings
TTS_RATE = 150
TTS_VOLUME = 1.0

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"

# Robot behavior
PARALLEL_TTS_AND_ACTION = True
SET_ROBOT_VOLUME_MIN_ON_START = True

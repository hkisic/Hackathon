"""Text-to-speech engine using Windows local pyttsx3."""

import pyttsx3


class TTSEngine:
    def __init__(self, rate=150, volume=1.0, voice_index=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        if voice_index is not None:
            voices = self.engine.getProperty("voices")
            if 0 <= voice_index < len(voices):
                self.engine.setProperty("voice", voices[voice_index].id)

    def speak(self, text: str):
        if not text:
            return
        print("TTS:", text)
        self.engine.say(text)
        self.engine.runAndWait()


def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} | {voice.id}")

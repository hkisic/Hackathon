"""Speech-to-text engine using local faster-whisper."""

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


class STTEngine:
    def __init__(
        self,
        mic_device=1,
        sample_rate=16000,
        record_seconds=4,
        model_name="base",
        audio_file="test.wav",
    ):
        self.mic_device = mic_device
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.audio_file = audio_file
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def record(self):
        print("Recording...")
        print("Speak now.")
        audio = sd.rec(
            int(self.record_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.mic_device,
        )
        sd.wait()
        write(self.audio_file, self.sample_rate, audio)
        print("Saved:", self.audio_file)
        return self.audio_file

    def transcribe(self):
        print("Transcribing...")
        segments, _ = self.model.transcribe(
            self.audio_file,
            language=None,
            beam_size=5,
            vad_filter=True,
            initial_prompt=(
                "Robot dog commands and normal conversation. "
                "Commands include stand, down, squat, plank, handshake, sleep, "
                "forward, backward, turn right, turn left, swing, push up, "
                "face change, twist, shake, tail, follow, patrol."
            ),
        )

        text = ""
        for segment in segments:
            text += segment.text

        text = text.strip()
        print("Recognized:", text)
        return text

    def listen_once(self):
        self.record()
        return self.transcribe()

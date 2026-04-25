"""Main entry point for the robot dog AI system."""

import time
import threading

from config import (
    ROBOT_PORT,
    ROBOT_BAUD,
    SERIAL_TIMEOUT,
    SET_ROBOT_VOLUME_MIN_ON_START,
    MIC_DEVICE,
    SAMPLE_RATE,
    RECORD_SECONDS,
    AUDIO_FILE,
    WHISPER_MODEL,
    TTS_RATE,
    TTS_VOLUME,
    OPENAI_MODEL,
    PARALLEL_TTS_AND_ACTION,
)
from robot_io import RobotIO
from stt_engine import STTEngine
from tts_engine import TTSEngine
from llm_decision import LLMDecision
from camera_emotion import CameraEmotionReader
from eeg_reader import EEGReader
from robot_action_reference import ACTIONS


def run_tts_and_action(tts, robot, say, action_id, parallel=True):
    if action_id is None:
        tts.speak(say)
        return

    action = ACTIONS.get(action_id)
    wait_time = action.estimated_wait_s if action else 3.0

    if parallel:
        t = threading.Thread(target=tts.speak, args=(say,))
        t.start()
        robot.send_action_id(action_id, wait=False)
        t.join()
        print(f"Waiting {wait_time} seconds for robot action...")
        time.sleep(wait_time)
    else:
        robot.send_action_id(action_id, wait=True)
        tts.speak(say)


def main():
    robot = RobotIO(
        port=ROBOT_PORT,
        baud=ROBOT_BAUD,
        timeout=SERIAL_TIMEOUT,
        set_volume_min=SET_ROBOT_VOLUME_MIN_ON_START,
    )
    stt = STTEngine(
        mic_device=MIC_DEVICE,
        sample_rate=SAMPLE_RATE,
        record_seconds=RECORD_SECONDS,
        model_name=WHISPER_MODEL,
        audio_file=AUDIO_FILE,
    )
    tts = TTSEngine(rate=TTS_RATE, volume=TTS_VOLUME)
    brain = LLMDecision(model=OPENAI_MODEL)
    camera = CameraEmotionReader()
    eeg = EEGReader()

    print("Robot AI system started.")
    print("Press Enter to speak. Type q to quit.")

    try:
        while True:
            cmd = input("\nPress Enter to talk, or q to quit: ").strip().lower()
            if cmd in ["q", "quit", "exit"]:
                break

            user_text = stt.listen_once()
            if not user_text:
                tts.speak("I did not hear anything.")
                continue

            emotion = camera.read()
            eeg_state = eeg.read()

            result = brain.decide(
                user_text=user_text,
                emotion=emotion,
                eeg_state=eeg_state,
            )

            print("Decision:", result)
            run_tts_and_action(
                tts=tts,
                robot=robot,
                say=result["say"],
                action_id=result["action_id"],
                parallel=PARALLEL_TTS_AND_ACTION,
            )

    finally:
        robot.close()
        print("Robot closed.")


if __name__ == "__main__":
    main()

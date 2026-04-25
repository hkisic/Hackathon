"""Test LLM decision without microphone or robot."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import OPENAI_MODEL
from llm_decision import LLMDecision

brain = LLMDecision(model=OPENAI_MODEL)

while True:
    text = input("User text, or q to quit > ").strip()
    if text.lower() in ["q", "quit", "exit"]:
        break
    result = brain.decide(text, emotion={"label": "unknown"}, eeg_state={"attention": None})
    print("Decision:", result)

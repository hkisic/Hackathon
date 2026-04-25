"""LLM decision layer: user text + emotion + EEG -> say + action_id."""

import os
import json
from openai import OpenAI
from robot_action_reference import ACTIONS, build_model_action_prompt


class LLMDecision:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def decide(self, user_text: str, emotion=None, eeg_state=None):
        prompt = f"""
You are the decision system for a small robot dog.

{build_model_action_prompt()}

Inputs:
- User speech text: {user_text}
- Camera emotion result: {emotion}
- EEG state result: {eeg_state}

Decide what the robot should say and what action it should do.

Return only valid JSON in this exact format:
{{
  "say": "short English sentence for TTS",
  "action_id": 8
}}

Rules:
- action_id must be one supported robot action ID or null.
- If the user only chats, use null.
- If emotion or EEG suggests the user is stressed, respond gently.
- Keep the sentence short and natural.
- Do not output markdown.
- Do not explain.
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw = response.output_text.strip()
        print("LLM raw:", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"say": "Sorry, I did not understand.", "action_id": None}

        say = data.get("say", "Okay.")
        action_id = data.get("action_id", None)

        if action_id is not None:
            try:
                action_id = int(action_id)
            except (ValueError, TypeError):
                action_id = None

        if action_id not in ACTIONS:
            action_id = None

        return {"say": say, "action_id": action_id}

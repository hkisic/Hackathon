epet robot dog AI project
File roles
`config.py`: all editable settings: COM port, microphone, model names, TTS speed.
`robot_action_reference.py`: robot action IDs, action names, wait times, and serial controller.
`robot_io.py`: simple wrapper for robot serial I/O.
`stt_engine.py`: microphone recording + local faster-whisper speech-to-text.
`tts_engine.py`: local text-to-speech using pyttsx3.
`llm_decision.py`: OpenAI API decision layer. It returns `say` and `action_id`.
`camera_emotion.py`: placeholder for future camera emotion recognition.
`eeg_reader.py`: placeholder for future EEG headset integration.
`main.py`: final main program.
`tests/`: individual test scripts.
Recommended test order
Edit `config.py`.
Test TTS:
```bash
   python tests/test_tts.py
   ```
Test STT:
```bash
   python tests/test_stt.py
   ```
Test robot serial:
```bash
   python tests/test_robot_serial.py
   ```
Test API decision:
```bash
   python tests/test_api_decision.py
   ```
Run full system:
```bash
   python main.py
   ```
Install dependencies
```bash
pip install pyserial pyttsx3 sounddevice scipy faster-whisper openai
```
API key
Windows PowerShell:
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```
Then reopen your terminal or IDE.

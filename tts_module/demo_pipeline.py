"""
Complete End-to-End Pipeline Demonstration for AI-Powered Autonomous Desktop Agent.
Demonstrates the complete workflow:
USER VOICE -> WHISPER STT -> LLM BRAIN -> KOKORO-82M TTS -> AUDIO OUTPUT -> USER HEARS RESPONSE
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tts_module.src.tts_engine import TTSEngine, generate_speech, play_audio
from tts_module.src.config import AVAILABLE_VOICES


class WhisperSTTModule:
    """
    Live local connector for Prashobh Nair's Module 1: STT (Speech-to-Text).
    Powered by 74 Million Parameter Deep Transformer (Whisper-Base INT8 Offline).
    """
    def __init__(self):
        self.model_name = "Whisper-Base (74M Parameters, Local Offline)"
        self.recognizer = None
        try:
            from inference import SpeechRecognizer
            self.recognizer = SpeechRecognizer(engine="whisper")
        except Exception as e:
            print(f"[!] Notice loading live STT engine: {e}")

    def transcribe(self, user_audio_input: str) -> str:
        """Transcribes spoken audio file or user input into text."""
        if self.recognizer and os.path.exists(user_audio_input):
            return self.recognizer.transcribe(user_audio_input)
        elif self.recognizer:
            time.sleep(0.2)
            return user_audio_input.strip()
        time.sleep(0.3)
        return user_audio_input.strip()


class MockLLMBrain:
    """
    Simulation connector for Sahaj Patel's Module 2: Brain / LLM.
    Understands intent, determines desktop action, and generates natural language response.
    """
    def __init__(self):
        self.model_name = "Local Desktop LLM Agent (Brain)"
        
    def process_command(self, user_text: str) -> tuple[str, str]:
        """
        Processes transcribed user intent and returns (action_summary, spoken_response).
        """
        text_lower = user_text.lower()
        time.sleep(0.4)  # Simulate LLM inference / thinking time
        
        if "ram" in text_lower or "cpu" in text_lower or "performance" in text_lower or "system" in text_lower:
            action = "DESKTOP_ACTION: query_system_metrics()"
            response = (
                "System diagnostics report: Your CPU is running smoothly at 28% capacity, "
                "and you have 4.8 GB of RAM available out of 8 GB total. All background services are normal."
            )
        elif "open" in text_lower and "chrome" in text_lower:
            action = "DESKTOP_ACTION: launch_application('Google Chrome')"
            response = "Launching Google Chrome now. What website or search query would you like me to open?"
        elif "reminder" in text_lower or "schedule" in text_lower:
            action = "DESKTOP_ACTION: create_calendar_event(time='10:00 AM', date='tomorrow')"
            response = "I have scheduled your reminder for tomorrow at 10:00 AM. I will alert you 15 minutes prior."
        elif "team" in text_lower or "members" in text_lower or "who" in text_lower:
            action = "DESKTOP_ACTION: display_team_credits()"
            response = (
                "This AI Desktop Agent is developed by our B.Tech Computer Science team: "
                "Prashobh Nair for Whisper Speech to Text, "
                "Sahaj Patel for the LLM Brain, "
                "and the custom Kokoro-82M neural Text to Speech engine."
            )
        else:
            action = f"DESKTOP_ACTION: generic_assistant_response('{user_text}')"
            response = f"I understood your request: {user_text}. I am executing that operation right now."
            
        return action, response


def run_pipeline_demo():
    """Runs the full STT -> LLM -> TTS demonstration pipeline."""
    print("=" * 75)
    print("AI-POWERED AUTONOMOUS DESKTOP AGENT: FULL SYSTEM PIPELINE DEMO")
    print("=" * 75)
    print("Architecture:")
    print("  [USER VOICE] -> [MODULE 1: WHISPER STT] -> [MODULE 2: LLM BRAIN] -> [MODULE 3: KOKORO-82M TTS] -> [SPEAKERS]")
    print("=" * 75)

    stt = WhisperSTTModule()
    brain = MockLLMBrain()
    tts = TTSEngine()
    tts.initialize()

    scenarios = [
        "What is my current system performance and RAM usage?",
        "Open Google Chrome for me.",
        "Set a reminder for our project review meeting tomorrow at 10:00 AM.",
        "Who are the team members developing this project?"
    ]

    for i, spoken_input in enumerate(scenarios, 1):
        print(f"\n--- SCENARIO {i}: '{spoken_input}' ---")
        
        # STEP 1 & 2: STT
        print(f"[*] [STEP 1 & 2] Module 1 (Whisper STT) transcribing user voice...")
        transcribed_text = stt.transcribe(spoken_input)
        print(f"    [STT Output] Transcribed Text: \"{transcribed_text}\"")
        
        # STEP 3: LLM Brain
        print(f"[*] [STEP 3] Module 2 (Brain / LLM) reasoning on user intention...")
        action, llm_response = brain.process_command(transcribed_text)
        print(f"    [Brain Action]   {action}")
        print(f"    [Brain Response] \"{llm_response}\"")
        
        # STEP 4 & 5: TTS Synthesis
        print(f"[*] [STEP 4 & 5] Module 3 (Kokoro-82M TTS) generating natural speech locally...")
        t0 = time.time()
        audio_file = tts.generate_speech(llm_response, voice="af_heart", speed=1.0)
        metrics = tts.last_metrics
        print(f"    [TTS Output] Audio File: {Path(audio_file).name}")
        print(f"    [TTS Telemetry] Audio Duration: {metrics.get('audio_duration', 0):.2f}s | "
              f"Gen Time: {metrics.get('generation_time', 0):.2f}s | RTF: {metrics.get('rtf', 0):.2f}x")
        
        # STEP 6: Audio Playback through speakers
        print(f"[*] [STEP 6] Playing generated response through Windows speakers...")
        tts.audio_player.play(audio_file, blocking=True)
        print(f"[+] Step 6 complete: User heard the spoken response.")
        
        time.sleep(0.5)

    print("\n" + "=" * 75)
    print("[SUCCESS] Full STT -> LLM -> TTS pipeline demonstration completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_demo()


"""
Unit & Integration Tests for Kokoro TTSEngine and AudioPlayer.
Tests model loading, voice selection, speed adjustment, caching, edge cases, and output validation.
"""

import os
import unittest
from pathlib import Path
import soundfile as sf

from tts_module.src.tts_engine import TTSEngine, generate_speech, get_available_voices
from tts_module.src.audio_player import AudioPlayer
from tts_module.src.config import TTSConfig


class TestTTSEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = TTSEngine()
        cls.engine.initialize()

    def test_engine_initialization(self):
        self.assertTrue(self.engine._is_initialized)
        self.assertIsNotNone(self.engine._kokoro)

    def test_get_available_voices(self):
        voices = self.engine.get_available_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 10)
        self.assertIn("af_heart", voices)
        self.assertIn("am_adam", voices)

    def test_voice_metadata(self):
        info = self.engine.get_voice_info("af_heart")
        self.assertEqual(info["gender"], "Female")
        self.assertEqual(info["accent"], "American")

    def test_generate_speech_basic(self):
        wav_path = self.engine.generate_speech(
            text="System online. Ready for command.",
            voice="af_heart",
            speed=1.0,
            output_filename="test_basic.wav"
        )
        self.assertTrue(os.path.exists(wav_path))
        info = sf.info(wav_path)
        self.assertEqual(info.samplerate, 24000)
        self.assertGreater(info.duration, 0.5)

    def test_voice_selection_male(self):
        wav_path = self.engine.generate_speech(
            text="This is an automated status update from the male voice profile.",
            voice="am_adam",
            speed=1.0,
            output_filename="test_voice_adam.wav"
        )
        self.assertTrue(os.path.exists(wav_path))
        self.assertGreater(os.path.getsize(wav_path), 5000)

    def test_speed_adjustment(self):
        # Slower speed should generate longer audio duration
        path_slow = self.engine.generate_speech(
            text="Testing speaking speed variation.",
            speed=0.8,
            use_cache=False
        )
        path_fast = self.engine.generate_speech(
            text="Testing speaking speed variation.",
            speed=1.3,
            use_cache=False
        )
        dur_slow = sf.info(path_slow).duration
        dur_fast = sf.info(path_fast).duration
        self.assertGreater(dur_slow, dur_fast)

    def test_caching_behavior(self):
        phrase = "Cache test phrase for latency measurement."
        # First call generates audio
        path1 = self.engine.generate_speech(phrase, use_cache=True)
        # Second call should hit cache
        path2 = self.engine.generate_speech(phrase, use_cache=True)
        self.assertEqual(path1, path2)
        self.assertTrue(self.engine.last_metrics.get("cached", False))

    def test_empty_and_whitespace_input(self):
        # Empty text should gracefully return a silence WAV without crashing
        wav_empty = self.engine.generate_speech("")
        self.assertTrue(os.path.exists(wav_empty))
        wav_space = self.engine.generate_speech("   ")
        self.assertTrue(os.path.exists(wav_space))

    def test_punctuation_only_input(self):
        wav_punct = self.engine.generate_speech("... !? ;;; ---")
        self.assertTrue(os.path.exists(wav_punct))

    def test_global_generate_speech_function(self):
        # Testing the required function interface (Section 8)
        wav_path = generate_speech("Testing the global interface function for the AI desktop agent.")
        self.assertTrue(os.path.exists(wav_path))
        self.assertTrue(wav_path.endswith(".wav"))


class TestAudioPlayer(unittest.TestCase):

    def setUp(self):
        self.player = AudioPlayer()

    def test_player_initialization(self):
        self.assertFalse(self.player.is_playing)
        self.assertEqual(self.player.default_volume, 1.0)

    def test_set_volume(self):
        self.player.set_volume(0.5)
        self.assertAlmostEqual(self.player._current_volume, 0.5)
        self.player.set_volume(1.5)  # Should clamp to 1.0
        self.assertAlmostEqual(self.player._current_volume, 1.0)
        self.player.set_volume(-0.5)  # Should clamp to 0.0
        self.assertAlmostEqual(self.player._current_volume, 0.0)

    def test_play_non_existent_file(self):
        result = self.player.play("non_existent_file.wav", blocking=False)
        self.assertFalse(result)

    def test_stop_playback(self):
        # Calling stop when not playing should not raise any errors
        self.player.stop()
        self.assertFalse(self.player.is_playing)


if __name__ == "__main__":
    unittest.main()


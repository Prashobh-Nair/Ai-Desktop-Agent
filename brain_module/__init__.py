"""
Module 2: Brain / LLM (Autonomous Desktop Task Reasoning & Intent Understanding)
Developed by Sahaj Patel

This module interprets transcribed voice commands from Module 1 (STT),
extracts structured intents and action steps, enforces confirmation for sensitive operations,
and returns natural language responses ready for synthesis by Module 3 (TTS).
"""

from .jarvis.brain import Brain, make_reply
from .jarvis.schema import Action, Intent, Plan, Step, enforce_safety
from .jarvis.llm import JarvisModel, ModelUnavailable, BadModelOutput
from .jarvis.planner import RulesPlanner
from .jarvis.normalize import normalize_text, resolve_app, split_clauses

__all__ = [
    "Brain",
    "make_reply",
    "Action",
    "Intent",
    "Plan",
    "Step",
    "enforce_safety",
    "JarvisModel",
    "RulesPlanner",
    "ModelUnavailable",
    "BadModelOutput",
    "normalize_text",
    "resolve_app",
    "split_clauses",
]

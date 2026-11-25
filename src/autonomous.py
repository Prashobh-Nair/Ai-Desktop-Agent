"""
Phase 6: Autonomous JARVIS — Multi-Step Goal Execution, Background Tasks,
Proactive Notifications, Error Recovery & Permission-Based Execution.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, List, Optional


class BackgroundTaskManager:
    """Runs background tasks and queues proactive completion notifications."""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.notifications: queue.Queue[str] = queue.Queue()
        self._counter = 0
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable[[], str]) -> str:
        with self._lock:
            self._counter += 1
            task_id = f"task-{self._counter}"
            self.tasks[task_id] = {
                "id": task_id,
                "name": name,
                "status": "running",
                "result": None,
                "started_at": time.strftime("%H:%M:%S"),
            }

        def _runner():
            try:
                res = fn()
                with self._lock:
                    self.tasks[task_id]["status"] = "done"
                    self.tasks[task_id]["result"] = res
                self.notifications.put(f"Background task '{name}' finished: {res}")
            except Exception as e:
                with self._lock:
                    self.tasks[task_id]["status"] = "failed"
                    self.tasks[task_id]["result"] = str(e)
                self.notifications.put(f"Background task '{name}' failed: {e}")

        threading.Thread(target=_runner, daemon=True).start()
        return task_id

    def pop_notifications(self) -> List[str]:
        notes = []
        while not self.notifications.empty():
            try:
                notes.append(self.notifications.get_nowait())
            except queue.Empty:
                break
        return notes

    def status_summary(self) -> str:
        with self._lock:
            if not self.tasks:
                return "No background tasks have been run yet."
            recent = list(self.tasks.values())[-5:]
        return "; ".join(f"{t['id']} ({t['name']}): {t['status']}" for t in recent)


class AutonomousGoalRunner:
    """Executes multi-step objectives with tool feedback, permission checks & error recovery."""

    def __init__(self, brain, executor, vision=None):
        self.brain = brain
        self.executor = executor
        self.vision = vision

    def run_objective(
        self,
        objective: str,
        session_id: str = "live_jarvis",
        confirm_fn: Optional[Callable[[str], bool]] = None,
        max_rounds: int = 3,
    ) -> tuple[str, List[str]]:
        """
        Executes an objective across up to `max_rounds` plan->execute->observe cycles.
        Returns (final_spoken_reply, all_action_logs).
        """
        all_logs: List[str] = []
        current_prompt = objective
        final_reply = "Done."

        for round_idx in range(max_rounds):
            out = self.brain.plan(current_prompt, session_id=session_id)
            plan = out.get("plan", {})
            final_reply = plan.get("reply") or final_reply
            steps = plan.get("steps") or []

            if not steps:
                break

            # Permission check for risky actions (shutdown, restart, terminal commands)
            if plan.get("needs_confirmation"):
                if confirm_fn and not confirm_fn(final_reply):
                    all_logs.append("Cancelled by user permission check.")
                    return "Action cancelled.", all_logs

            round_logs: List[str] = []
            # Handle screen vision step if present
            if any(s.get("action") == "see_screen" for s in steps) and self.vision is not None:
                vis_res = self.vision.analyze_screen(current_prompt)
                round_logs.append(vis_res)
                plan["steps"] = [s for s in steps if s.get("action") != "see_screen"]

            if plan["steps"]:
                round_logs.extend(self.executor.execute_plan(plan, memory=self.brain.memory))

            all_logs.extend(round_logs)

            # Check if any tool returned useful data (git/file/memory/weather/calendar/terminal) or an error
            joined = " | ".join(round_logs)
            has_error = any(k in joined.lower() for k in ("error", "failed", "not found", "exit "))
            needs_followup = any(
                s.get("action") in ("file_op", "git_op", "memory_op", "calendar_op", "web_fetch", "run_terminal", "see_screen")
                for s in steps
            )

            if has_error and round_idx + 1 < max_rounds:
                current_prompt = (
                    f"Objective was: {objective}\n"
                    f"Previous step output/error: {joined}\n"
                    "Recover from the error or explain the result clearly in 1-2 sentences."
                )
                continue

            if needs_followup and joined:
                # If single-step tool already returned a clean human-readable result, speak it directly
                if len(round_logs) == 1 and not has_error:
                    final_reply = round_logs[0]
                    break
                current_prompt = (
                    f"Objective was: {objective}\n"
                    f"Tool outputs: {joined}\n"
                    "Summarize the final result concisely in 1-2 sentences for speaking aloud."
                )
                continue
            break

        return final_reply, all_logs

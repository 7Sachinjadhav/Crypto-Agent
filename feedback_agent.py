"""
Agent 5 – Hermes Agnes Loop Feedback Agent
=============================================
Implements the feedback / reflection loop in the Hermes Agnes (agents) pattern.

After a prediction cycle completes, this agent:
  1. Compares predictions vs actual market moves (if available).
  2. Logs the outcome and computes a rolling accuracy score.
  3. Feeds results back as context for the next cycle (Agnes loop).
  4. Suggests parameter adjustments (confidence threshold, Kelly cap).

The Agnes loop runs continuously: predict → act → observe → reflect → repeat.
"""

import json
import re
import os
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.logger import logger


FEEDBACK_LOG = "logs/feedback_history.json"


class FeedbackAgent(BaseAgent):
    name = "FeedbackAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a performance analyst for an autonomous crypto trading system. "
            "Given prediction history and outcomes, compute accuracy metrics and "
            "suggest improvements to model parameters. "
            "Return ONLY a JSON object with keys: "
            "overall_accuracy, asset_accuracy (dict), "
            "suggested_confidence_threshold, suggested_kelly_cap, "
            "insights (list of strings), next_cycle_adjustments (dict)."
        )

    def _load_history(self) -> list[dict]:
        if not os.path.exists(FEEDBACK_LOG):
            return []
        try:
            with open(FEEDBACK_LOG, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self, history: list[dict]) -> None:
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        with open(FEEDBACK_LOG, "w") as f:
            json.dump(history, f, indent=2)

    def record_prediction(
        self,
        predictions: list[dict],
        actual_moves: dict | None = None,
    ) -> list[dict]:
        """
        Record predictions and (optionally) actual moves.

        Args:
            predictions:  List of prediction dicts (asset, direction, confidence).
            actual_moves: {"ETH": "UP", "BTC": "DOWN"} – filled in next cycle.

        Returns:
            Updated history list.
        """
        history = self._load_history()
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": predictions,
            "actual_moves": actual_moves or {},
            "scored": actual_moves is not None,
        }
        history.append(entry)
        self._save_history(history)
        logger.info(f"[{self.name}] Recorded {len(predictions)} predictions to history.")
        return history

    def _score_history(self, history: list[dict]) -> dict:
        """Compute per-asset accuracy from scored history entries."""
        scored = [h for h in history if h.get("scored") and h.get("actual_moves")]
        if not scored:
            return {}

        asset_correct: dict[str, int] = {}
        asset_total:   dict[str, int] = {}

        for entry in scored:
            for pred in entry["predictions"]:
                asset = pred["asset"]
                actual = entry["actual_moves"].get(asset)
                if actual is None:
                    continue
                asset_total[asset] = asset_total.get(asset, 0) + 1
                if pred["direction"] == actual:
                    asset_correct[asset] = asset_correct.get(asset, 0) + 1

        accuracy = {
            a: round(asset_correct.get(a, 0) / asset_total[a], 4)
            for a in asset_total
        }
        return accuracy

    def run(
        self,
        current_predictions: list[dict] | None = None,
        actual_moves: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Run one feedback loop cycle.

        Args:
            current_predictions: Predictions from this cycle.
            actual_moves:        Actual price moves observed (for scoring).

        Returns:
            dict with accuracy metrics and suggestions for next cycle.
        """
        logger.info(f"[{self.name}] Running feedback loop...")

        history = self.record_prediction(
            current_predictions or [],
            actual_moves=actual_moves,
        )

        accuracy = self._score_history(history)
        overall_acc = (
            round(sum(accuracy.values()) / len(accuracy), 4)
            if accuracy else None
        )

        recent_preds = current_predictions or []
        prompt = (
            f"Current predictions: {recent_preds}\n"
            f"Asset accuracy so far: {accuracy}\n"
            f"Overall accuracy: {overall_acc}\n"
            f"Total cycles in history: {len(history)}\n\n"
            "Analyse performance and suggest improvements. Return ONLY valid JSON."
        )
        raw = self._llm(prompt)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            analysis = json.loads(raw)
        except Exception:
            analysis = {
                "overall_accuracy": overall_acc,
                "asset_accuracy": accuracy,
                "suggested_confidence_threshold": 0.55,
                "suggested_kelly_cap": 0.20,
                "insights": ["Insufficient data for deep analysis."],
                "next_cycle_adjustments": {},
            }

        logger.info(
            f"[{self.name}] Feedback | accuracy={overall_acc} | "
            f"suggestions={analysis.get('next_cycle_adjustments')}"
        )
        return {"feedback": analysis, "history_length": len(history)}

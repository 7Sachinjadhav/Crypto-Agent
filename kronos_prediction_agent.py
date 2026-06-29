"""
Agent 3 – Kronos Prediction Agent
=====================================
Predicts the next UP/DOWN move using statistical / ML methods inspired by
the Kronos library (https://github.com/shiyu-coder/Kronos).

Since Kronos is a research repository without a pip package, this agent
implements the same core idea:
  - Markov-chain transition probabilities on discretised price returns
  - Momentum / mean-reversion signal
  - LLM synthesis of signals into a final prediction

Scales to multi-timeframe arbitrage (1-min n+5 used on 5-min n+1) as
suggested in the assessment brief.
"""

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from utils.logger import logger
from config.settings import ASSETS


class KronosPredictionAgent(BaseAgent):
    name = "KronosPredictionAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a quantitative crypto analyst specialising in short-term price prediction. "
            "Given statistical signals (Markov chain probabilities, momentum, volatility) "
            "for a crypto asset over a 5-minute horizon, synthesise a final prediction. "
            "Return ONLY a JSON object with keys: "
            "asset, direction (UP or DOWN), confidence (0.0-1.0), "
            "reasoning (brief), signals_used."
        )

    # ------------------------------------------------------------------
    # Statistical signal computation
    # ------------------------------------------------------------------

    def _compute_returns(self, bars: list[dict]) -> np.ndarray:
        closes = np.array([b["close"] for b in bars], dtype=float)
        returns = np.diff(closes) / closes[:-1]
        return returns

    def _markov_signal(self, returns: np.ndarray, n_states: int = 3) -> dict:
        """
        Discretise returns into states and compute transition matrix.
        States: 0=DOWN, 1=FLAT, 2=UP  (tertiles)
        Returns probability of UP in next bar given current state.
        """
        if len(returns) < 20:
            return {"up_prob": 0.5, "current_state": "UNKNOWN"}

        thresholds = np.percentile(returns, [33, 67])

        def classify(r):
            if r < thresholds[0]:
                return 0
            elif r < thresholds[1]:
                return 1
            return 2

        states = np.array([classify(r) for r in returns])

        # Build transition matrix
        transition = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            transition[states[i], states[i + 1]] += 1

        # Normalise rows
        row_sums = transition.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition /= row_sums

        current_state = int(states[-1])
        up_prob = float(transition[current_state, 2])  # prob of moving to UP state

        return {
            "up_prob": up_prob,
            "current_state": ["DOWN", "FLAT", "UP"][current_state],
            "transition_matrix": transition.tolist(),
        }

    def _momentum_signal(self, returns: np.ndarray, window: int = 12) -> dict:
        """Simple momentum: mean return over last `window` bars."""
        if len(returns) < window:
            return {"momentum": 0.0, "direction": "FLAT"}
        mom = float(np.mean(returns[-window:]))
        return {
            "momentum": mom,
            "direction": "UP" if mom > 0 else "DOWN",
        }

    def _volatility(self, returns: np.ndarray, window: int = 20) -> float:
        if len(returns) < window:
            return float(np.std(returns)) if len(returns) > 1 else 0.0
        return float(np.std(returns[-window:]))

    def _multitimeframe_signal(self, bars: list[dict]) -> dict:
        """
        Internal arbitrage signal:
        Compare 1-min momentum (n+5 look-ahead proxy) vs 5-min momentum (n+1).
        Uses last 60 bars as 1-min proxy and last 12 bars as 5-min proxy.
        """
        returns = self._compute_returns(bars)
        signal_1min = self._momentum_signal(returns, window=60)
        signal_5min = self._momentum_signal(returns, window=12)
        agreement = signal_1min["direction"] == signal_5min["direction"]
        return {
            "signal_1min": signal_1min,
            "signal_5min": signal_5min,
            "agreement": agreement,
            "combined_direction": signal_5min["direction"] if agreement else "FLAT",
        }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, asset_data: dict | None = None, **kwargs) -> dict:
        """
        Generate Kronos-style predictions for all assets.

        Args:
            asset_data: Output from DataFetcherAgent {"ETH": {..., "bars": [...]}, ...}

        Returns:
            {"predictions": [{"asset": ..., "direction": ..., "confidence": ...}, ...]}
        """
        logger.info(f"[{self.name}] Starting Kronos predictions...")
        predictions = []

        for asset in ASSETS:
            bars = []
            if asset_data and asset in asset_data:
                bars = asset_data[asset].get("bars", [])

            if len(bars) < 30:
                logger.warning(f"[{self.name}] Insufficient bars for {asset} ({len(bars)}). Using neutral.")
                predictions.append({
                    "asset": asset,
                    "direction": "UNKNOWN",
                    "confidence": 0.5,
                    "reasoning": "Insufficient data",
                    "signals_used": [],
                })
                continue

            returns = self._compute_returns(bars)
            markov  = self._markov_signal(returns)
            momentum = self._momentum_signal(returns)
            vol = self._volatility(returns)
            mtf = self._multitimeframe_signal(bars)

            # Assemble context for LLM synthesis
            prompt = (
                f"Asset: {asset}\n"
                f"Markov signal: {markov}\n"
                f"Momentum signal: {momentum}\n"
                f"Volatility (20-bar std): {vol:.6f}\n"
                f"Multi-timeframe signal: {mtf}\n\n"
                "Synthesise a 5-minute ahead prediction. Return ONLY valid JSON."
            )
            import json, re
            raw = self._llm(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                pred = json.loads(raw)
                pred["asset"] = asset
            except Exception:
                # Fallback: simple vote between markov + momentum
                votes_up = [
                    markov["up_prob"] > 0.5,
                    momentum["direction"] == "UP",
                    mtf["combined_direction"] == "UP",
                ]
                up_vote = sum(votes_up)
                pred = {
                    "asset": asset,
                    "direction": "UP" if up_vote >= 2 else "DOWN",
                    "confidence": 0.5 + (up_vote - 1.5) * 0.15,
                    "reasoning": "Statistical vote (LLM parse failed)",
                    "signals_used": ["markov", "momentum", "mtf"],
                }

            predictions.append(pred)
            logger.info(f"[{self.name}] {asset}: {pred.get('direction')} @ {pred.get('confidence'):.2f}")

        return {"predictions": predictions}

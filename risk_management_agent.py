"""
Agent 4 – Risk Management Agent
===================================
Applies the Kelly Criterion to size positions based on model predictions.

Combines signals from:
  - PredictionSearchAgent  (crowd wisdom from Polymarket/Kalshi)
  - KronosPredictionAgent  (statistical/ML model)

Weights crowd-wisdom at 40% and model prediction at 60% to compute a
blended confidence score, then passes it to the Kelly formula.

Reference:
  https://managebankroll.com/blog/polymarket-kelly-criterion-position-sizing
  https://mintlify.wiki/joicodev/polymarket-bot/risk/kelly-criterion
"""

import json
import re
from agents.base_agent import BaseAgent
from utils.kelly import kelly_fraction, multi_asset_kelly
from utils.logger import logger
from config.settings import ASSETS, MAX_KELLY_FRACTION


CROWD_WEIGHT  = 0.40
MODEL_WEIGHT  = 0.60


class RiskManagementAgent(BaseAgent):
    name = "RiskManagementAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a risk management specialist for a crypto trading system. "
            "Given blended prediction signals and Kelly Criterion position sizing, "
            "produce a final trading recommendation. "
            "Return ONLY a JSON object with keys: "
            "asset, action (BUY/SELL/NO_BET), position_size_pct (0-100), "
            "stop_loss_pct, take_profit_pct, risk_reward_ratio, rationale."
        )

    def _blend_signals(
        self,
        crowd_pred: dict,
        model_pred: dict,
    ) -> dict:
        """
        Blend crowd and model predictions into a single confidence score.

        Args:
            crowd_pred: {"asset": ..., "direction": ..., "confidence": ...}
            model_pred: same shape

        Returns:
            {"asset": ..., "direction": ..., "confidence": ...}
        """
        asset = crowd_pred.get("asset") or model_pred.get("asset")

        # Normalise confidence to UP probability
        def up_prob(pred: dict) -> float:
            c = float(pred.get("confidence", 0.5))
            return c if pred.get("direction") == "UP" else (1.0 - c)

        crowd_up = up_prob(crowd_pred)
        model_up = up_prob(model_pred)

        blended_up = CROWD_WEIGHT * crowd_up + MODEL_WEIGHT * model_up
        direction = "UP" if blended_up >= 0.5 else "DOWN"
        confidence = blended_up if direction == "UP" else (1.0 - blended_up)

        logger.debug(
            f"[{self.name}] {asset} | crowd_up={crowd_up:.3f} model_up={model_up:.3f} "
            f"blended_up={blended_up:.3f} => {direction} @ {confidence:.3f}"
        )

        return {"asset": asset, "direction": direction, "confidence": confidence}

    def run(
        self,
        crowd_predictions: list[dict] | None = None,
        model_predictions: list[dict] | None = None,
        bankroll: float = 1000.0,
        **kwargs,
    ) -> dict:
        """
        Compute position sizes and trade recommendations.

        Args:
            crowd_predictions: From PredictionSearchAgent.
            model_predictions: From KronosPredictionAgent.
            bankroll:          Total capital available.

        Returns:
            {"recommendations": [...], "total_risk_pct": ...}
        """
        logger.info(f"[{self.name}] Computing risk-managed recommendations...")

        crowd_map = {p["asset"]: p for p in (crowd_predictions or [])}
        model_map  = {p["asset"]: p for p in (model_predictions or [])}

        blended = []
        for asset in ASSETS:
            crowd_p = crowd_map.get(asset, {"asset": asset, "direction": "UNKNOWN", "confidence": 0.5})
            model_p = model_map.get(asset, {"asset": asset, "direction": "UNKNOWN", "confidence": 0.5})
            blended.append(self._blend_signals(crowd_p, model_p))

        sized = multi_asset_kelly(blended, bankroll=bankroll)

        recommendations = []
        for s in sized:
            asset = s["asset"]
            prompt = (
                f"Asset: {asset}\n"
                f"Blended direction: {s['direction']}\n"
                f"Blended confidence: {s['confidence']:.4f}\n"
                f"Kelly action: {s['action']}\n"
                f"Kelly capped fraction: {s['capped_fraction']:.4f}\n"
                f"Suggested position size: ${s['position_size']:.2f} "
                f"(bankroll=${bankroll/len(ASSETS):.2f})\n"
                f"Max Kelly fraction cap: {MAX_KELLY_FRACTION}\n\n"
                "Generate a risk-managed trading recommendation. Return ONLY valid JSON."
            )
            raw = self._llm(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                rec = json.loads(raw)
                rec["asset"] = asset
                rec["kelly_fraction"] = s["capped_fraction"]
                rec["position_size_usd"] = s["position_size"]
            except Exception:
                rec = {
                    "asset": asset,
                    "action": s["action"],
                    "position_size_pct": round(s["capped_fraction"] * 100, 2),
                    "stop_loss_pct": 1.0,
                    "take_profit_pct": 2.0,
                    "risk_reward_ratio": 2.0,
                    "rationale": "Kelly sizing applied (LLM parse failed)",
                    "kelly_fraction": s["capped_fraction"],
                    "position_size_usd": s["position_size"],
                }

            recommendations.append(rec)
            logger.info(
                f"[{self.name}] {asset} | action={rec.get('action')} "
                f"size=${rec.get('position_size_usd', 0):.2f}"
            )

        total_risk_pct = sum(
            r.get("kelly_fraction", 0) * 100 for r in recommendations
        )

        return {
            "recommendations": recommendations,
            "total_risk_pct": round(total_risk_pct, 2),
            "bankroll": bankroll,
        }

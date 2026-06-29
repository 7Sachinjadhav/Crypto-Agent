"""
Agent 3 — Prediction Agent
Predicts next 5-min up/down move using:
  - Technical indicator signals (Kronos-inspired rule engine)
  - LLM synthesis of multi-timeframe signals
  - Ensemble voting from 1min n+5 → 5min n+1 cross-validation

Reference: https://github.com/shiyu-coder/Kronos
"""
import json
import numpy as np
import pandas as pd
import httpx
from loguru import logger
from utils.config import config


# ---------------------------------------------------------------------------
# Kronos-inspired signal engine (rule-based, mirrors Kronos logic)
# ---------------------------------------------------------------------------

def _rsi_signal(rsi: float) -> tuple[str, float]:
    if rsi < 30:
        return "UP", 0.75
    elif rsi > 70:
        return "DOWN", 0.75
    elif rsi < 45:
        return "UP", 0.55
    elif rsi > 55:
        return "DOWN", 0.55
    return "NEUTRAL", 0.5


def _macd_signal(macd_hist: float, prev_macd_hist: float) -> tuple[str, float]:
    if macd_hist > 0 and prev_macd_hist <= 0:
        return "UP", 0.70  # Bullish crossover
    elif macd_hist < 0 and prev_macd_hist >= 0:
        return "DOWN", 0.70  # Bearish crossover
    elif macd_hist > 0:
        return "UP", 0.55
    elif macd_hist < 0:
        return "DOWN", 0.55
    return "NEUTRAL", 0.5


def _bb_signal(bb_pct: float, returns: float) -> tuple[str, float]:
    if bb_pct < 0.05:
        return "UP", 0.65  # Price at lower band → bounce
    elif bb_pct > 0.95:
        return "DOWN", 0.65  # Price at upper band → reversal
    return "NEUTRAL", 0.5


def _ema_signal(ema9: float, ema21: float, close: float) -> tuple[str, float]:
    if ema9 > ema21 and close > ema9:
        return "UP", 0.60
    elif ema9 < ema21 and close < ema9:
        return "DOWN", 0.60
    return "NEUTRAL", 0.5


def _volume_signal(vol_ratio: float, returns: float) -> tuple[str, float]:
    if vol_ratio > 1.5 and returns > 0:
        return "UP", 0.65  # High volume upward move → continuation
    elif vol_ratio > 1.5 and returns < 0:
        return "DOWN", 0.65
    return "NEUTRAL", 0.5


def _stoch_signal(stoch_k: float, stoch_d: float) -> tuple[str, float]:
    if stoch_k < 20 and stoch_k > stoch_d:
        return "UP", 0.65
    elif stoch_k > 80 and stoch_k < stoch_d:
        return "DOWN", 0.65
    return "NEUTRAL", 0.5


def _kronos_ensemble(df: pd.DataFrame) -> dict:
    """
    Kronos-inspired multi-signal ensemble predictor.
    Aggregates 6 technical signals via weighted voting.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signals = {
        "rsi": _rsi_signal(last["rsi"]),
        "macd": _macd_signal(last["macd_hist"], prev["macd_hist"]),
        "bb": _bb_signal(last["bb_pct"], last["returns"]),
        "ema": _ema_signal(last["ema_9"], last["ema_21"], last["close"]),
        "volume": _volume_signal(last["vol_ratio"], last["returns"]),
        "stoch": _stoch_signal(last["stoch_k"], last["stoch_d"]),
    }

    weights = {
        "rsi": 0.20,
        "macd": 0.25,
        "bb": 0.15,
        "ema": 0.20,
        "volume": 0.10,
        "stoch": 0.10,
    }

    score_up = sum(
        weights[k] * v[1] for k, v in signals.items() if v[0] == "UP"
    )
    score_down = sum(
        weights[k] * v[1] for k, v in signals.items() if v[0] == "DOWN"
    )
    score_neutral = 1 - score_up - score_down

    if score_up > score_down and score_up > 0.3:
        direction = "UP"
        confidence = min(score_up / (score_up + score_down + 0.01), 0.99)
    elif score_down > score_up and score_down > 0.3:
        direction = "DOWN"
        confidence = min(score_down / (score_up + score_down + 0.01), 0.99)
    else:
        direction = "NEUTRAL"
        confidence = 0.5

    return {
        "direction": direction,
        "confidence": round(confidence, 3),
        "signals": {k: {"signal": v[0], "conf": v[1]} for k, v in signals.items()},
        "score_up": round(score_up, 3),
        "score_down": round(score_down, 3),
    }


# ---------------------------------------------------------------------------
# Multi-timeframe arbitrage (1min n+5 → 5min n+1)
# ---------------------------------------------------------------------------

def _multi_timeframe_signal(df_5m: pd.DataFrame, df_1m: pd.DataFrame | None) -> dict:
    """
    Checks internal arbitrage: 1min n+5 signal vs 5min n+1 signal.
    Returns an alignment flag and combined confidence boost.
    """
    signal_5m = _kronos_ensemble(df_5m)

    if df_1m is None or df_1m.empty:
        return {
            "aligned": None,
            "signal_5m": signal_5m,
            "signal_1m": None,
            "combined_direction": signal_5m["direction"],
            "combined_confidence": signal_5m["confidence"],
        }

    signal_1m = _kronos_ensemble(df_1m)

    aligned = signal_5m["direction"] == signal_1m["direction"]
    # If signals align, boost confidence
    if aligned and signal_5m["direction"] != "NEUTRAL":
        combined_conf = min((signal_5m["confidence"] + signal_1m["confidence"]) / 2 + 0.05, 0.95)
    else:
        combined_conf = (signal_5m["confidence"] + signal_1m["confidence"]) / 2

    return {
        "aligned": aligned,
        "signal_5m": signal_5m,
        "signal_1m": signal_1m,
        "combined_direction": signal_5m["direction"],
        "combined_confidence": round(combined_conf, 3),
    }


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def _llm_predict(asset: str, data_summary: dict, kronos_result: dict) -> dict:
    """Use OpenRouter LLM to synthesize final prediction."""
    try:
        prompt = f"""
You are a professional quantitative crypto analyst.

Asset: {asset}
Timeframe: Next 5 minutes

Technical Data Summary:
{json.dumps(data_summary, indent=2)}

Kronos Signal Engine Result:
{json.dumps(kronos_result, indent=2)}

Based on this data, predict the next 5-minute price direction.
Respond ONLY with valid JSON:
{{
  "direction": "UP" | "DOWN" | "NEUTRAL",
  "confidence": <float 0-1>,
  "entry_rationale": "<brief reasoning>",
  "key_risks": ["<risk1>", "<risk2>"]
}}
"""
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/crowdwisdomtrading",
        }
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a quantitative crypto trading analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.1,
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)

    except Exception as e:
        logger.warning(f"[LLM Prediction] Failed: {e}. Using Kronos result.")
        return {
            "direction": kronos_result.get("combined_direction", "NEUTRAL"),
            "confidence": kronos_result.get("combined_confidence", 0.5),
            "entry_rationale": "LLM unavailable; using Kronos signal engine",
            "key_risks": ["LLM synthesis unavailable"],
        }


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def run_prediction_agent(
    asset: str,
    data_result: dict,
    data_result_1m: dict | None = None,
) -> dict:
    """
    Run the full prediction pipeline.

    Args:
        asset: 'BTC' or 'ETH'
        data_result: output from DataFetcherAgent (5m bars)
        data_result_1m: optional output from DataFetcherAgent (1m bars)

    Returns:
        dict with full prediction result
    """
    logger.info(f"[PredictionAgent] Running for {asset}")

    df_5m = data_result.get("df", pd.DataFrame())
    df_1m = data_result_1m.get("df", pd.DataFrame()) if data_result_1m else None
    summary = data_result.get("summary", {})

    if df_5m.empty:
        logger.error(f"[PredictionAgent] No data for {asset}")
        return {
            "asset": asset,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "error": "No data available",
        }

    # Multi-timeframe Kronos signal
    mtf = _multi_timeframe_signal(df_5m, df_1m)

    # LLM synthesis
    llm_result = _llm_predict(asset, summary, mtf)

    # Final ensemble: blend Kronos + LLM
    directions = [mtf["combined_direction"], llm_result.get("direction", "NEUTRAL")]
    confs = [mtf["combined_confidence"], llm_result.get("confidence", 0.5)]

    # Majority vote
    up_votes = directions.count("UP")
    down_votes = directions.count("DOWN")
    if up_votes > down_votes:
        final_direction = "UP"
        final_conf = sum(c for d, c in zip(directions, confs) if d == "UP") / max(up_votes, 1)
    elif down_votes > up_votes:
        final_direction = "DOWN"
        final_conf = sum(c for d, c in zip(directions, confs) if d == "DOWN") / max(down_votes, 1)
    else:
        final_direction = "NEUTRAL"
        final_conf = 0.5

    result = {
        "asset": asset,
        "direction": final_direction,
        "confidence": round(final_conf, 3),
        "kronos_signal": mtf,
        "llm_signal": llm_result,
        "entry_rationale": llm_result.get("entry_rationale", ""),
        "key_risks": llm_result.get("key_risks", []),
        "market_regime": summary.get("market_regime", "unknown"),
        "latest_price": summary.get("latest_close", 0),
    }

    logger.info(
        f"[PredictionAgent] {asset} → {final_direction} "
        f"({final_conf:.0%} confidence) | Regime: {result['market_regime']}"
    )
    return result

"""
Kelly Criterion position-sizing utility.

Formula:  f* = (b*p - q) / b
  where:
    p = probability of winning (UP prediction confidence)
    q = 1 - p  (probability of losing)
    b = net odds received on the wager (e.g. 1.0 for even-odds bet)

Reference:
  https://managebankroll.com/blog/polymarket-kelly-criterion-position-sizing
  https://mintlify.wiki/joicodev/polymarket-bot/risk/kelly-criterion
"""

from config.settings import MAX_KELLY_FRACTION
from utils.logger import logger


def kelly_fraction(
    win_probability: float,
    net_odds: float = 1.0,
    bankroll: float = 1.0,
) -> dict:
    """
    Calculate the Kelly fraction and suggested position size.

    Args:
        win_probability: Model's confidence that price goes UP (0–1).
        net_odds:        Net profit per unit risked (default 1.0 = even odds).
        bankroll:        Total available capital (used to compute absolute size).

    Returns:
        dict with keys:
            fraction      – raw Kelly fraction (may be negative = no bet)
            capped_fraction – fraction capped at MAX_KELLY_FRACTION
            position_size  – suggested dollar amount to risk
            action         – "BUY", "SELL", or "NO_BET"
    """
    if not (0.0 <= win_probability <= 1.0):
        raise ValueError(f"win_probability must be in [0,1], got {win_probability}")

    q = 1.0 - win_probability
    raw_f = (net_odds * win_probability - q) / net_odds

    logger.debug(
        f"Kelly | p={win_probability:.4f} q={q:.4f} b={net_odds} raw_f={raw_f:.4f}"
    )

    if raw_f <= 0:
        return {
            "fraction": raw_f,
            "capped_fraction": 0.0,
            "position_size": 0.0,
            "action": "NO_BET",
        }

    capped = min(raw_f, MAX_KELLY_FRACTION)
    position_size = round(capped * bankroll, 2)

    action = "BUY" if win_probability >= 0.5 else "SELL"

    logger.info(
        f"Kelly result | action={action} fraction={capped:.4f} size=${position_size}"
    )

    return {
        "fraction": raw_f,
        "capped_fraction": capped,
        "position_size": position_size,
        "action": action,
    }


def multi_asset_kelly(predictions: list[dict], bankroll: float = 1000.0) -> list[dict]:
    """
    Apply Kelly sizing to a list of asset predictions.

    Args:
        predictions: List of dicts with keys 'asset', 'direction', 'confidence'.
        bankroll:    Total capital to allocate across all positions.

    Returns:
        List of dicts enriched with Kelly sizing fields.
    """
    results = []
    per_asset_bankroll = bankroll / max(len(predictions), 1)

    for pred in predictions:
        confidence = pred.get("confidence", 0.5)
        # If direction is DOWN, flip probability for kelly (we're shorting)
        if pred.get("direction") == "DOWN":
            win_prob = 1.0 - confidence
        else:
            win_prob = confidence

        sizing = kelly_fraction(win_prob, bankroll=per_asset_bankroll)
        results.append({**pred, **sizing})

    return results

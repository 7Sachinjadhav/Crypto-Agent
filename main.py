"""
CrowdWisdomTrading – Crypto Predictions Agent
==============================================
Main entry point. Orchestrates the full Hermes agent pipeline:

  1. PredictionSearchAgent  – crowd wisdom (Polymarket + Kalshi)
  2. DataFetcherAgent       – OHLCV data via Apify / CoinGecko
  3. KronosPredictionAgent  – statistical UP/DOWN prediction
  4. RiskManagementAgent    – Kelly Criterion position sizing
  5. FeedbackAgent          – Agnes loop performance tracking

Run:
    python main.py                 # single cycle
    python main.py --loop          # continuous (runs every 5 min)
    python main.py --bankroll 5000 # custom bankroll

Author: CrowdWisdomTrading Intern Assessment
"""

import argparse
import time
import json
from datetime import datetime
from utils.logger import logger
from agents.prediction_search_agent import PredictionSearchAgent
from agents.data_fetcher_agent import DataFetcherAgent
from agents.kronos_prediction_agent import KronosPredictionAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.feedback_agent import FeedbackAgent
from config.settings import PREDICTION_INTERVAL_MIN


def run_cycle(bankroll: float = 1000.0) -> dict:
    """Execute one full prediction → risk management cycle."""
    cycle_start = datetime.utcnow()
    logger.info(f"{'='*60}")
    logger.info(f"CYCLE START: {cycle_start.isoformat()}")
    logger.info(f"{'='*60}")

    results = {"timestamp": cycle_start.isoformat()}

    # ── Agent 1: Crowd Wisdom Search ───────────────────────────────
    logger.info("▶ Agent 1: PredictionSearchAgent")
    try:
        search_agent = PredictionSearchAgent()
        search_result = search_agent.run()
        crowd_predictions = search_result.get("predictions", [])
        results["crowd_predictions"] = crowd_predictions
        logger.success(f"Agent 1 done | {len(crowd_predictions)} predictions")
    except Exception as exc:
        logger.error(f"Agent 1 failed: {exc}")
        crowd_predictions = []

    # ── Agent 2: Data Fetcher ───────────────────────────────────────
    logger.info("▶ Agent 2: DataFetcherAgent")
    try:
        data_agent = DataFetcherAgent()
        data_result = data_agent.run()
        asset_data = data_result.get("data", {})
        results["asset_data_summary"] = {
            k: {key: v[key] for key in ("latest_close", "change_24h_pct", "bars_fetched", "data_quality") if key in v}
            for k, v in asset_data.items()
        }
        logger.success(f"Agent 2 done | assets fetched: {list(asset_data.keys())}")
    except Exception as exc:
        logger.error(f"Agent 2 failed: {exc}")
        asset_data = {}

    # ── Agent 3: Kronos Prediction ─────────────────────────────────
    logger.info("▶ Agent 3: KronosPredictionAgent")
    try:
        kronos_agent = KronosPredictionAgent()
        kronos_result = kronos_agent.run(asset_data=asset_data)
        model_predictions = kronos_result.get("predictions", [])
        results["model_predictions"] = model_predictions
        logger.success(f"Agent 3 done | {len(model_predictions)} predictions")
    except Exception as exc:
        logger.error(f"Agent 3 failed: {exc}")
        model_predictions = []

    # ── Agent 4: Risk Management ────────────────────────────────────
    logger.info("▶ Agent 4: RiskManagementAgent")
    try:
        risk_agent = RiskManagementAgent()
        risk_result = risk_agent.run(
            crowd_predictions=crowd_predictions,
            model_predictions=model_predictions,
            bankroll=bankroll,
        )
        recommendations = risk_result.get("recommendations", [])
        results["recommendations"] = recommendations
        results["total_risk_pct"] = risk_result.get("total_risk_pct")
        logger.success(
            f"Agent 4 done | total_risk={risk_result.get('total_risk_pct')}%"
        )
    except Exception as exc:
        logger.error(f"Agent 4 failed: {exc}")
        recommendations = []

    # ── Agent 5: Feedback Loop ──────────────────────────────────────
    logger.info("▶ Agent 5: FeedbackAgent")
    try:
        all_preds = crowd_predictions + model_predictions
        feedback_agent = FeedbackAgent()
        feedback_result = feedback_agent.run(current_predictions=all_preds)
        results["feedback"] = feedback_result.get("feedback", {})
        logger.success("Agent 5 done")
    except Exception as exc:
        logger.error(f"Agent 5 failed: {exc}")

    # ── Summary ─────────────────────────────────────────────────────
    duration = (datetime.utcnow() - cycle_start).total_seconds()
    results["duration_seconds"] = round(duration, 2)

    logger.info(f"{'='*60}")
    logger.info("CYCLE SUMMARY")
    logger.info(f"{'='*60}")
    for rec in recommendations:
        logger.info(
            f"  {rec.get('asset'):>4} | {rec.get('action', 'N/A'):>6} | "
            f"${rec.get('position_size_usd', 0):>8.2f} | "
            f"Kelly={rec.get('kelly_fraction', 0):.3f}"
        )
    logger.info(f"Cycle completed in {duration:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading Crypto Predictions Agent"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=f"Run continuously every {PREDICTION_INTERVAL_MIN} minutes",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Total capital available for trading (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save cycle results as JSON",
    )
    args = parser.parse_args()

    if args.loop:
        logger.info(
            f"Starting continuous loop (every {PREDICTION_INTERVAL_MIN} min)..."
        )
        cycle_num = 0
        while True:
            cycle_num += 1
            logger.info(f"Loop cycle #{cycle_num}")
            result = run_cycle(bankroll=args.bankroll)
            if args.output:
                _save_result(result, args.output, cycle_num)
            logger.info(
                f"Sleeping {PREDICTION_INTERVAL_MIN * 60}s until next cycle..."
            )
            time.sleep(PREDICTION_INTERVAL_MIN * 60)
    else:
        result = run_cycle(bankroll=args.bankroll)
        if args.output:
            _save_result(result, args.output, 1)
        # Pretty-print to stdout
        print(json.dumps(result, indent=2, default=str))


def _save_result(result: dict, path: str, cycle: int) -> None:
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    base, ext = os.path.splitext(path)
    filename = f"{base}_cycle{cycle:04d}{ext or '.json'}"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    main()

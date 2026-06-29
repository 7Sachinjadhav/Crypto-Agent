"""
Agent 1 – Prediction Search Agent
===================================
Searches Polymarket AND Kalshi for crowd-sourced 5-minute price predictions
for Ethereum and Bitcoin.

Data flow:
  1. Query Polymarket public API for relevant markets.
  2. Query Kalshi public API for relevant events.
  3. Pass raw results to LLM to extract a structured UP/DOWN signal.
  4. Return a dict with crowd-wisdom predictions per asset.
"""

import requests
from agents.base_agent import BaseAgent
from utils.logger import logger
from config.settings import (
    ASSETS,
    POLYMARKET_ETH_KEYWORD,
    POLYMARKET_BTC_KEYWORD,
    KALSHI_ETH_KEYWORD,
    KALSHI_BTC_KEYWORD,
)


POLYMARKET_API = "https://clob.polymarket.com/markets"
KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2/events"


class PredictionSearchAgent(BaseAgent):
    name = "PredictionSearchAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a crypto market intelligence analyst. "
            "Given raw JSON data from prediction markets (Polymarket and Kalshi), "
            "extract the crowd-wisdom signal for each crypto asset. "
            "For each asset output ONLY a JSON object with keys: "
            "asset, direction (UP or DOWN), confidence (0.0-1.0), source, market_title. "
            "If no relevant market found, set direction=UNKNOWN and confidence=0.5."
        )

    def _fetch_polymarket(self, keyword: str) -> list[dict]:
        """Fetch markets from Polymarket CLOB API matching keyword."""
        try:
            resp = requests.get(
                POLYMARKET_API,
                params={"next_cursor": "", "limit": 20},
                timeout=10,
            )
            resp.raise_for_status()
            markets = resp.json().get("data", [])
            keyword_lower = keyword.lower()
            filtered = [
                m for m in markets
                if keyword_lower in m.get("question", "").lower()
                or keyword_lower in m.get("description", "").lower()
            ]
            logger.info(f"Polymarket: {len(filtered)} markets for '{keyword}'")
            return filtered[:3]
        except Exception as exc:
            logger.warning(f"Polymarket fetch failed: {exc}")
            return []

    def _fetch_kalshi(self, keyword: str) -> list[dict]:
        """Fetch events from Kalshi public API matching keyword."""
        try:
            resp = requests.get(
                KALSHI_API,
                params={"limit": 20, "status": "open"},
                headers={"Accept": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json().get("events", [])
            keyword_lower = keyword.lower()
            filtered = [
                e for e in events
                if keyword_lower in e.get("title", "").lower()
                or keyword_lower in e.get("sub_title", "").lower()
            ]
            logger.info(f"Kalshi: {len(filtered)} events for '{keyword}'")
            return filtered[:3]
        except Exception as exc:
            logger.warning(f"Kalshi fetch failed: {exc}")
            return []

    def _extract_signal(self, asset: str, poly_data: list, kalshi_data: list) -> dict:
        """Use LLM to parse crowd-wisdom signal from raw market data."""
        prompt = (
            f"Asset: {asset}\n\n"
            f"Polymarket markets:\n{poly_data}\n\n"
            f"Kalshi events:\n{kalshi_data}\n\n"
            "Extract the prediction signal. Return ONLY valid JSON."
        )
        raw = self._llm(prompt)
        import json, re
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            result = json.loads(raw)
            result["asset"] = asset
            return result
        except json.JSONDecodeError:
            logger.warning(f"LLM JSON parse failed for {asset}, using fallback.")
            return {
                "asset": asset,
                "direction": "UNKNOWN",
                "confidence": 0.5,
                "source": "parse_error",
                "market_title": "N/A",
            }

    def run(self, **kwargs) -> dict:
        """
        Run the prediction search for all configured assets.

        Returns:
            {"predictions": [{"asset": "ETH", "direction": "UP", ...}, ...]}
        """
        logger.info(f"[{self.name}] Starting prediction search...")
        keywords = {
            "ETH": (POLYMARKET_ETH_KEYWORD, KALSHI_ETH_KEYWORD),
            "BTC": (POLYMARKET_BTC_KEYWORD, KALSHI_BTC_KEYWORD),
        }

        predictions = []
        for asset in ASSETS:
            poly_kw, kalshi_kw = keywords[asset]
            poly_data  = self._fetch_polymarket(poly_kw)
            kalshi_data = self._fetch_kalshi(kalshi_kw)
            signal = self._extract_signal(asset, poly_data, kalshi_data)
            predictions.append(signal)
            logger.info(f"[{self.name}] {asset} signal: {signal}")

        return {"predictions": predictions}

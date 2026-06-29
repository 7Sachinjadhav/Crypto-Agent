"""
Agent 2 – Data Fetcher Agent
================================
Uses Apify to fetch the last N OHLCV bars for each crypto asset.

Apify actor used: `dtrungtin/yahoo-finance-scraper` (free tier, no auth needed
beyond APIFY_API_TOKEN).  Falls back to CoinGecko public API if Apify fails.

Returns a pandas DataFrame (serialised to list of dicts) with columns:
    timestamp, open, high, low, close, volume
"""

import requests
import pandas as pd
from apify_client import ApifyClient
from agents.base_agent import BaseAgent
from utils.logger import logger
from config.settings import APIFY_API_TOKEN, KRONOS_LOOKBACK_BARS, ASSETS


COINGECKO_IDS = {"ETH": "ethereum", "BTC": "bitcoin"}
# Yahoo Finance symbols for Apify
YAHOO_SYMBOLS = {"ETH": "ETH-USD", "BTC": "BTC-USD"}


class DataFetcherAgent(BaseAgent):
    name = "DataFetcherAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a data validation assistant. "
            "Given OHLCV candlestick data for a crypto asset, verify it looks correct "
            "and summarise basic statistics (latest close, 24h change %, volatility). "
            "Return ONLY a JSON object with keys: asset, latest_close, change_24h_pct, "
            "volatility, data_quality (good/bad), bars_fetched."
        )

    def _fetch_via_apify(self, symbol: str) -> list[dict]:
        """Fetch OHLCV bars via Apify Yahoo Finance scraper actor."""
        if not APIFY_API_TOKEN:
            logger.warning("APIFY_API_TOKEN not set – skipping Apify fetch.")
            return []
        try:
            client = ApifyClient(APIFY_API_TOKEN)
            run_input = {
                "symbols": [symbol],
                "period1": "3mo",   # 3 months of data
                "interval": "5m",   # 5-minute bars
            }
            logger.info(f"[DataFetcherAgent] Starting Apify actor for {symbol}...")
            run = client.actor("dtrungtin/yahoo-finance-scraper").call(
                run_input=run_input
            )
            items = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                items.append(item)
            logger.info(f"[DataFetcherAgent] Apify returned {len(items)} items for {symbol}")
            return items[-KRONOS_LOOKBACK_BARS:]
        except Exception as exc:
            logger.warning(f"Apify fetch failed for {symbol}: {exc}")
            return []

    def _fetch_via_coingecko(self, asset: str) -> list[dict]:
        """Fallback: fetch hourly OHLCV from CoinGecko public API (free, no key)."""
        cg_id = COINGECKO_IDS.get(asset)
        if not cg_id:
            return []
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
            resp = requests.get(url, params={"vs_currency": "usd", "days": "14"}, timeout=15)
            resp.raise_for_status()
            raw = resp.json()  # [[timestamp_ms, open, high, low, close], ...]
            bars = [
                {
                    "timestamp": r[0],
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": 0,
                }
                for r in raw
            ]
            logger.info(f"[DataFetcherAgent] CoinGecko returned {len(bars)} bars for {asset}")
            return bars[-KRONOS_LOOKBACK_BARS:]
        except Exception as exc:
            logger.warning(f"CoinGecko fetch failed for {asset}: {exc}")
            return []

    def _validate_with_llm(self, asset: str, bars: list[dict]) -> dict:
        """Ask LLM to validate and summarise the fetched data."""
        sample = bars[-10:] if len(bars) >= 10 else bars
        prompt = (
            f"Asset: {asset}\n"
            f"Total bars fetched: {len(bars)}\n"
            f"Last 10 bars (sample): {sample}\n\n"
            "Validate and summarise. Return ONLY valid JSON."
        )
        import json, re
        raw = self._llm(prompt)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            result = json.loads(raw)
            result["asset"] = asset
            result["bars"] = bars
            return result
        except Exception:
            return {
                "asset": asset,
                "latest_close": bars[-1].get("close", 0) if bars else 0,
                "change_24h_pct": 0.0,
                "volatility": 0.0,
                "data_quality": "unknown",
                "bars_fetched": len(bars),
                "bars": bars,
            }

    def run(self, **kwargs) -> dict:
        """
        Fetch last 1000 bars for each asset.

        Returns:
            {"data": {"ETH": {..., "bars": [...]}, "BTC": {..., "bars": [...]}}}
        """
        logger.info(f"[{self.name}] Starting data fetch...")
        result = {}

        for asset in ASSETS:
            symbol = YAHOO_SYMBOLS[asset]
            bars = self._fetch_via_apify(symbol)
            if not bars:
                bars = self._fetch_via_coingecko(asset)

            summary = self._validate_with_llm(asset, bars)
            result[asset] = summary
            logger.info(
                f"[{self.name}] {asset} | bars={summary.get('bars_fetched')} "
                f"close={summary.get('latest_close')} quality={summary.get('data_quality')}"
            )

        return {"data": result}

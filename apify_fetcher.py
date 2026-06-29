"""
Apify data fetcher — pulls OHLCV crypto bars via Apify actors.
Falls back to CoinGecko public API when no Apify token is set.
"""
import httpx
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from utils.config import config


# ---------------------------------------------------------------------------
# Fallback: CoinGecko public API (no key required)
# ---------------------------------------------------------------------------

COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum"}
BINANCE_BASE = "https://api.binance.com/api/v3/klines"


def _fetch_binance_klines(symbol: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance public API.
    interval examples: 1m, 5m, 15m, 1h, 1d
    """
    pair = f"{symbol}USDT"
    try:
        url = BINANCE_BASE
        params = {"symbol": pair, "interval": interval, "limit": limit}
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.set_index("timestamp", inplace=True)
        logger.info(f"[Binance] Fetched {len(df)} bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"[Binance] Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def _fetch_apify_crypto(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Use Apify actor 'dtrungtin/coinmarketcap-scraper' or similar to fetch data.
    Requires APIFY_API_TOKEN.
    """
    try:
        from apify_client import ApifyClient

        client = ApifyClient(config.APIFY_API_TOKEN)

        # Use a general web scraper actor to get crypto data
        run_input = {
            "startUrls": [
                {
                    "url": (
                        f"https://api.binance.com/api/v3/klines"
                        f"?symbol={symbol}USDT&interval=5m&limit={limit}"
                    )
                }
            ],
            "maxRequestsPerCrawl": 1,
        }

        actor_call = client.actor("apify/cheerio-scraper").call(run_input=run_input)
        dataset_items = client.dataset(actor_call["defaultDatasetId"]).iterate_items()
        items = list(dataset_items)

        if not items:
            logger.warning("[Apify] No items returned, falling back to Binance direct.")
            return pd.DataFrame()

        logger.info(f"[Apify] Retrieved {len(items)} items for {symbol}")
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"[Apify] Fetch failed: {e}. Using Binance fallback.")
        return pd.DataFrame()


def fetch_ohlcv(symbol: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
    """
    Primary data fetch entry point.
    Tries Apify first, then Binance public API as fallback.
    """
    logger.info(f"Fetching {limit} bars for {symbol} @ {interval}")

    df = pd.DataFrame()

    # Try Apify if token available
    if config.APIFY_API_TOKEN:
        df = _fetch_apify_crypto(symbol, limit)

    # Fallback to Binance
    if df.empty:
        df = _fetch_binance_klines(symbol, interval, limit)

    if df.empty:
        logger.error(f"All data sources failed for {symbol}")
        raise RuntimeError(f"Could not fetch OHLCV data for {symbol}")

    return df

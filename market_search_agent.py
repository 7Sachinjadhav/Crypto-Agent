"""
Agent 1 — Market Search Agent
Searches Polymarket and Kalshi for crypto prediction markets (ETH/BTC, next 5min).
Uses Hermes agent framework with OpenRouter LLM.
"""
import httpx
import json
from loguru import logger
from utils.config import config


# ---------------------------------------------------------------------------
# Polymarket helpers
# ---------------------------------------------------------------------------

def search_polymarket(keyword: str) -> list[dict]:
    """Search Polymarket CLOB for markets matching keyword."""
    try:
        url = f"{config.POLYMARKET_BASE_URL}/markets"
        params = {"active": "true", "closed": "false", "limit": 20}
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        markets = data.get("data", data) if isinstance(data, dict) else data
        filtered = [
            m for m in markets
            if keyword.lower() in str(m.get("question", "")).lower()
            or keyword.lower() in str(m.get("description", "")).lower()
        ]
        logger.info(f"[Polymarket] Found {len(filtered)} markets for '{keyword}'")
        return filtered[:5]

    except Exception as e:
        logger.error(f"[Polymarket] Search failed: {e}")
        return []


def get_polymarket_prices(market_id: str) -> dict:
    """Get current YES/NO prices for a Polymarket market."""
    try:
        url = f"{config.POLYMARKET_BASE_URL}/book"
        params = {"token_id": market_id}
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"[Polymarket] Price fetch failed for {market_id}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Kalshi helpers
# ---------------------------------------------------------------------------

def search_kalshi(keyword: str) -> list[dict]:
    """Search Kalshi for active crypto markets."""
    try:
        url = f"{config.KALSHI_BASE_URL}/markets"
        params = {
            "status": "open",
            "limit": 20,
            "series_ticker": keyword.upper(),
        }
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        markets = data.get("markets", [])
        logger.info(f"[Kalshi] Found {len(markets)} markets for '{keyword}'")
        return markets[:5]

    except Exception as e:
        logger.error(f"[Kalshi] Search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# LLM-powered synthesis via OpenRouter (Hermes-style agent)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call OpenRouter LLM to synthesize market crowd wisdom."""
    try:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/crowdwisdomtrading",
            "X-Title": "CrowdWisdomTrading",
        }
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a crypto market analyst. Analyze prediction market data "
                        "and give a concise directional signal: UP, DOWN, or NEUTRAL. "
                        "Always respond in JSON with keys: signal, confidence (0-1), reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.2,
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content
    except Exception as e:
        logger.error(f"[LLM] Call failed: {e}")
        return json.dumps({"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "LLM unavailable"})


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_market_search_agent(asset: str) -> dict:
    """
    Search Polymarket + Kalshi for next 5min crypto predictions.
    Returns aggregated crowd-wisdom signal.

    Args:
        asset: 'BTC' or 'ETH'

    Returns:
        dict with keys: asset, polymarket_markets, kalshi_markets, crowd_signal
    """
    logger.info(f"[MarketSearchAgent] Running for {asset}")

    full_name = config.ASSET_FULL_NAMES.get(asset, asset)

    # Fetch markets
    poly_markets = search_polymarket(full_name) or search_polymarket(asset)
    kalshi_markets = search_kalshi(asset)

    # Build LLM prompt
    prompt = f"""
Asset: {full_name} ({asset})
Timeframe: Next 5 minutes

Polymarket prediction markets found:
{json.dumps(poly_markets, indent=2)[:1500]}

Kalshi prediction markets found:
{json.dumps(kalshi_markets, indent=2)[:1500]}

Based on this crowd wisdom from prediction markets, what is the likely next 5-minute 
price direction for {asset}? Consider the implied probabilities.
Respond ONLY with valid JSON.
"""

    llm_response = _call_llm(prompt)

    try:
        crowd_signal = json.loads(llm_response)
    except json.JSONDecodeError:
        crowd_signal = {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": llm_response}

    result = {
        "asset": asset,
        "polymarket_markets": poly_markets,
        "kalshi_markets": kalshi_markets,
        "crowd_signal": crowd_signal,
    }

    logger.info(f"[MarketSearchAgent] {asset} crowd signal: {crowd_signal.get('signal')} "
                f"({crowd_signal.get('confidence', 0):.0%} confidence)")
    return result

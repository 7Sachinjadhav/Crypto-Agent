"""
Configuration loader for CrowdWisdomTrading Crypto Agent.
"""
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class Config:
    # LLM
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

    # Apify
    APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "")

    # Markets
    POLYMARKET_BASE_URL: str = os.getenv(
        "POLYMARKET_BASE_URL", "https://clob.polymarket.com"
    )
    KALSHI_BASE_URL: str = os.getenv(
        "KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2"
    )

    # Crypto assets
    ASSETS: list[str] = ["ETH", "BTC"]
    ASSET_FULL_NAMES: dict = {
        "ETH": "Ethereum",
        "BTC": "Bitcoin",
    }

    # Data
    BARS_LIMIT: int = 1000
    PREDICTION_INTERVAL: str = "5min"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/agent.log"

    @classmethod
    def validate(cls) -> bool:
        missing = []
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.APIFY_API_TOKEN:
            missing.append("APIFY_API_TOKEN")
        if missing:
            logger.warning(f"Missing env vars: {missing}. Some features may be limited.")
            return False
        return True


config = Config()

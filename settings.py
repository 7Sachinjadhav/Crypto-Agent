"""
Central configuration for CrowdWisdomTrading Crypto Agent.
Loads from .env file or environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct")

# Apify
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/crypto_agent.log")

# Trading assets
ASSETS = ["ETH", "BTC"]

# Prediction timeframes
PREDICTION_INTERVAL_MIN = 5
KRONOS_LOOKBACK_BARS = 1000

# Kelly Criterion risk cap
MAX_KELLY_FRACTION = 0.25

# Polymarket / Kalshi search keywords
POLYMARKET_ETH_KEYWORD = "Ethereum price 5 minutes"
POLYMARKET_BTC_KEYWORD = "Bitcoin price 5 minutes"
KALSHI_ETH_KEYWORD = "ETH next 5 min"
KALSHI_BTC_KEYWORD = "BTC next 5 min"

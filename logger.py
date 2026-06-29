"""
Centralised logging setup using loguru.
Import `logger` from here in every module.
"""

import sys
import os
from loguru import logger
from config.settings import LOG_LEVEL, LOG_FILE

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Remove default handler
logger.remove()

# Console handler
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File handler (rotating, 10 MB max, keep 7 days)
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

__all__ = ["logger"]

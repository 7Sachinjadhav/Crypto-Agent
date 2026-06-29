"""
Technical indicator calculations for prediction pipeline.
Uses the 'ta' library + custom implementations.
"""
import pandas as pd
import numpy as np
from loguru import logger


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a suite of technical indicators to OHLCV dataframe.
    Returns enriched dataframe.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Trend ---
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["sma_20"] = close.rolling(20).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Momentum ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Volatility ---
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # --- Volume ---
    df["vol_sma"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / df["vol_sma"].replace(0, np.nan)

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"] = obv

    # --- Price action ---
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["volatility_5"] = df["log_returns"].rolling(5).std()

    # Candle body / wick ratios
    df["body"] = (close - df["open"]).abs()
    df["upper_wick"] = high - pd.concat([close, df["open"]], axis=1).max(axis=1)
    df["lower_wick"] = pd.concat([close, df["open"]], axis=1).min(axis=1) - low

    df.dropna(inplace=True)
    logger.debug(f"Indicators computed. Shape: {df.shape}")
    return df


def get_market_regime(df: pd.DataFrame) -> str:
    """Classify current market regime: trending_up, trending_down, ranging."""
    if df.empty or len(df) < 50:
        return "unknown"

    last = df.iloc[-1]
    if last["ema_9"] > last["ema_21"] > last["ema_50"]:
        return "trending_up"
    elif last["ema_9"] < last["ema_21"] < last["ema_50"]:
        return "trending_down"
    else:
        return "ranging"

import numpy as np
import pandas as pd
from typing import List


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    for n in need:
        if n not in cols:
            raise ValueError(f"Missing OHLCV column '{n}' in dataframe. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "open": df[cols["open"]].astype(float).values,
        "high": df[cols["high"]].astype(float).values,
        "low": df[cols["low"]].astype(float).values,
        "close": df[cols["close"]].astype(float).values,
        "volume": df[cols["volume"]].astype(float).values,
    }, index=df.index)
    return out


def compute_rsi(close: np.ndarray, period: int = 14, eps: float = 1e-8) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    roll_down = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = roll_up / (roll_down + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def make_features(df: pd.DataFrame, feature_columns: List[str] | None = None) -> pd.DataFrame:
    base = _ensure_ohlcv(df)

    c = base["close"].values
    o = base["open"].values
    h = base["high"].values
    l = base["low"].values
    v = base["volume"].values

    logret = np.diff(np.log(c + 1e-12), prepend=np.log(c[0] + 1e-12))
    hl_range = (h - l) / (c + 1e-12)
    upper_wick = (h - np.maximum(o, c)) / (c + 1e-12)
    lower_wick = (np.minimum(o, c) - l) / (c + 1e-12)
    rsi = compute_rsi(c, period=14) / 100.0
    vol_z = (v - v.mean()) / (v.std() + 1e-8)

    # ATR normalized by close
    tr1 = h - l
    tr2 = np.abs(h - np.roll(c, 1))
    tr3 = np.abs(l - np.roll(c, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy() / (c + 1e-12)

    # Realized volatility of returns (rolling std)
    ret_std_14 = pd.Series(logret).rolling(14, min_periods=1).std().fillna(0.0).to_numpy()

    # Moving averages and MACD features normalized by close
    ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().to_numpy()
    ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().to_numpy()
    ema_fast_ratio = (ema_fast - c) / (c + 1e-12)
    ema_slow_ratio = (ema_slow - c) / (c + 1e-12)
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().to_numpy()
    macd_hist_norm = (macd - macd_signal) / (c + 1e-12)

    # Bollinger Bands width (20, 2) normalized by middle band
    mid = pd.Series(c).rolling(20, min_periods=1).mean()
    std = pd.Series(c).rolling(20, min_periods=1).std().fillna(0.0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    bb_width = ((upper - lower) / (mid.abs() + 1e-12)).to_numpy()

    feats = pd.DataFrame({
        "open": base["open"].values,
        "high": base["high"].values,
        "low": base["low"].values,
        "close": base["close"].values,
        "volume": base["volume"].values,
        "logret": logret,
        "hl_range": hl_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "rsi": rsi,
        "vol_z": vol_z,
        "atr": atr,
        "ret_std_14": ret_std_14,
        "ema_fast_ratio": ema_fast_ratio,
        "ema_slow_ratio": ema_slow_ratio,
        "macd_hist_norm": macd_hist_norm,
        "bb_width": bb_width,
    }, index=base.index)

    # If funding_rate present (from loader), add it and smoothed version
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float).reindex(feats.index).fillna(0.0)
        feats["funding_rate"] = fr.values
        feats["funding_rate_ma"] = fr.rolling(24, min_periods=1).mean().fillna(0.0).values

    # Sanitize any residual NaN/Inf from source data
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in feats.columns]
        # Create any missing columns as zeros to preserve model input layout
        for c in missing:
            feats[c] = 0.0
        # Reorder/select exactly the requested columns
        feats = feats.reindex(columns=feature_columns)
    return feats



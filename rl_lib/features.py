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
    }, index=base.index)

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in feats.columns]
        if missing:
            raise ValueError(f"Requested features missing: {missing}")
        feats = feats[feature_columns]
    return feats



import numpy as np
import pandas as pd
from typing import List, Optional


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    for n in need:
        if n not in cols:
            raise ValueError(f"Missing OHLCV column '{n}' in dataframe. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "open": df[cols["open"].lower() if cols.get("open") is None else cols["open"]].astype(float).values if False else df[cols["open"]].astype(float).values,
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


def make_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] | None = None,
    mode: Optional[str] = None,
    basic_lookback: int = 64,
) -> pd.DataFrame:
    base = _ensure_ohlcv(df)

    c = base["close"].values
    o = base["open"].values
    h = base["high"].values
    l = base["low"].values
    v = base["volume"].values

    mode_l = (mode or "full").lower()
    if mode_l == "basic":
        # Minimal, causal, standardized features: close_z, change, d_hl
        change = np.diff(np.log(c + 1e-12), prepend=np.log(c[0] + 1e-12))
        d_hl = (h - l) / (c + 1e-12)
        s = pd.Series(c)
        roll_mean = s.rolling(basic_lookback, min_periods=1).mean()
        roll_std = s.rolling(basic_lookback, min_periods=1).std().fillna(0.0)
        roll_std = roll_std.replace(0.0, 1e-8)
        close_z = ((s - roll_mean) / roll_std).to_numpy()
        feats = pd.DataFrame({
            "open": base["open"].values,
            "high": base["high"].values,
            "low": base["low"].values,
            "close": base["close"].values,
            "volume": base["volume"].values,
            "close_z": close_z,
            "change": change,
            "d_hl": d_hl,
        }, index=base.index)
    else:
        # Full feature set (causal). Removed MACD/Bollinger for consistency.
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
        }, index=base.index)

    # Removed funding_rate usage to keep features consistent across datasets

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



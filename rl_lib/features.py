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


def compute_hurst_proxy_from_returns(
    logret: np.ndarray,
    var_lookback: int = 256,
    scales: Optional[List[int]] | None = None,
) -> np.ndarray:
    """Causal, efficient Hurst exponent proxy via multiscale EWM std slope.

    Approximates H by regressing log(std of aggregated returns over scale m) on log(m).
    Uses EWM std for robustness and causality. Clipped to [0, 1].
    """
    if scales is None:
        scales = [1, 4, 16]
    s = pd.Series(logret)
    log_std_list: List[np.ndarray] = []
    for m in scales:
        # Aggregate returns over scale m (causal rolling sum)
        agg = s.rolling(m, min_periods=1).sum()
        std_m = agg.ewm(span=max(2, int(var_lookback)), adjust=False).std(bias=False)
        log_std_list.append(np.log(std_m.to_numpy() + 1e-12))
    log_ms = np.log(np.asarray(scales, dtype=float))
    xc = log_ms - log_ms.mean()
    denom = float(np.sum(xc * xc) + 1e-12)
    L = np.vstack(log_std_list).T  # (T, S)
    slope = (L @ xc) / denom
    hurst = np.clip(slope, 0.0, 1.0)
    return hurst


def compute_tail_alpha_proxy(
    logret: np.ndarray,
    window: int = 256,
    q1: float = 0.98,
    q2: float = 0.995,
) -> np.ndarray:
    """Causal tail index proxy via high-quantile ratio method.

    For a power-law tail, Q(1-p) ~ C p^{-1/alpha}. Using two tail quantiles q1<q2:
        alpha ≈ log((1-q2)/(1-q1)) / log(Q(q2)/Q(q1))
    We apply this to absolute log returns over a rolling window.
    """
    abs_r = np.abs(logret)
    s = pd.Series(abs_r)
    q1_series = s.rolling(max(10, int(window)), min_periods=10).quantile(q1)
    q2_series = s.rolling(max(10, int(window)), min_periods=10).quantile(q2)
    ratio = (q2_series.to_numpy() + 1e-12) / (q1_series.to_numpy() + 1e-12)
    c = np.log((1.0 - q2) / (1.0 - q1) + 1e-12)
    alpha = c / np.log(ratio + 1e-12)
    alpha = np.clip(alpha, 0.5, 10.0)
    return alpha


def make_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] | None = None,
    mode: Optional[str] = None,
    basic_lookback: int = 64,
    extra_timeframes: Optional[List[str]] = None,
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
        # Add minimal volatility estimate for gating
        ret_std_14 = pd.Series(change).rolling(14, min_periods=1).std().fillna(0.0).to_numpy()
        # Hurst and tail alpha proxies from returns
        hurst = compute_hurst_proxy_from_returns(change, var_lookback=max(64, basic_lookback * 2))
        hurst = np.nan_to_num(hurst, nan=0.5, posinf=1.0, neginf=0.0)
        tail_alpha = compute_tail_alpha_proxy(change, window=max(128, basic_lookback * 2))
        tail_alpha = np.nan_to_num(tail_alpha, nan=2.0, posinf=10.0, neginf=0.5)
        # Risk gate: shrink when vol high or tail_alpha small
        vol_median = pd.Series(ret_std_14).rolling(200, min_periods=1).median().to_numpy()
        vol_ratio = ret_std_14 / (vol_median + 1e-12)
        vol_gate = 1.0 / (1.0 + np.maximum(0.0, vol_ratio))
        alpha_gate = np.clip((tail_alpha - 1.5) / (3.5 - 1.5), 0.0, 1.0)
        risk_gate = np.clip(alpha_gate * vol_gate, 0.1, 1.0)
        feats = pd.DataFrame({
            "open": base["open"].values,
            "high": base["high"].values,
            "low": base["low"].values,
            "close": base["close"].values,
            "volume": base["volume"].values,
            "close_z": close_z,
            "change": change,
            "d_hl": d_hl,
            "ret_std_14": ret_std_14,
            "hurst": hurst,
            "tail_alpha": tail_alpha,
            "risk_gate": risk_gate,
        }, index=base.index)
    else:
        # Full feature set (causal). Keep MACD/EMA; remove Bollinger and funding.
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

        # Hurst and tail alpha proxies
        hurst = compute_hurst_proxy_from_returns(logret, var_lookback=256)
        hurst = np.nan_to_num(hurst, nan=0.5, posinf=1.0, neginf=0.0)
        tail_alpha = compute_tail_alpha_proxy(logret, window=256)
        tail_alpha = np.nan_to_num(tail_alpha, nan=2.0, posinf=10.0, neginf=0.5)

        # Risk gate: reduce effective exposure when tails are heavy or vol elevated
        vol_median = pd.Series(ret_std_14).rolling(200, min_periods=1).median().to_numpy()
        vol_ratio = ret_std_14 / (vol_median + 1e-12)
        vol_gate = 1.0 / (1.0 + np.maximum(0.0, vol_ratio))
        alpha_gate = np.clip((tail_alpha - 1.5) / (3.5 - 1.5), 0.0, 1.0)
        risk_gate = np.clip(alpha_gate * vol_gate, 0.1, 1.0)

        # EMA ratios and MACD histogram normalized by close
        ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().to_numpy()
        ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().to_numpy()
        ema_fast_ratio = (ema_fast - c) / (c + 1e-12)
        ema_slow_ratio = (ema_slow - c) / (c + 1e-12)
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().to_numpy()
        macd_hist_norm = (macd - macd_signal) / (c + 1e-12)

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
            "hurst": hurst,
            "tail_alpha": tail_alpha,
            "risk_gate": risk_gate,
            "ema_fast_ratio": ema_fast_ratio,
            "ema_slow_ratio": ema_slow_ratio,
            "macd_hist_norm": macd_hist_norm,
        }, index=base.index)

    # Add higher timeframe features (resampled and forward-filled) if requested
    if extra_timeframes:
        if isinstance(base.index, pd.DatetimeIndex):
            for tf in extra_timeframes:
                try:
                    tf_str = str(tf).upper()
                    agg = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    res = base.resample(tf_str).agg(agg).dropna(how="all")
                    # Compute the same feature set on HTF
                    rc = res["close"].values
                    ro = res["open"].values
                    rh = res["high"].values
                    rl = res["low"].values
                    rv = res["volume"].values

                    if mode_l == "basic":
                        change_tf = np.diff(np.log(rc + 1e-12), prepend=np.log(rc[0] + 1e-12))
                        d_hl_tf = (rh - rl) / (rc + 1e-12)
                        s_tf = pd.Series(rc, index=res.index)
                        roll_mean_tf = s_tf.rolling(basic_lookback, min_periods=1).mean()
                        roll_std_tf = s_tf.rolling(basic_lookback, min_periods=1).std().fillna(0.0)
                        roll_std_tf = roll_std_tf.replace(0.0, 1e-8)
                        close_z_tf = ((s_tf - roll_mean_tf) / roll_std_tf)
                        feats_tf = pd.DataFrame({
                            f"{tf}_close_z": close_z_tf,
                            f"{tf}_change": change_tf,
                            f"{tf}_d_hl": d_hl_tf,
                        }, index=res.index)
                    else:
                        logret_tf = np.diff(np.log(rc + 1e-12), prepend=np.log(rc[0] + 1e-12))
                        hl_range_tf = (rh - rl) / (rc + 1e-12)
                        upper_wick_tf = (rh - np.maximum(ro, rc)) / (rc + 1e-12)
                        lower_wick_tf = (np.minimum(ro, rc) - rl) / (rc + 1e-12)
                        rsi_tf = compute_rsi(rc, period=14) / 100.0
                        vol_z_tf = (rv - rv.mean()) / (rv.std() + 1e-8)
                        tr1_tf = rh - rl
                        tr2_tf = np.abs(rh - np.roll(rc, 1))
                        tr3_tf = np.abs(rl - np.roll(rc, 1))
                        tr_tf = np.maximum.reduce([tr1_tf, tr2_tf, tr3_tf])
                        tr_tf[0] = rh[0] - rl[0]
                        atr_tf = pd.Series(tr_tf, index=res.index).ewm(alpha=1 / 14, adjust=False).mean().to_numpy() / (rc + 1e-12)
                        ret_std_14_tf = pd.Series(logret_tf, index=res.index).rolling(14, min_periods=1).std().fillna(0.0).to_numpy()
                        ema_fast_tf = pd.Series(rc, index=res.index).ewm(span=12, adjust=False).mean().to_numpy()
                        ema_slow_tf = pd.Series(rc, index=res.index).ewm(span=26, adjust=False).mean().to_numpy()
                        ema_fast_ratio_tf = (ema_fast_tf - rc) / (rc + 1e-12)
                        ema_slow_ratio_tf = (ema_slow_tf - rc) / (rc + 1e-12)
                        macd_tf = ema_fast_tf - ema_slow_tf
                        macd_signal_tf = pd.Series(macd_tf, index=res.index).ewm(span=9, adjust=False).mean().to_numpy()
                        macd_hist_norm_tf = (macd_tf - macd_signal_tf) / (rc + 1e-12)
                        feats_tf = pd.DataFrame({
                            f"{tf}_logret": logret_tf,
                            f"{tf}_hl_range": hl_range_tf,
                            f"{tf}_upper_wick": upper_wick_tf,
                            f"{tf}_lower_wick": lower_wick_tf,
                            f"{tf}_rsi": rsi_tf,
                            f"{tf}_vol_z": vol_z_tf,
                            f"{tf}_atr": atr_tf,
                            f"{tf}_ret_std_14": ret_std_14_tf,
                            f"{tf}_ema_fast_ratio": ema_fast_ratio_tf,
                            f"{tf}_ema_slow_ratio": ema_slow_ratio_tf,
                            f"{tf}_macd_hist_norm": macd_hist_norm_tf,
                        }, index=res.index)
                    # Align to base index causally
                    feats_tf_aligned = feats_tf.reindex(base.index).ffill().fillna(0.0)
                    feats = feats.join(feats_tf_aligned, how="left")
                except Exception:
                    # If resampling fails, skip this timeframe silently
                    continue
        else:
            # Non-datetime index: ignore extra timeframes to avoid misalignment
            pass

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



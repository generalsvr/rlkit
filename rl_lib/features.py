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


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Wilder's ADX approximation using EWM for smoothing."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr1 = (h - l).to_numpy()
    tr2 = (h - c.shift(1)).abs().to_numpy()
    tr3 = (l - c.shift(1)).abs().to_numpy()
    tr = np.maximum.reduce([
        np.nan_to_num(tr1, nan=0.0),
        np.nan_to_num(tr2, nan=0.0),
        np.nan_to_num(tr3, nan=0.0),
    ])
    # Wilder smoothing via EWM alpha = 1/period
    tr_s = pd.Series(tr).ewm(alpha=1 / max(1, period), adjust=False).mean().to_numpy()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=1 / max(1, period), adjust=False).mean().to_numpy()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=1 / max(1, period), adjust=False).mean().to_numpy()
    plus_di = 100.0 * (plus_dm_s / (tr_s + 1e-12))
    minus_di = 100.0 * (minus_dm_s / (tr_s + 1e-12))
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx = pd.Series(dx).ewm(alpha=1 / max(1, period), adjust=False).mean().fillna(0.0).to_numpy()
    # Normalize to [0,1] for ML friendliness
    return np.clip(adx / 100.0, 0.0, 1.0)


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
        # Full/All feature set (causal). Includes trend/momentum, volatility, volume, microstructure, stats, risk.
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

        # New: Lagged returns (t-1..t-5) and return autocorrelation (lag-1) over window
        s_lr = pd.Series(logret)
        logret_lag_1 = s_lr.shift(1).fillna(0.0).to_numpy()
        logret_lag_2 = s_lr.shift(2).fillna(0.0).to_numpy()
        logret_lag_3 = s_lr.shift(3).fillna(0.0).to_numpy()
        logret_lag_4 = s_lr.shift(4).fillna(0.0).to_numpy()
        logret_lag_5 = s_lr.shift(5).fillna(0.0).to_numpy()
        ret_autocorr_64 = s_lr.rolling(64, min_periods=20).corr(s_lr.shift(1)).fillna(0.0).to_numpy()

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

        # EMA ratios
        ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().to_numpy()
        ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().to_numpy()
        ema_fast_ratio = (ema_fast - c) / (c + 1e-12)
        ema_slow_ratio = (ema_slow - c) / (c + 1e-12)

        # Trend & Momentum additions
        sma_fast = pd.Series(c).rolling(20, min_periods=1).mean().to_numpy()
        sma_slow = pd.Series(c).rolling(50, min_periods=1).mean().to_numpy()
        sma_ratio = (sma_fast - sma_slow) / (sma_slow + 1e-12)
        # EMA slope difference (angle proxy)
        ema_fast_slope = np.diff(ema_fast, prepend=ema_fast[0]) / (c + 1e-12)
        ema_slow_slope = np.diff(ema_slow, prepend=ema_slow[0]) / (c + 1e-12)
        ema_cross_angle = ema_fast_slope - ema_slow_slope
        # Linear regression slope over 90 bars (normalized by price)
        try:
            import numpy as _np
            idx = _np.arange(len(c))
            def _lr_slope(series: _np.ndarray, win: int = 90) -> _np.ndarray:
                out = _np.zeros_like(series)
                for i in range(len(series)):
                    j0 = max(0, i - win + 1)
                    x = idx[j0:i+1]
                    y = series[j0:i+1]
                    if x.size >= 3:
                        x_mean = x.mean(); y_mean = y.mean()
                        num = _np.sum((x - x_mean) * (y - y_mean))
                        den = _np.sum((x - x_mean) ** 2) + 1e-12
                        out[i] = num / den
                    else:
                        out[i] = 0.0
                return out
            lr_slope_90 = _lr_slope(c) / (c + 1e-12)
        except Exception:
            lr_slope_90 = np.zeros_like(c)
        # MACD
        macd_line = (ema_fast - ema_slow) / (c + 1e-12)
        macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().to_numpy()
        # Deceleration/curvature
        try:
            d_lr_slope_90_60 = pd.Series(lr_slope_90).diff(60).fillna(0.0).to_numpy()
        except Exception:
            d_lr_slope_90_60 = np.zeros_like(lr_slope_90)
        try:
            ema_slow_curvature = np.diff(ema_slow, n=2, prepend=ema_slow[0], append=ema_slow[-1]) / (c + 1e-12)
        except Exception:
            ema_slow_curvature = np.zeros_like(c)
        # Stochastic Oscillator
        hh = pd.Series(h).rolling(14, min_periods=1).max().to_numpy()
        ll = pd.Series(l).rolling(14, min_periods=1).min().to_numpy()
        stoch_k = np.clip((c - ll) / (hh - ll + 1e-12), 0.0, 1.0)
        stoch_d = pd.Series(stoch_k).rolling(3, min_periods=1).mean().to_numpy()
        # ROC
        roc_3 = (c / (np.roll(c, 3) + 1e-12)) - 1.0
        roc_5 = (c / (np.roll(c, 5) + 1e-12)) - 1.0
        roc_10 = (c / (np.roll(c, 10) + 1e-12)) - 1.0

        # Volatility & Risk additions
        bb_mid = pd.Series(c).rolling(20, min_periods=1).mean()
        bb_std = pd.Series(c).rolling(20, min_periods=1).std().fillna(0.0)
        bb_width = (2.0 * bb_std).to_numpy() / (bb_mid.to_numpy() + 1e-12)
        donch_w = (pd.Series(h).rolling(20, min_periods=1).max() - pd.Series(l).rolling(20, min_periods=1).min()).to_numpy() / (c + 1e-12)
        true_range_pct = atr
        # Vol-of-Vol (rolling std of ATR and realized vol)
        vol_of_vol_atr_64 = pd.Series(atr).rolling(64, min_periods=14).std().fillna(0.0).to_numpy()
        vol_of_vol_retstd_64 = pd.Series(ret_std_14).rolling(64, min_periods=14).std().fillna(0.0).to_numpy()
        # Skewness of returns (rolling 30)
        def _skew(arr: np.ndarray) -> float:
            m = float(np.mean(arr))
            sd = float(np.std(arr))
            if sd == 0.0:
                return 0.0
            return float(np.mean(((arr - m) / (sd + 1e-12)) ** 3))
        vol_skewness = pd.Series(logret).rolling(30, min_periods=10).apply(_skew, raw=True).fillna(0.0).to_numpy()
        # Volatility regime (z-score of ret_std_14 over rolling 200)
        vol_mean = pd.Series(ret_std_14).rolling(200, min_periods=20).mean().fillna(0.0).to_numpy()
        vol_std = pd.Series(ret_std_14).rolling(200, min_periods=20).std().fillna(1e-8).to_numpy()
        volatility_regime = (ret_std_14 - vol_mean) / (vol_std + 1e-8)

        # Volume / Participation additions
        obv = np.cumsum(np.sign(logret) * v)
        obv_z = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)
        volume_delta = np.sign(logret) * v
        volume_delta = (volume_delta - pd.Series(volume_delta).rolling(50, min_periods=10).mean().fillna(0.0)) / (pd.Series(volume_delta).rolling(50, min_periods=10).std().fillna(1e-8))
        typical_price = (h + l + c) / 3.0
        vwap_num = pd.Series(v * typical_price).rolling(20, min_periods=1).sum()
        vwap_den = pd.Series(v).rolling(20, min_periods=1).sum().replace(0.0, 1e-12)
        vwap = (vwap_num / vwap_den).to_numpy()
        vwap_ratio = (c - vwap) / (c + 1e-12)
        # Volume/Range ratio and Amihud illiquidity
        vol_range_ratio = v / (h - l + 1e-12)
        amihud_illiquidity = np.abs(logret) / (v + 1e-12)

        # Price shape / Microstructure additions
        body = np.abs(c - o)
        rng = (h - l) + 1e-12
        candle_body_frac = body / rng
        body_range_ratio_signed = (c - o) / (h - l + 1e-12)
        hl_asymmetry = (h - c) / (c - l + 1e-12)
        upper_shadow_frac = (h - np.maximum(o, c)) / rng
        lower_shadow_frac = (np.minimum(o, c) - l) / rng
        # Trend persistence (signed streak length, clipped)
        sgn = np.sign(logret)
        streak = np.zeros_like(sgn)
        for i in range(1, len(sgn)):
            if sgn[i] != 0 and sgn[i] == sgn[i-1]:
                streak[i] = streak[i-1] + sgn[i]
            else:
                streak[i] = sgn[i]
        candle_trend_persistence = np.tanh(streak / 10.0)
        # Directional change count (sign flips over last N bars)
        flips = (pd.Series(np.sign(logret)) != pd.Series(np.sign(logret)).shift(1)).astype(int)
        dir_change_count_20 = flips.rolling(20, min_periods=1).sum().fillna(0.0).to_numpy()
        # Kurtosis rolling (100)
        try:
            kurtosis_rolling = pd.Series(logret).rolling(100, min_periods=20).kurt().fillna(0.0).to_numpy()
        except Exception:
            kurtosis_rolling = np.zeros_like(logret)

        # Statistical / Fractal
        # Simple DFA exponent approximation
        def _dfa_exponent(series: np.ndarray, window: int = 64) -> np.ndarray:
            out = np.zeros_like(series)
            scales = [4, 8, 16, 32]
            for i in range(len(series)):
                j0 = max(0, i - window + 1)
                y = series[j0:i+1]
                if y.size < max(scales) + 2:
                    out[i] = 0.5
                    continue
                rms = []
                for m in scales:
                    segs = []
                    for k in range(m, y.size + 1, m):
                        seg = y[k-m:k]
                        x = np.arange(m)
                        # Detrend
                        xm = x.mean(); ym = seg.mean()
                        num = np.sum((x - xm) * (seg - ym))
                        den = np.sum((x - xm) ** 2) + 1e-12
                        a = num / den
                        b = ym - a * xm
                        detr = seg - (a * x + b)
                        segs.append(np.sqrt(np.mean(detr ** 2)))
                    if segs:
                        rms.append(np.mean(segs))
                if len(rms) >= 2:
                    xs = np.log(np.array(scales[:len(rms)]) + 1e-12)
                    ys = np.log(np.array(rms) + 1e-12)
                    xm = xs.mean(); ym = ys.mean()
                    num = np.sum((xs - xm) * (ys - ym))
                    den = np.sum((xs - xm) ** 2) + 1e-12
                    out[i] = num / den
                else:
                    out[i] = 0.5
            return np.clip(out, 0.0, 2.0)
        dfa_exponent = _dfa_exponent(logret, window=64)

        # Entropy of returns (Shannon, normalized)
        def _entropy(arr: np.ndarray, bins: int = 20) -> float:
            if arr.size < 10:
                return 0.0
            hist, _ = np.histogram(arr, bins=bins)
            p = hist.astype(float) / (np.sum(hist) + 1e-12)
            p = p[p > 0]
            H = -np.sum(p * np.log(p + 1e-12))
            Hn = H / (np.log(bins) + 1e-12)
            return float(np.clip(Hn, 0.0, 1.0))
        entropy_return = pd.Series(logret).rolling(64, min_periods=20).apply(_entropy, raw=True).fillna(0.0).to_numpy()

        # Risk & Trade Management
        # Rolling drawdown and z-score against vol
        cummax = np.maximum.accumulate(c)
        drawdown = (cummax - c) / (cummax + 1e-12)
        dd_std = pd.Series(logret).rolling(64, min_periods=20).std().replace(0.0, 1e-8).to_numpy()
        drawdown_z = drawdown / dd_std
        # Expected shortfall (ES) of returns at alpha over 128 window
        def _es_alpha(arr: np.ndarray, alpha: float = 0.05) -> float:
            neg = arr[arr < 0.0]
            if neg.size == 0:
                return 0.0
            q = np.quantile(neg, alpha)
            tail = neg[neg <= q]
            if tail.size == 0:
                return 0.0
            return float(np.abs(np.mean(tail)))
        expected_shortfall = pd.Series(logret).rolling(128, min_periods=32).apply(lambda a: _es_alpha(a, 0.05), raw=True).fillna(0.0).to_numpy()
        es_trend_20 = pd.Series(expected_shortfall).ewm(span=20, adjust=False).mean() - pd.Series(expected_shortfall).ewm(span=60, adjust=False).mean()
        es_trend_20 = es_trend_20.fillna(0.0).to_numpy()

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
            # New trend/momentum
            "sma_ratio": sma_ratio,
            "ema_cross_angle": ema_cross_angle,
            "lr_slope_90": lr_slope_90,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "d_lr_slope_90_60": d_lr_slope_90_60,
            "ema_slow_curvature": ema_slow_curvature,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "roc_10": roc_10,
            # New short-horizon cumulative returns and lagged returns
            "roc_3": roc_3,
            "roc_5": roc_5,
            "logret_lag_1": logret_lag_1,
            "logret_lag_2": logret_lag_2,
            "logret_lag_3": logret_lag_3,
            "logret_lag_4": logret_lag_4,
            "logret_lag_5": logret_lag_5,
            "ret_autocorr_64": ret_autocorr_64,
            # New vol/risk
            "bb_width_20": bb_width,
            "donchian_width_20": donch_w,
            "true_range_pct": true_range_pct,
            "vol_of_vol_atr_64": vol_of_vol_atr_64,
            "vol_of_vol_retstd_64": vol_of_vol_retstd_64,
            "vol_skewness_30": vol_skewness,
            "volatility_regime": volatility_regime,
            # Volume
            "obv_z": obv_z,
            "volume_delta_z": volume_delta.astype(float),
            "vwap_ratio_20": vwap_ratio,
            "vol_range_ratio": vol_range_ratio,
            "amihud_illiquidity": amihud_illiquidity,
            # Microstructure
            "candle_body_frac": candle_body_frac,
            "body_range_ratio_signed": body_range_ratio_signed,
            "hl_asymmetry": hl_asymmetry,
            "upper_shadow_frac": upper_shadow_frac,
            "lower_shadow_frac": lower_shadow_frac,
            "candle_trend_persistence": candle_trend_persistence,
            "dir_change_count_20": dir_change_count_20,
            "kurtosis_rolling_100": kurtosis_rolling,
            # Stats/Fractal
            "dfa_exponent_64": dfa_exponent,
            "entropy_return_64": entropy_return,
            # Risk mgmt
            "drawdown_z_64": drawdown_z,
            "expected_shortfall_0_05_128": expected_shortfall,
            "es_trend_20": es_trend_20,
        }, index=base.index)

        # ADX (14) normalized
        try:
            adx_14 = compute_adx(h, l, c, period=14)
        except Exception:
            adx_14 = np.zeros(len(c), dtype=float)
        feats["adx_14"] = adx_14

        # RLTE pockets (Linear Regression 90 over price with BB(30, 0.9))
        try:
            idx = np.arange(len(c), dtype=float)
            L_reg = 90
            # Rolling sums for slope/intercept in O(T)
            s_idx = pd.Series(idx)
            s_c = pd.Series(c)
            roll = L_reg
            minp = 10
            sum_x = s_idx.rolling(roll, min_periods=minp).sum()
            sum_y = s_c.rolling(roll, min_periods=minp).sum()
            sum_xx = pd.Series(idx * idx).rolling(roll, min_periods=minp).sum()
            sum_xy = pd.Series(idx * c).rolling(roll, min_periods=minp).sum()
            n = s_idx.rolling(roll, min_periods=minp).count()
            denom = (n * sum_xx - sum_x * sum_x).replace(0.0, np.nan)
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            reg90 = (slope * s_idx + intercept).to_numpy()
            # Fill early NaNs conservatively
            reg90 = np.nan_to_num(reg90, nan=c.astype(float))

            # Bollinger(30, 0.9)
            bb_len = 30
            bb_mult = 0.9
            bb_mid_rlte = pd.Series(c).rolling(bb_len, min_periods=1).mean()
            bb_std_rlte = pd.Series(c).rolling(bb_len, min_periods=1).std().fillna(0.0)
            bb_upper_rlte = (bb_mid_rlte + bb_mult * bb_std_rlte).to_numpy()
            bb_lower_rlte = (bb_mid_rlte - bb_mult * bb_std_rlte).to_numpy()

            in_buy = (reg90 < bb_lower_rlte).astype(float)
            in_sell = (reg90 > bb_upper_rlte).astype(float)
            buy_gap = np.maximum(0.0, bb_lower_rlte - reg90) / (c + 1e-12)
            sell_gap = np.maximum(0.0, reg90 - bb_upper_rlte) / (c + 1e-12)
            price_inside_buy = ((in_buy > 0) & (c >= reg90) & (c <= bb_lower_rlte)).astype(float)
            price_inside_sell = ((in_sell > 0) & (c >= bb_upper_rlte) & (c <= reg90)).astype(float)
            reg90_ratio = (reg90 - c) / (c + 1e-12)

            feats["rlte_reg90"] = reg90.astype(float)
            feats["rlte_reg90_ratio"] = reg90_ratio
            feats["rlte_in_buy_pocket"] = in_buy.astype(float)
            feats["rlte_in_sell_pocket"] = in_sell.astype(float)
            feats["rlte_buy_pocket_gap"] = buy_gap.astype(float)
            feats["rlte_sell_pocket_gap"] = sell_gap.astype(float)
            feats["rlte_price_inside_buy"] = price_inside_buy.astype(float)
            feats["rlte_price_inside_sell"] = price_inside_sell.astype(float)
            # RLTE densities and run lengths over 60 bars
            try:
                rb = pd.Series(feats["rlte_in_buy_pocket"]).rolling(60, min_periods=5).mean().fillna(0.0).to_numpy()
                rs = pd.Series(feats["rlte_in_sell_pocket"]).rolling(60, min_periods=5).mean().fillna(0.0).to_numpy()
                feats["rlte_buy_density_60"] = rb
                feats["rlte_sell_density_60"] = rs
            except Exception:
                feats["rlte_buy_density_60"] = 0.0
                feats["rlte_sell_density_60"] = 0.0
        except Exception:
            # If RLTE computation fails, add zeros to preserve schema
            feats["rlte_reg90"] = 0.0
            feats["rlte_reg90_ratio"] = 0.0
            feats["rlte_in_buy_pocket"] = 0.0
            feats["rlte_in_sell_pocket"] = 0.0
            feats["rlte_buy_pocket_gap"] = 0.0
            feats["rlte_sell_pocket_gap"] = 0.0
            feats["rlte_price_inside_buy"] = 0.0
            feats["rlte_price_inside_sell"] = 0.0

        # Multi-timeframe moving averages inspired by Pine script (daily/weekly)
        # Compute on resampled closes and forward-fill to align with base index
        if isinstance(base.index, pd.DatetimeIndex):
            close_series = base["close"].astype(float)
            try:
                daily_close = close_series.resample("1D").last()
                weekly_close = close_series.resample("1W").last()
                ma365d = daily_close.rolling(365, min_periods=1).mean().reindex(base.index).ffill().to_numpy()
                ma200d = daily_close.rolling(200, min_periods=1).mean().reindex(base.index).ffill().to_numpy()
                ma50d = daily_close.rolling(50, min_periods=1).mean().reindex(base.index).ffill().to_numpy()
                ma20w = weekly_close.rolling(20, min_periods=1).mean().reindex(base.index).ffill().to_numpy()
            except Exception:
                # Fallback to zeros if resampling fails
                ma365d = np.zeros(len(base), dtype=float)
                ma200d = np.zeros(len(base), dtype=float)
                ma50d = np.zeros(len(base), dtype=float)
                ma20w = np.zeros(len(base), dtype=float)
        else:
            ma365d = np.zeros(len(base), dtype=float)
            ma200d = np.zeros(len(base), dtype=float)
            ma50d = np.zeros(len(base), dtype=float)
            ma20w = np.zeros(len(base), dtype=float)

        feats["MA365D"] = ma365d
        feats["MA200D"] = ma200d
        feats["MA50D"] = ma50d
        feats["MA20W"] = ma20w

        # Long-horizon valuation and cycle distance (daily/weekly aligned)
        try:
            if isinstance(base.index, pd.DatetimeIndex):
                close_series = base["close"].astype(float)
                daily_close = close_series.resample("1D").last()
                # Rolling daily ATH/ATL and distances
                ath_365 = daily_close.rolling(365, min_periods=5).max().reindex(base.index).ffill()
                atl_365 = daily_close.rolling(365, min_periods=5).min().reindex(base.index).ffill()
                feats["dist_ATH_365D"] = (ath_365.to_numpy() - c) / (np.maximum(ath_365.to_numpy(), 1e-12))
                feats["dist_ATL_365D"] = (c - atl_365.to_numpy()) / (np.maximum(atl_365.to_numpy(), 1e-12))
                # Days since ATH/ATL (approximate via window argmax/argmin position)
                dc = daily_close
                def _days_since_extreme(s: pd.Series, win: int, mode: str) -> pd.Series:
                    def _pos(a: np.ndarray) -> float:
                        if a.size == 0:
                            return 0.0
                        idx = int(np.argmax(a)) if mode == "max" else int(np.argmin(a))
                        return float(win - 1 - idx)
                    return s.rolling(win, min_periods=5).apply(_pos, raw=True)
                ds_ath = _days_since_extreme(dc, 365, "max").reindex(base.index).ffill().fillna(0.0)
                ds_atl = _days_since_extreme(dc, 365, "min").reindex(base.index).ffill().fillna(0.0)
                feats["days_since_ATH_365D"] = ds_ath.to_numpy()
                feats["days_since_ATL_365D"] = ds_atl.to_numpy()
                # Drawdown over daily horizons
                dd365 = (ath_365.to_numpy() - c) / (np.maximum(ath_365.to_numpy(), 1e-12))
                # 730D uses weekly proxy if fewer bars exist
                weekly_close = close_series.resample("1W").last()
                ath_730 = weekly_close.rolling(104, min_periods=5).max().reindex(base.index).ffill()
                dd730 = (ath_730.to_numpy() - c) / (np.maximum(ath_730.to_numpy(), 1e-12))
                feats["drawdown_365D"] = dd365
                feats["drawdown_730D"] = dd730
            else:
                feats["dist_ATH_365D"] = 0.0
                feats["dist_ATL_365D"] = 0.0
                feats["days_since_ATH_365D"] = 0.0
                feats["days_since_ATL_365D"] = 0.0
                feats["drawdown_365D"] = 0.0
                feats["drawdown_730D"] = 0.0
        except Exception:
            feats["dist_ATH_365D"] = 0.0
            feats["dist_ATL_365D"] = 0.0
            feats["days_since_ATH_365D"] = 0.0
            feats["days_since_ATL_365D"] = 0.0
            feats["drawdown_365D"] = 0.0
            feats["drawdown_730D"] = 0.0

    # Add higher timeframe features (resampled and forward-filled) if requested
    if extra_timeframes:
        if isinstance(base.index, pd.DatetimeIndex):
            for tf in extra_timeframes:
                try:
                    tf_str = str(tf).upper()
                    # Normalize alias per pandas guidance: hours 'h', days 'D', weeks 'W'
                    # If the alias ends with 'H' replace with 'h', else keep uppercase (e.g., 1D, 1W)
                    if tf_str.endswith('H'):
                        resample_tf = tf_str[:-1] + 'h'
                    else:
                        resample_tf = tf_str
                    agg = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    res = base.resample(resample_tf).agg(agg).dropna(how="all")
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
                        }, index=res.index)

                        # HTF RLTE pockets on rc (Linear Regression 90 + BB(30,0.9))
                        try:
                            idx_tf = np.arange(len(rc), dtype=float)
                            L_reg = 90
                            s_idx_tf = pd.Series(idx_tf, index=res.index)
                            s_c_tf = pd.Series(rc, index=res.index)
                            minp = 10
                            sum_x = s_idx_tf.rolling(L_reg, min_periods=minp).sum()
                            sum_y = s_c_tf.rolling(L_reg, min_periods=minp).sum()
                            sum_xx = pd.Series(idx_tf * idx_tf, index=res.index).rolling(L_reg, min_periods=minp).sum()
                            sum_xy = pd.Series(idx_tf * rc, index=res.index).rolling(L_reg, min_periods=minp).sum()
                            n_tf = s_idx_tf.rolling(L_reg, min_periods=minp).count()
                            denom = (n_tf * sum_xx - sum_x * sum_x).replace(0.0, np.nan)
                            slope_tf = (n_tf * sum_xy - sum_x * sum_y) / denom
                            intercept_tf = (sum_y - slope_tf * sum_x) / n_tf
                            reg90_tf = (slope_tf * s_idx_tf + intercept_tf).to_numpy()
                            reg90_tf = np.nan_to_num(reg90_tf, nan=rc.astype(float))

                            bb_len = 30
                            bb_mult = 0.9
                            bb_mid_tf = pd.Series(rc, index=res.index).rolling(bb_len, min_periods=1).mean()
                            bb_std_tf = pd.Series(rc, index=res.index).rolling(bb_len, min_periods=1).std().fillna(0.0)
                            bb_upper_tf = (bb_mid_tf + bb_mult * bb_std_tf).to_numpy()
                            bb_lower_tf = (bb_mid_tf - bb_mult * bb_std_tf).to_numpy()

                            in_buy_tf = (reg90_tf < bb_lower_tf).astype(float)
                            in_sell_tf = (reg90_tf > bb_upper_tf).astype(float)
                            buy_gap_tf = np.maximum(0.0, bb_lower_tf - reg90_tf) / (rc + 1e-12)
                            sell_gap_tf = np.maximum(0.0, reg90_tf - bb_upper_tf) / (rc + 1e-12)
                            price_inside_buy_tf = ((in_buy_tf > 0) & (rc >= reg90_tf) & (rc <= bb_lower_tf)).astype(float)
                            price_inside_sell_tf = ((in_sell_tf > 0) & (rc >= bb_upper_tf) & (rc <= reg90_tf)).astype(float)
                            reg90_ratio_tf = (reg90_tf - rc) / (rc + 1e-12)

                            feats_tf[f"{tf}_rlte_reg90"] = reg90_tf.astype(float)
                            feats_tf[f"{tf}_rlte_reg90_ratio"] = reg90_ratio_tf
                            feats_tf[f"{tf}_rlte_in_buy_pocket"] = in_buy_tf.astype(float)
                            feats_tf[f"{tf}_rlte_in_sell_pocket"] = in_sell_tf.astype(float)
                            feats_tf[f"{tf}_rlte_buy_pocket_gap"] = buy_gap_tf.astype(float)
                            feats_tf[f"{tf}_rlte_sell_pocket_gap"] = sell_gap_tf.astype(float)
                            feats_tf[f"{tf}_rlte_price_inside_buy"] = price_inside_buy_tf.astype(float)
                            feats_tf[f"{tf}_rlte_price_inside_sell"] = price_inside_sell_tf.astype(float)
                        except Exception:
                            # If RLTE fails, proceed without
                            pass

                        # HTF ADX(14) normalized
                        try:
                            adx_tf = compute_adx(rh, rl, rc, period=14)
                        except Exception:
                            adx_tf = np.zeros(len(rc), dtype=float)
                        feats_tf[f"{tf}_adx_14"] = adx_tf
                    # Align to base index causally
                    feats_tf_aligned = feats_tf.reindex(base.index).ffill().fillna(0.0)
                    feats = feats.join(feats_tf_aligned, how="left")
                    # HTF alignment feature: sign(LTF logret) * sign(HTF logret)
                    try:
                        align = np.sign(feats["logret"].astype(float).values) * np.sign(feats[f"{tf}_logret"].astype(float).values)
                        feats[f"{tf}_align_ret"] = align
                    except Exception:
                        feats[f"{tf}_align_ret"] = 0.0
                except Exception:
                    # If resampling fails, skip this timeframe silently
                    continue
            # After processing all HTFs, compute TF agreement composites
            try:
                def _col(name: str) -> np.ndarray:
                    return feats[name].astype(float).to_numpy() if name in feats.columns else np.zeros(len(feats), dtype=float)
                ema_list = [np.sign(_col("ema_fast_ratio"))]
                rsi_list = [(_col("rsi") > 0.5).astype(float)]
                if any(c.startswith("4H_") for c in feats.columns):
                    ema_list.append(np.sign(_col("4H_ema_fast_ratio")))
                    rsi_list.append((_col("4H_rsi") > 0.5).astype(float))
                if any(c.startswith("1D_") for c in feats.columns):
                    ema_list.append(np.sign(_col("1D_ema_fast_ratio")))
                    rsi_list.append((_col("1D_rsi") > 0.5).astype(float))
                if any(c.startswith("1W_") for c in feats.columns):
                    ema_list.append(np.sign(_col("1W_ema_fast_ratio")))
                    rsi_list.append((_col("1W_rsi") > 0.5).astype(float))
                feats["tf_trend_agreement"] = np.mean(np.vstack(ema_list), axis=0)
                feats["tf_rsi_agreement"] = np.mean(np.vstack(rsi_list), axis=0)
            except Exception:
                feats["tf_trend_agreement"] = 0.0
                feats["tf_rsi_agreement"] = 0.0
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



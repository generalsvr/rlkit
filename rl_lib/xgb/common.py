from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl_lib.train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df


# ---------- IO helpers ----------
def _ensure_outdir(p: str) -> str:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return p


def _save_feature_columns(model_path: str, cols: List[str]):
    try:
        with open(str(Path(model_path).with_suffix("").as_posix()) + "_feature_columns.json", "w") as f:
            json.dump(list(cols), f)
    except Exception:
        pass


def _coerce_opt(value: Any, default: Any):
    try:
        from typer.models import OptionInfo  # type: ignore
        if isinstance(value, OptionInfo):
            return default
    except Exception:
        pass
    return value


def _resolve_device(dev: str) -> str:
    s = str(dev).lower()
    if s in ("auto", "cuda"):
        try:
            import cupy as _cp  # type: ignore
            _ = _cp.zeros(1)
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


# ---------- Plotting ----------
def _ts_outdir(base: str, prefix: str = "eval") -> str:
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base, f"{prefix}_{ts}")
    os.makedirs(root, exist_ok=True)
    return root


def _safe_savefig(fig, out_path: str):
    try:
        import matplotlib.pyplot as _plt
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        _plt.close(fig)
    except Exception:
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass


def _plot_price_with_events(index, close: np.ndarray, events: Dict[str, np.ndarray], title: str, out_path: str, *,
                            o: Optional[np.ndarray] = None,
                            h: Optional[np.ndarray] = None,
                            l: Optional[np.ndarray] = None,
                            use_candles: bool = False):
    import matplotlib.pyplot as plt
    import numpy as _np
    fig, ax = plt.subplots(figsize=(14, 6))
    if bool(use_candles) and o is not None and h is not None and l is not None and len(o) == len(close):
        try:
            Tn = len(close)
            width = 0.8
            try:
                import pandas as _pd
                if isinstance(index, _pd.DatetimeIndex) and len(index) >= 2:
                    dt = (index[1] - index[0]).total_seconds() / 86400.0
                    width = max(0.0005, float(dt) * 0.7)
            except Exception:
                pass
            up = (close >= o)
            colors_body = _np.where(up, "#2ca02c", "#d62728")
            ax.vlines(index, l, h, color=colors_body, linewidth=0.8, alpha=0.9)
            heights = _np.abs(close - o)
            bottoms = _np.minimum(close, o)
            ax.bar(index, height=heights, bottom=bottoms, width=width, color=colors_body, alpha=0.75, linewidth=0.0)
        except Exception:
            ax.plot(index, close, color="#1f77b4", linewidth=1.2, label="Close")
    else:
        ax.plot(index, close, color="#1f77b4", linewidth=1.2, label="Close")
    colors = {
        "bottom": "#2ca02c",
        "top": "#d62728",
        "up_sig": "#17becf",
        "dn_sig": "#9467bd",
        "tc_up": "#2ca02c",
        "tc_dn": "#d62728",
        "trend_up": "#2ca02c",
        "trend_dn": "#d62728",
        "buy": "#2ca02c",
        "sell": "#d62728",
    }
    for name, mask in events.items():
        if mask is None:
            continue
        try:
            xs = index[mask.astype(bool)]
            ys = close[mask.astype(bool)]
            ax.scatter(xs, ys, s=18, label=name, alpha=0.85, zorder=5, color=colors.get(name, None))
        except Exception:
            pass
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _safe_savefig(fig, out_path)


def _plot_prob_series(index, series: Dict[str, np.ndarray], thresholds: Optional[Dict[str, float]], title: str, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, arr in series.items():
        ax.plot(index, arr, linewidth=1.0, label=name)
    if thresholds:
        for name, thr in thresholds.items():
            ax.axhline(float(thr), linestyle="--", linewidth=0.8, alpha=0.6, label=f"{name}_thr={thr}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=4)
    _safe_savefig(fig, out_path)


def _plot_feature_importance(model: Any, feature_names: List[str], out_path: str, title: str, top_k: int = 30):
    import matplotlib.pyplot as plt
    names = list(feature_names)
    imp_map: Dict[str, float] = {}
    try:
        booster = model.get_booster()  # type: ignore[attr-defined]
        raw = booster.get_score(importance_type="gain")
        for k, v in raw.items():
            try:
                idx = int(k[1:]) if k.startswith("f") else int(k)
                nm = names[idx] if 0 <= idx < len(names) else k
            except Exception:
                nm = k
            imp_map[nm] = float(v)
    except Exception:
        try:
            vals = getattr(model, "feature_importances_", None)
            if vals is not None:
                for i, v in enumerate(list(vals)):
                    nm = names[i] if i < len(names) else f"f{i}"
                    imp_map[nm] = float(v)
        except Exception:
            imp_map = {}
    if not imp_map:
        return
    items = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top_k))]
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, int(len(items) * 0.3))))
    ax.barh(labels, values, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    _safe_savefig(fig, out_path)


def _extract_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    names = list(feature_names)
    imp_map: Dict[str, float] = {}
    try:
        booster = model.get_booster()  # type: ignore[attr-defined]
        raw = booster.get_score(importance_type="gain")
        for k, v in raw.items():
            try:
                idx = int(k[1:]) if k.startswith("f") else int(k)
                nm = names[idx] if 0 <= idx < len(names) else k
            except Exception:
                nm = k
            imp_map[nm] = float(v)
    except Exception:
        try:
            vals = getattr(model, "feature_importances_", None)
            if vals is not None:
                for i, v in enumerate(list(vals)):
                    nm = names[i] if i < len(names) else f"f{i}"
                    imp_map[nm] = float(v)
        except Exception:
            imp_map = {}
    return imp_map


def _save_feature_importance_text(model: Any, feature_names: List[str], out_txt_path: str, top_k: int = 200):
    try:
        imp_map = _extract_feature_importance(model, feature_names)
        if not imp_map:
            return
        items = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top_k))]
        lines = [f"{k}\t{v:.6g}" for k, v in items]
        with open(out_txt_path, 'w') as f:
            f.write("\n".join(lines))
    except Exception:
        pass


def _plot_regdir_shading(index, close: np.ndarray, reg_dir: np.ndarray, title: str, out_path: str):
    import matplotlib.pyplot as plt
    import numpy as _np
    T = int(_np.size(close))
    vals = _np.clip(_np.asarray(reg_dir, dtype=float), -2.0, 2.0) / 2.0
    bg = vals.reshape(1, T)
    fig, ax = plt.subplots(figsize=(14, 5))
    try:
        ax.imshow(bg, aspect='auto', cmap='RdYlGn', alpha=0.18,
                  extent=[0, T, float(_np.nanmin(close)), float(_np.nanmax(close))])
    except Exception:
        pass
    ax.plot(range(T), close, color='#1f77b4', linewidth=1.1, label='Close')
    ax.axhline(float(_np.nanmedian(close)), linestyle=':', color='#999', linewidth=0.7, alpha=0.6)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _safe_savefig(fig, out_path)


def _plot_logret_classes(index, close: np.ndarray, prob_mat: np.ndarray, classes: List[int], title: str, out_path: str, strong_thr: float = 0.5):
    import matplotlib.pyplot as plt
    import numpy as _np
    T = prob_mat.shape[0]
    cls_idx = _np.argmax(prob_mat, axis=1)
    max_p = _np.max(prob_mat, axis=1)
    cls_vals = _np.asarray([classes[i] for i in cls_idx], dtype=int)
    band = (_np.array([cls_vals]) + 2).astype(float)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 0.5], hspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(range(T), close, color='#1f77b4', linewidth=1.0)
    strong_mask = max_p >= float(strong_thr)
    pos_mask = strong_mask & (cls_vals > 0)
    neg_mask = strong_mask & (cls_vals < 0)
    neu_mask = strong_mask & (cls_vals == 0)
    try:
        ax1.scatter(_np.where(pos_mask)[0], close[pos_mask], s=18, color='#2ca02c', label='Strong +', zorder=5)
        ax1.scatter(_np.where(neg_mask)[0], close[neg_mask], s=18, color='#d62728', label='Strong -', zorder=5)
        ax1.scatter(_np.where(neu_mask)[0], close[neu_mask], s=14, color='#7f7f7f', label='Strong 0', zorder=5)
    except Exception:
        pass
    ax1.legend(loc='best')
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    try:
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['#8c564b', '#d62728', '#7f7f7f', '#2ca02c', '#17becf'])
        ax2.imshow(band, aspect='auto', cmap=cmap, interpolation='nearest')
        ax2.set_yticks([])
        ax2.set_xlim(0, T)
    except Exception:
        pass
    _safe_savefig(fig, out_path)


# ---------- CV & Permutation ----------
def _make_time_series_splits(
    num_rows: int,
    n_splits: int,
    min_train: int,
    val_size: int,
    scheme: str = "expanding",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if n_splits <= 0 or num_rows <= (min_train + val_size):
        return splits
    s = scheme.strip().lower()
    for k in range(int(n_splits)):
        if s == "rolling":
            train_start = max(0, (min_train + k * val_size) - min_train)
            train_end = min_train + k * val_size
        else:  # expanding
            train_start = 0
            train_end = min_train + k * val_size
        val_start = train_end
        val_end = min(num_rows, val_start + val_size)
        if val_end - val_start < max(20, int(val_size * 0.5)):
            break
        if val_start <= train_start or train_end <= train_start:
            break
        train_idx = np.arange(train_start, train_end, dtype=int)
        val_idx = np.arange(val_start, val_end, dtype=int)
        splits.append((train_idx, val_idx))
        if val_end >= num_rows:
            break
    return splits


def _permute_labels_time_aware(y: np.ndarray, mode: str = "shift", rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(42)
    y_perm = y.copy()
    m = mode.strip().lower()
    if m == "shuffle":
        rng.shuffle(y_perm)
        return y_perm
    if y_perm.size <= 2:
        return y_perm
    off = int(rng.randint(1, max(2, y_perm.size - 1)))
    return np.roll(y_perm, off)


# ---------- XGB utils ----------
def _load_logret_horizon(logret_path: str) -> int:
    try:
        stem = str(Path(str(logret_path)).with_suffix("").as_posix())
        meta_path = stem + "_meta.json"
        if os.path.exists(meta_path):
            import json as _json
            with open(meta_path, 'r') as f:
                md = _json.load(f)
            return int(max(1, int(md.get("horizon", 1))))
    except Exception:
        pass
    return 1


def _train_xgb_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
    n_estimators: int = 600,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    n_jobs: int = 0,
    objective: str = "binary:logistic",
    num_class: int = 0,
) -> Tuple[Any, Dict[str, float]]:
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, f1_score, accuracy_score

    dev = _resolve_device(device)
    kwargs = dict(
        tree_method="hist",
        device=dev,
        random_state=42,
        n_jobs=(int(n_jobs) if int(n_jobs) > 0 else -1),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_weight=float(min_child_weight),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        n_estimators=int(n_estimators),
        eval_metric=("mlogloss" if objective.startswith("multi:") else "aucpr"),
    )
    if objective.startswith("multi:"):
        model = xgb.XGBClassifier(objective=objective, num_class=int(max(2, num_class)), **kwargs)
    else:
        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        spw = float(max(1.0, (neg / max(1.0, pos))))
        model = xgb.XGBClassifier(objective=objective, scale_pos_weight=spw, **kwargs)

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    metrics: Dict[str, float] = {}
    if objective.startswith("multi:"):
        p = model.predict_proba(X_val)
        y_hat = np.argmax(p, axis=1)
        try:
            metrics["acc"] = float(accuracy_score(y_val, y_hat))
        except Exception:
            metrics["acc"] = float("nan")
    else:
        p = model.predict_proba(X_val)[:, 1]
        y_hat = (p >= 0.5).astype(int)
        try:
            metrics["auprc"] = float(average_precision_score(y_val, p))
            metrics["f1"] = float(f1_score(y_val, y_hat))
            metrics["acc"] = float(accuracy_score(y_val, y_hat))
        except Exception:
            metrics["auprc"] = metrics.get("auprc", float("nan"))
    return model, metrics


def _predict_with_cols(model: Any, feats: pd.DataFrame, cols: Optional[List[str]]) -> Optional[np.ndarray]:
    if cols is None:
        return None
    X = feats.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    Xv = X.reindex(columns=cols).values
    try:
        return model.predict_proba(Xv)
    except Exception:
        return None


# ---------- Dataset & backtest ----------
def _ensure_backtest_config(userdir: str, timeframe: str, exchange: str, pair: str) -> Path:
    config_path = Path(userdir) / "config.json"
    if not config_path.exists():
        cfg = {
            "timeframe": timeframe,
            "user_data_dir": str(userdir),
            "strategy": "XGBStackedStrategy",
            "exchange": {
                "name": exchange,
                "key": "",
                "secret": "",
                "pair_whitelist": [pair]
            },
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "dry_run": True,
            "max_open_trades": 1,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "dataformat_ohlcv": "parquet",
        }
        os.makedirs(userdir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return config_path


def _run_backtest_stacked(
    userdir: str,
    pair: str,
    timeframe: str,
    timerange: str,
    model_dir: str,
    extra_tfs: str,
    exchange: str,
):
    env = os.environ.copy()
    env["XGB_BOTTOM_PATH"] = str(Path(model_dir) / "best_topbot_bottom.json")
    env["XGB_TOP_PATH"] = str(Path(model_dir) / "best_topbot_top.json")
    env["XGB_LOGRET_PATH"] = str(Path(model_dir) / "best_logret.json")
    env["XGB_META_PATH"] = str(Path(model_dir) / "best_meta.json")
    up_path = Path(model_dir) / "best_impulse_up.json"
    dn_path = Path(model_dir) / "best_impulse_down.json"
    if up_path.exists():
        env["XGB_UP_PATH"] = str(up_path)
    if dn_path.exists():
        env["XGB_DN_PATH"] = str(dn_path)
    env["XGB_EXTRA_TFS"] = extra_tfs
    config_path = _ensure_backtest_config(userdir, timeframe, exchange, pair)
    cmd = [
        "freqtrade", "backtesting",
        "--userdir", userdir,
        "--config", str(config_path),
        "--strategy", "XGBStackedStrategy",
        "--timeframe", timeframe,
        "--pairs", pair,
        "--timerange", timerange,
    ]
    subprocess.run(cmd, check=True, env=env)


def _ensure_dataset(userdir: str, pair: str, timeframe: str, exchange: str, timerange: str = "20190101-") -> Optional[str]:
    hit = _find_data_file(userdir, pair, timeframe, prefer_exchange=exchange)
    if hit and os.path.exists(hit):
        return hit
    Path(userdir).mkdir(parents=True, exist_ok=True)

    variants = [pair]
    up = pair.upper()
    if ":" not in pair and (up.endswith("/USDT") or up.endswith("_USDT")):
        variants.append(f"{pair}:USDT")

    last_err: Optional[Exception] = None
    for pv in variants:
        cmd = [
            "freqtrade", "download-data",
            "--pairs", pv,
            "--timeframes", timeframe,
            "--userdir", userdir,
            "--timerange", timerange,
            "--exchange", exchange,
            "--data-format-ohlcv", "parquet",
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            last_err = e
        hit = _find_data_file(userdir, pv, timeframe, prefer_exchange=exchange)
        if hit and os.path.exists(hit):
            return hit

    if last_err is not None:
        pass
    return _find_data_file(userdir, pair, timeframe, prefer_exchange=exchange)


# ---------- Labeling helpers ----------
def _label_pivots(close: np.ndarray, left: int, right: int, min_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(close)
    y_bot = np.zeros(T, dtype=int)
    y_top = np.zeros(T, dtype=int)
    last_b = -10**9
    last_t = -10**9
    for i in range(T):
        l0 = max(0, i - left)
        r1 = min(T, i + right + 1)
        win = close[l0:r1]
        if win.size == 0:
            continue
        c = close[i]
        if i - last_b >= min_gap and c == np.min(win):
            y_bot[i] = 1
            last_b = i
        if i - last_t >= min_gap and c == np.max(win):
            y_top[i] = 1
            last_t = i
    amb = (y_bot == 1) & (y_top == 1)
    y_bot[amb] = 0
    y_top[amb] = 0
    return y_bot, y_top


def _compute_atr_norm_fast(feats: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    if "atr" in feats.columns:
        try:
            a = feats["atr"].astype(float).to_numpy()
            if np.any(np.isfinite(a)):
                return a
        except Exception:
            pass
    high = feats["high"].astype(float).to_numpy() if "high" in feats.columns else raw["high"].astype(float).to_numpy()
    low = feats["low"].astype(float).to_numpy() if "low" in feats.columns else raw["low"].astype(float).to_numpy()
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().to_numpy() / (close + 1e-12)
    return atr


def _strong_pivot_mask(
    close: np.ndarray,
    atr_norm: np.ndarray,
    y_top: np.ndarray,
    y_bot: np.ndarray,
    post_h: int,
    pre_h: int,
    thr_mult: float,
    pre_mult: float,
) -> np.ndarray:
    T = int(len(close))
    Hf = int(max(1, post_h))
    Hb = int(max(1, pre_h))
    strong = np.zeros(T, dtype=bool)
    for k in np.flatnonzero((y_top == 1) | (y_bot == 1)):
        c0 = float(close[k])
        atr0 = float(atr_norm[k]) if np.isfinite(atr_norm[k]) else 0.0
        if atr0 <= 0.0:
            continue
        j0 = max(0, k - Hb)
        if y_top[k] == 1:
            pre_move = (c0 - float(np.min(close[j0:k+1]))) / (c0 + 1e-12)
        else:
            pre_move = (float(np.max(close[j0:k+1])) - c0) / (c0 + 1e-12)
        if pre_move < (pre_mult * atr0):
            continue
        j1 = min(T, k + Hf + 1)
        if (j1 - (k+1)) <= 0:
            continue
        if y_top[k] == 1:
            post_move = (c0 - float(np.min(close[k+1:j1]))) / (c0 + 1e-12)
        else:
            post_move = (float(np.max(close[k+1:j1])) - c0) / (c0 + 1e-12)
        if post_move >= (thr_mult * atr0):
            strong[k] = True
    return strong


def _label_trendchange_strong(
    close: np.ndarray,
    atr_norm: np.ndarray,
    y_top: np.ndarray,
    y_bot: np.ndarray,
    horizon: int,
    pre_h: int,
    thr_mult: float,
    pre_mult: float,
    anchor_offset: int,
    mode: str = "anchor",
) -> np.ndarray:
    strong = _strong_pivot_mask(close, atr_norm, y_top, y_bot, post_h=int(horizon), pre_h=int(pre_h), thr_mult=float(thr_mult), pre_mult=float(pre_mult))
    piv_idx = np.flatnonzero(strong)
    T = int(len(close))
    y = np.zeros(T, dtype=int)
    m = str(mode).lower().strip()
    if m == "window":
        H = int(max(1, horizon))
        for t in range(T):
            r1 = t + 1
            r2 = min(T, t + H + 1)
            if r1 < r2:
                if np.any(strong[r1:r2]):
                    y[t] = 1
    else:
        off = int(max(0, anchor_offset))
        for k in piv_idx:
            t = int(max(0, k - off))
            y[t] = 1
    return y


def _load_xgb(path: str, device: str = "auto") -> Tuple[Any, Optional[List[str]]]:
    import xgboost as xgb
    dev = _resolve_device(device)
    clf = xgb.XGBClassifier(device=dev)
    clf.load_model(path)
    cols_path = str(Path(path).with_suffix("").as_posix()) + "_feature_columns.json"
    cols = None
    if os.path.exists(cols_path):
        try:
            with open(cols_path, "r") as f:
                cols = json.load(f)
        except Exception:
            cols = None
    return clf, cols



"""
CLI for training XGBoost stack models:
 - Top/Bottom classifiers (pivot detectors)
 - Logret multi-class classifier (movement character)
 - Meta MLP manager trained with Triple-Barrier labels

Usage examples (BTC 1h):
  python xgb_trainer.py topbot-train --pair BTC/USDT --timeframe 1h --timerange 20190101-
  python xgb_trainer.py logret-train --pair BTC/USDT --timeframe 1h --timerange 20190101-
  python xgb_trainer.py meta-train --pair BTC/USDT --timeframe 1h --timerange 20190101-
  python xgb_trainer.py train-all --pair BTC/USDT --timeframe 1h --timerange 20190101-
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import subprocess

import numpy as np
import pandas as pd
import typer

from rl_lib.train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df
from rl_lib.features import make_features
from rl_lib.meta import triple_barrier_labels, train_meta_mlp_manager
from rl_lib.autoencoder import AETrainParams, train_autoencoder, compute_embeddings, compute_embeddings_from_raw


app = typer.Typer(add_completion=False)


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
@app.command("ae-train")
def ae_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    window: int = typer.Option(128),
    embed_dim: int = typer.Option(16),
    base_channels: int = typer.Option(32),
    batch_size: int = typer.Option(256),
    epochs: int = typer.Option(40),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-5),
    device: str = typer.Option("auto"),
    out_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "ae_conv1d.json")),
    raw_htf: bool = typer.Option(False, help="Train AE on raw OHLCV + multi-HTF only (no indicators)"),
    raw_extra_timeframes: str = typer.Option("4H,1D,1W", help="HTFs to include when raw_htf=True"),
    ae_cols: str = typer.Option("close,volume", help="Comma list of base columns to include (when raw_htf=True)"),
):
    params = AETrainParams(
        pair=str(_coerce_opt(pair, "BTC/USDT")),
        timeframe=str(_coerce_opt(timeframe, "1h")),
        userdir=str(_coerce_opt(userdir, str(Path(__file__).resolve().parent / "freqtrade_userdir"))),
        timerange=str(_coerce_opt(timerange, "20190101-")),
        prefer_exchange=str(_coerce_opt(prefer_exchange, "bybit")),
        feature_mode=str(_coerce_opt(feature_mode, "full")),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=str(_coerce_opt(extra_timeframes, "4H,1D,1W")),
        window=int(_coerce_opt(window, 128)),
        embed_dim=int(_coerce_opt(embed_dim, 16)),
        base_channels=int(_coerce_opt(base_channels, 32)),
        batch_size=int(_coerce_opt(batch_size, 256)),
        epochs=int(_coerce_opt(epochs, 40)),
        lr=float(_coerce_opt(lr, 1e-3)),
        weight_decay=float(_coerce_opt(weight_decay, 1e-5)),
        device=str(_coerce_opt(device, "auto")),
        out_path=str(_coerce_opt(out_path, out_path)),
    )
    params.raw_htf = bool(_coerce_opt(raw_htf, False))
    params.raw_extra_timeframes = str(_coerce_opt(raw_extra_timeframes, "4H,1D,1W"))
    params.ae_cols = str(_coerce_opt(ae_cols, "close,volume"))
    os.makedirs(Path(out_path).parent, exist_ok=True)
    meta = train_autoencoder(params)
    typer.echo(json.dumps({"ae": meta}, indent=2))



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
    # Clear ambiguous overlap where both 1 -> set none
    amb = (y_bot == 1) & (y_top == 1)
    y_bot[amb] = 0
    y_top[amb] = 0
    return y_bot, y_top


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


# -----------------------------
# Plotting helpers
# -----------------------------

def _ts_outdir(base: str, prefix: str = "eval") -> str:
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base, f"{prefix}_{ts}")
    os.makedirs(root, exist_ok=True)
    return root


def _safe_savefig(fig, out_path: str):
    try:
        import matplotlib.pyplot as _plt
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        _plt.close(fig)
    except Exception:
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass


def _plot_price_with_events(index, close: np.ndarray, events: Dict[str, np.ndarray], title: str, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(index, close, color="#1f77b4", linewidth=1.2, label="Close")
    colors = {
        "bottom": "#2ca02c",
        "top": "#d62728",
        "up_sig": "#17becf",
        "dn_sig": "#9467bd",
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
    # Try booster-based importance for robustness
    names = list(feature_names)
    imp_map: Dict[str, float] = {}
    try:
        booster = model.get_booster()  # type: ignore[attr-defined]
        raw = booster.get_score(importance_type="gain")
        # raw keys like 'f0','f1', map to names if available
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


# -----------------------------
# CV and permutation helpers
# -----------------------------

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
    # default: circular shift by random offset preserving run-lengths distribution
    if y_perm.size <= 2:
        return y_perm
    off = int(rng.randint(1, max(2, y_perm.size - 1)))
    return np.roll(y_perm, off)


# -----------------------------
# Logret advanced visuals
# -----------------------------

def _plot_regdir_shading(index, close: np.ndarray, reg_dir: np.ndarray, title: str, out_path: str):
    import matplotlib.pyplot as plt
    import numpy as _np
    T = int(_np.size(close))
    vals = _np.clip(_np.asarray(reg_dir, dtype=float), -2.0, 2.0) / 2.0  # [-1,1]
    # Build a 2D array for background shading across full y-range
    bg = vals.reshape(1, T)
    fig, ax = plt.subplots(figsize=(14, 5))
    # imshow with diverging colormap; red negative, green positive
    try:
        # extent uses x in sample index space
        ax.imshow(bg, aspect='auto', cmap='RdYlGn', alpha=0.18,
                  extent=[0, T, float(_np.nanmin(close)), float(_np.nanmax(close))])
    except Exception:
        pass
    ax.plot(range(T), close, color='#1f77b4', linewidth=1.1, label='Close')
    ax.axhline(float(_np.nanmedian(close)), linestyle=':', color='#999', linewidth=0.7, alpha=0.6)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _safe_savefig(fig, out_path)


def _plot_logret_classes(index, close: np.ndarray, prob_mat: np.ndarray, classes: list[int], title: str, out_path: str, strong_thr: float = 0.5):
    import matplotlib.pyplot as plt
    import numpy as _np
    T = prob_mat.shape[0]
    cls_idx = _np.argmax(prob_mat, axis=1)
    max_p = _np.max(prob_mat, axis=1)
    # Map classes {-2,-1,0,1,2} to 0..4 for colormap
    cls_vals = _np.asarray([classes[i] for i in cls_idx], dtype=int)
    # Normalize to 0..4 band
    band = (_np.array([cls_vals]) + 2).astype(float)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 0.5], hspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # Price with strong signals
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

    # Class band (0..4) with custom colormap mapping [-2..2]
    try:
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['#8c564b', '#d62728', '#7f7f7f', '#2ca02c', '#17becf'])
        ax2.imshow(band, aspect='auto', cmap=cmap, interpolation='nearest')
        ax2.set_yticks([])
        ax2.set_xlim(0, T)
        # xticks keep
    except Exception:
        pass
    _safe_savefig(fig, out_path)


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
        # Class imbalance weight for binary
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
    typer.echo(f"Running backtest: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def _ensure_dataset(userdir: str, pair: str, timeframe: str, exchange: str, timerange: str = "20190101-") -> Optional[str]:
    """Ensure dataset exists; if not, try multiple pair variants with Freqtrade download-data."""
    # Fast path
    hit = _find_data_file(userdir, pair, timeframe, prefer_exchange=exchange)
    if hit and os.path.exists(hit):
        return hit
    Path(userdir).mkdir(parents=True, exist_ok=True)

    variants = [pair]
    up = pair.upper()
    # For Bybit futures, BTC/USDT:USDT is often required
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
            typer.echo(f"Downloading dataset: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            last_err = e
        # Check after each attempt
        hit = _find_data_file(userdir, pv, timeframe, prefer_exchange=exchange)
        if hit and os.path.exists(hit):
            return hit

    if last_err is not None:
        typer.echo(f"Download attempts failed for variants {variants}: {last_err}")
    # Final check with original pair
    return _find_data_file(userdir, pair, timeframe, prefer_exchange=exchange)


@app.command()
def topbot_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W", help="Optional comma-separated HTFs e.g. '4H,1D,1W'"),
    ae_path: str = typer.Option("", help="Optional path to AE manifest (.json) to append embeddings"),
    left_bars: int = typer.Option(3),
    right_bars: int = typer.Option(3),
    min_gap_bars: int = typer.Option(4),
    device: str = typer.Option("auto"),
    n_estimators: int = typer.Option(600),
    max_depth: int = typer.Option(6),
    min_child_weight: float = typer.Option(1.0),
    learning_rate: float = typer.Option(0.05),
    subsample: float = typer.Option(0.8),
    colsample_bytree: float = typer.Option(0.8),
    reg_alpha: float = typer.Option(0.0),
    reg_lambda: float = typer.Option(1.0),
    n_jobs: int = typer.Option(0),
    n_trials: int = typer.Option(0, "--n-trials", "--n_trials", help="If >0, run Optuna tuning"),
    sampler: str = typer.Option("tpe", help="tpe|random"),
    seed: int = typer.Option(42),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    # validation options
    cv_splits: int = typer.Option(0, help="If >0, run time-aware CV with given splits"),
    cv_scheme: str = typer.Option("expanding", help="expanding|rolling"),
    cv_val_size: int = typer.Option(2000, help="Validation fold size (bars)"),
    perm_test: int = typer.Option(0, help="If >0, run permutation test with N shuffles"),
):
    # Auto-download base and HTFs
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        for tf in ["4h", "1d", "1w"]:
            try:
                _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
            except Exception:
                pass
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found. Run download first.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]
    # Coerce Typer OptionInfo to concrete values when invoked programmatically
    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    left_bars = int(_coerce_opt(left_bars, 3))
    right_bars = int(_coerce_opt(right_bars, 3))
    min_gap_bars = int(_coerce_opt(min_gap_bars, 4))
    device = str(_coerce_opt(device, "auto"))
    n_estimators = int(_coerce_opt(n_estimators, 600))
    max_depth = int(_coerce_opt(max_depth, 6))
    min_child_weight = float(_coerce_opt(min_child_weight, 1.0))
    learning_rate = float(_coerce_opt(learning_rate, 0.05))
    subsample = float(_coerce_opt(subsample, 0.8))
    colsample_bytree = float(_coerce_opt(colsample_bytree, 0.8))
    reg_alpha = float(_coerce_opt(reg_alpha, 0.0))
    reg_lambda = float(_coerce_opt(reg_lambda, 1.0))
    n_jobs = int(_coerce_opt(n_jobs, 0))
    n_trials = int(_coerce_opt(n_trials, 0))
    sampler = str(_coerce_opt(sampler, "tpe"))
    seed = int(_coerce_opt(seed, 42))
    left_bars = int(_coerce_opt(left_bars, 3))
    right_bars = int(_coerce_opt(right_bars, 3))
    min_gap_bars = int(_coerce_opt(min_gap_bars, 4))
    device = str(_coerce_opt(device, "auto"))
    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None))
    feats = feats.reset_index(drop=True)
    if str(ae_path).strip():
        try:
            # Detect if AE was trained in raw_htf mode by reading manifest keys
            import json as _json
            with open(ae_path, "r") as _f:
                _man = _json.load(_f)
            if bool(_man.get("raw_htf", False)):
                ae_df = compute_embeddings_from_raw(raw, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae")
                ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
            else:
                ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=int(basic_lookback) if int(basic_lookback) > 0 else 128)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (TopBot), proceeding without: {e}")
    c = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    yb, yt = _label_pivots(c, int(left_bars), int(right_bars), int(min_gap_bars))
    # Split
    valid_len = len(feats)
    if valid_len < 400:
        raise ValueError("Not enough data to train TopBot.")
    cut = int(valid_len * 0.8)
    X_tr = feats.iloc[:cut, :].values
    X_val = feats.iloc[cut:, :].values
    yb_tr = yb[:cut]
    yb_val = yb[cut:]
    yt_tr = yt[:cut]
    yt_val = yt[cut:]

    # Train binary classifiers (Optuna optional)
    def _fit_eval(params: Dict[str, Any]) -> Tuple[Any, Any, float]:
        bot_m, m1 = _train_xgb_classifier(
            X_tr, yb_tr, X_val, yb_val, device,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs,
            objective="binary:logistic",
        )
        top_m, m2 = _train_xgb_classifier(
            X_tr, yt_tr, X_val, yt_val, device,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs,
            objective="binary:logistic",
        )
        val = float(np.nanmean([float(m1.get("auprc", float("nan"))), float(m2.get("auprc", float("nan")))]))
        return bot_m, top_m, val

    if int(n_trials) and n_trials > 0:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler
        smp = RandomSampler(seed=int(seed)) if sampler.lower() == "random" else TPESampler(seed=int(seed))
        def objective(trial: "optuna.trial.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            _, _, score = _fit_eval(params)
            return score
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna TopBot trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        bot_model, top_model, best_score = _fit_eval(best_params)
        typer.echo(f"Best TopBot mean AUPRC: {best_score:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate), subsample=float(subsample), colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)
        )
        bot_model, top_model, _ = _fit_eval(base_params)

    os.makedirs(outdir, exist_ok=True)
    bot_path = os.path.join(outdir, "best_topbot_bottom.json")
    top_path = os.path.join(outdir, "best_topbot_top.json")
    bot_model.save_model(_ensure_outdir(bot_path))
    top_model.save_model(_ensure_outdir(top_path))
    _save_feature_columns(bot_path, list(feats.columns))
    _save_feature_columns(top_path, list(feats.columns))

    # Compute validation metrics for report (robust across tuning paths)
    try:
        from sklearn.metrics import average_precision_score as _aps
        p_b = bot_model.predict_proba(X_val)[:, 1]
        p_t = top_model.predict_proba(X_val)[:, 1]
        m_bot = {"auprc": float(_aps(yb_val, p_b))}
        m_top = {"auprc": float(_aps(yt_val, p_t))}
    except Exception:
        m_bot = {}; m_top = {}
    typer.echo(json.dumps({"bottom": {"path": bot_path, **m_bot}, "top": {"path": top_path, **m_top}}, indent=2))

    # Time-aware CV (optional)
    if int(cv_splits) > 0:
        splits = _make_time_series_splits(
            num_rows=valid_len,
            n_splits=int(cv_splits),
            min_train=max(400, int(valid_len * 0.3)),
            val_size=int(cv_val_size),
            scheme=str(cv_scheme),
        )
        cv_metrics: List[Dict[str, float]] = []
        from sklearn.metrics import average_precision_score
        for tr_idx, va_idx in splits:
            X_tr = feats.iloc[tr_idx, :].values
            X_va = feats.iloc[va_idx, :].values
            yb_tr = yb[tr_idx]
            yb_va = yb[va_idx]
            yt_tr = yt[tr_idx]
            yt_va = yt[va_idx]
            bm, _ = _train_xgb_classifier(X_tr, yb_tr, X_va, yb_va, device, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=n_jobs, objective="binary:logistic")
            tm, _ = _train_xgb_classifier(X_tr, yt_tr, X_va, yt_va, device, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=n_jobs, objective="binary:logistic")
            p_b = bm.predict_proba(X_va)[:, 1]
            p_t = tm.predict_proba(X_va)[:, 1]
            cv_metrics.append({
                "ap_bottom": float(average_precision_score(yb_va, p_b)),
                "ap_top": float(average_precision_score(yt_va, p_t)),
            })
        typer.echo(json.dumps({"cv": cv_metrics}, indent=2))

    # Permutation test (optional)
    if int(perm_test) > 0:
        from sklearn.metrics import average_precision_score
        rng = np.random.RandomState(42)
        p_b = bot_model.predict_proba(X_val)[:, 1]
        p_t = top_model.predict_proba(X_val)[:, 1]
        ap_b = float(average_precision_score(yb_val, p_b))
        ap_t = float(average_precision_score(yt_val, p_t))
        null_b: List[float] = []
        null_t: List[float] = []
        for _ in range(int(perm_test)):
            yb_perm = _permute_labels_time_aware(yb_val, mode="shift", rng=rng)
            yt_perm = _permute_labels_time_aware(yt_val, mode="shift", rng=rng)
            null_b.append(float(average_precision_score(yb_perm, p_b)))
            null_t.append(float(average_precision_score(yt_perm, p_t)))
        pval_b = float((np.sum(np.array(null_b) >= ap_b) + 1) / (len(null_b) + 1))
        pval_t = float((np.sum(np.array(null_t) >= ap_t) + 1) / (len(null_t) + 1))
        typer.echo(json.dumps({"perm_test": {"ap_bottom": ap_b, "pval_bottom": pval_b, "ap_top": ap_t, "pval_top": pval_t}}, indent=2))


@app.command()
def logret_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W", help="Optional comma-separated HTFs e.g. '4H,1D,1W'"),
    ae_path: str = typer.Option("", help="Optional path to AE manifest (.json) to append embeddings"),
    horizon: int = typer.Option(1, help="Forward bars for return aggregation"),
    strong_mult: float = typer.Option(1.5, help="Threshold multiplier for strong moves vs ATR"),
    device: str = typer.Option("auto"),
    n_estimators: int = typer.Option(600),
    max_depth: int = typer.Option(6),
    min_child_weight: float = typer.Option(1.0),
    learning_rate: float = typer.Option(0.05),
    subsample: float = typer.Option(0.8),
    colsample_bytree: float = typer.Option(0.8),
    reg_alpha: float = typer.Option(0.0),
    reg_lambda: float = typer.Option(1.0),
    n_jobs: int = typer.Option(0),
    n_trials: int = typer.Option(0, "--n-trials", "--n_trials", help="If >0, run Optuna tuning"),
    sampler: str = typer.Option("tpe"),
    seed: int = typer.Option(42),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    # validation options
    cv_splits: int = typer.Option(0, help="If >0, run time-aware CV"),
    cv_scheme: str = typer.Option("expanding"),
    cv_val_size: int = typer.Option(2000),
    perm_test: int = typer.Option(0, help="If >0, run permutation test (val only)"),
):
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        for tf in ["4h", "1d", "1w"]:
            try:
                _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
            except Exception:
                pass
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]
    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    horizon = int(_coerce_opt(horizon, 1))
    strong_mult = float(_coerce_opt(strong_mult, 1.5))
    device = str(_coerce_opt(device, "auto"))
    n_estimators = int(_coerce_opt(n_estimators, 600))
    max_depth = int(_coerce_opt(max_depth, 6))
    min_child_weight = float(_coerce_opt(min_child_weight, 1.0))
    learning_rate = float(_coerce_opt(learning_rate, 0.05))
    subsample = float(_coerce_opt(subsample, 0.8))
    colsample_bytree = float(_coerce_opt(colsample_bytree, 0.8))
    reg_alpha = float(_coerce_opt(reg_alpha, 0.0))
    reg_lambda = float(_coerce_opt(reg_lambda, 1.0))
    n_jobs = int(_coerce_opt(n_jobs, 0))
    n_trials = int(_coerce_opt(n_trials, 0))
    sampler = str(_coerce_opt(sampler, "tpe"))
    seed = int(_coerce_opt(seed, 42))
    horizon = int(_coerce_opt(horizon, 1))
    strong_mult = float(_coerce_opt(strong_mult, 1.5))
    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None))
    feats = feats.reset_index(drop=True)
    if str(ae_path).strip():
        try:
            import json as _json
            with open(ae_path, "r") as _f:
                _man = _json.load(_f)
            if bool(_man.get("raw_htf", False)):
                ae_df = compute_embeddings_from_raw(raw, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae")
                ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
            else:
                ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=int(basic_lookback) if int(basic_lookback) > 0 else 128)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (Logret), proceeding without: {e}")
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    logp = np.log(close + 1e-12)
    H = int(max(1, horizon))
    T = len(feats)
    valid = T - H
    if valid <= 300:
        raise ValueError("Not enough data to train Logret classifier.")
    fwd = logp[H:] - logp[:-H]
    atr = feats["atr"].astype(float).to_numpy()
    atr = atr[:valid]
    # Labels: map {-2,-1,0,1,2} -> indices {0,1,2,3,4} for XGBoost
    raw_labels = np.zeros(valid, dtype=int)
    thr = float(strong_mult) * atr
    raw_labels[fwd > thr] = 2
    raw_labels[(fwd > 0.0) & (fwd <= thr)] = 1
    raw_labels[(fwd < 0.0) & (fwd >= -thr)] = -1
    raw_labels[fwd < -thr] = -2
    classes_ord = np.array([-2, -1, 0, 1, 2], dtype=int)
    label_to_idx = {c: i for i, c in enumerate(classes_ord)}
    labels = np.vectorize(lambda x: label_to_idx.get(int(x), 2))(raw_labels).astype(int)
    # Features aligned to valid range
    X = feats.iloc[:valid, :].copy()
    # Train/val split
    cut = int(max(100, min(valid - 50, int(valid * 0.8))))
    X_tr = X.iloc[:cut, :].values
    X_val = X.iloc[cut:, :].values
    y_tr = labels[:cut]
    y_val = labels[cut:]
    # Fit multi-class XGB (Optuna optional)
    def _fit_eval(params: Dict[str, Any]) -> Tuple[Any, float]:
        mdl, metrics = _train_xgb_classifier(
            X_tr, y_tr, X_val, y_val, device,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs,
            objective="multi:softprob",
            num_class=5,
        )
        return mdl, float(metrics.get("acc", float("nan")))

    if int(n_trials) and n_trials > 0:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler
        smp = RandomSampler(seed=int(seed)) if sampler.lower() == "random" else TPESampler(seed=int(seed))
        def objective(trial: "optuna.trial.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            _, acc = _fit_eval(params)
            return acc
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna Logret trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        model, best_acc = _fit_eval(best_params)
        metrics = {"acc": float(best_acc)}
        typer.echo(f"Best Logret ACC: {best_acc:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate), subsample=float(subsample), colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)
        )
        model, acc = _fit_eval(base_params)
        metrics = {"acc": float(acc)}
    os.makedirs(outdir, exist_ok=True)
    path_out = os.path.join(outdir, "best_logret.json")
    model.save_model(_ensure_outdir(path_out))
    _save_feature_columns(path_out, list(X.columns))
    # Save label mapping
    meta = {"classes": [-2, -1, 0, 1, 2], "horizon": int(H), "strong_mult": float(strong_mult), "metrics": metrics}
    with open(str(Path(path_out).with_suffix("").as_posix()) + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    typer.echo(json.dumps({"logret": {"path": path_out, **metrics}}, indent=2))

    # Time-aware CV (optional)
    if int(cv_splits) > 0:
        splits = _make_time_series_splits(
            num_rows=valid,
            n_splits=int(cv_splits),
            min_train=max(600, int(valid * 0.3)),
            val_size=int(cv_val_size),
            scheme=str(cv_scheme),
        )
        cv_metrics: List[Dict[str, float]] = []
        from sklearn.metrics import accuracy_score
        for tr_idx, va_idx in splits:
            X_tr = X.iloc[tr_idx, :].values
            X_va = X.iloc[va_idx, :].values
            y_tr = labels[tr_idx]
            y_va = labels[va_idx]
            mdl, _ = _train_xgb_classifier(X_tr, y_tr, X_va, y_va, device, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=n_jobs, objective="multi:softprob", num_class=5)
            pr = mdl.predict_proba(X_va)
            y_hat = np.argmax(pr, axis=1)
            cv_metrics.append({"acc": float(accuracy_score(y_va, y_hat))})
        typer.echo(json.dumps({"cv": cv_metrics}, indent=2))

    # Permutation test (optional)
    if int(perm_test) > 0:
        from sklearn.metrics import accuracy_score
        pr = model.predict_proba(X_val)
        y_hat = np.argmax(pr, axis=1)
        acc = float(accuracy_score(y_val, y_hat))
        rng = np.random.RandomState(42)
        null_acc: List[float] = []
        for _ in range(int(perm_test)):
            y_perm = _permute_labels_time_aware(y_val, mode="shift", rng=rng)
            null_acc.append(float(accuracy_score(y_perm, y_hat)))
        pval = float((np.sum(np.array(null_acc) >= acc) + 1) / (len(null_acc) + 1))
        typer.echo(json.dumps({"perm_test": {"acc": acc, "pval": pval}}, indent=2))


def _impulse_train_impl(
    pair: str,
    timeframe: str,
    userdir: str,
    timerange: str,
    prefer_exchange: str,
    feature_mode: str,
    basic_lookback: int,
    extra_timeframes: str,
    horizon: int,
    label_mode: str,
    alpha_up: float,
    alpha_dn: float,
    vol_lookback: int,
    thr_up_bps: float,
    thr_dn_bps: float,
    device: str,
    n_estimators: int,
    max_depth: int,
    min_child_weight: float,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    n_jobs: int,
    n_trials: int,
    sampler: str,
    seed: int,
    outdir: str,
    autodownload: bool,
    ae_path: str = "",
    # validation options
    cv_splits: int = 0,
    cv_scheme: str = "expanding",
    cv_val_size: int = 2000,
    perm_test: int = 0,
):
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        for tf in ["4h", "1d", "1w"]:
            try:
                _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
            except Exception:
                pass
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]

    # Coerce
    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    horizon = int(_coerce_opt(horizon, 8))
    label_mode = str(_coerce_opt(label_mode, "vol")).lower()
    alpha_up = float(_coerce_opt(alpha_up, 2.0))
    alpha_dn = float(_coerce_opt(alpha_dn, 2.0))
    vol_lookback = int(_coerce_opt(vol_lookback, 256))
    thr_up_bps = float(_coerce_opt(thr_up_bps, 30.0))
    thr_dn_bps = float(_coerce_opt(thr_dn_bps, 30.0))
    device = str(_coerce_opt(device, "auto"))
    n_estimators = int(_coerce_opt(n_estimators, 600))
    max_depth = int(_coerce_opt(max_depth, 6))
    min_child_weight = float(_coerce_opt(min_child_weight, 1.0))
    learning_rate = float(_coerce_opt(learning_rate, 0.05))
    subsample = float(_coerce_opt(subsample, 0.8))
    colsample_bytree = float(_coerce_opt(colsample_bytree, 0.8))
    reg_alpha = float(_coerce_opt(reg_alpha, 0.0))
    reg_lambda = float(_coerce_opt(reg_lambda, 1.0))
    n_jobs = int(_coerce_opt(n_jobs, 0))
    n_trials = int(_coerce_opt(n_trials, 0))
    sampler = str(_coerce_opt(sampler, "tpe"))
    seed = int(_coerce_opt(seed, 42))

    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None))
    feats = feats.reset_index(drop=True)
    if str(ae_path).strip():
        try:
            ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=int(basic_lookback) if int(basic_lookback) > 0 else 128)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (Impulse), proceeding without: {e}")
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    logp_s = pd.Series(np.log(close + 1e-12), index=feats.index)
    T = len(feats)
    valid = T - horizon
    if valid <= 300:
        raise ValueError("Not enough data to train Impulse classifier.")
    fwd = [logp_s.shift(-i) - logp_s for i in range(1, horizon + 1)]
    fwd_mat = np.vstack([s.to_numpy() for s in fwd])
    fwd_valid = fwd_mat[:, :valid]
    fwd_max = np.nanmax(fwd_valid, axis=0)
    fwd_min = np.nanmin(fwd_valid, axis=0)
    logret = np.diff(np.log(close + 1e-12), prepend=np.log(close[0] + 1e-12))
    sigma = pd.Series(logret).rolling(vol_lookback, min_periods=20).std().fillna(0.0).to_numpy()
    sigma = sigma[:valid]

    if label_mode == "vol":
        thr = sigma * np.sqrt(max(1, horizon))
        y_up = (fwd_max > (alpha_up * thr)).astype(int)
        y_dn = (fwd_min < (-alpha_dn * thr)).astype(int)
    else:
        bps_to_lr = 1e-4
        y_up = (fwd_max > (thr_up_bps * bps_to_lr)).astype(int)
        y_dn = (fwd_min < (-(thr_dn_bps * bps_to_lr))).astype(int)

    X = feats.iloc[:valid, :].copy()
    cut = int(max(100, min(valid - 50, int(valid * 0.8))))
    X_tr = X.iloc[:cut, :].values
    X_val = X.iloc[cut:, :].values
    y_up_tr = y_up[:cut]
    y_up_val = y_up[cut:]
    y_dn_tr = y_dn[:cut]
    y_dn_val = y_dn[cut:]

    def _fit_eval(params: Dict[str, Any]) -> Tuple[Any, Any, float]:
        up_m, mu = _train_xgb_classifier(
            X_tr, y_up_tr, X_val, y_up_val, device,
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]), learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=n_jobs,
            objective="binary:logistic",
        )
        dn_m, md = _train_xgb_classifier(
            X_tr, y_dn_tr, X_val, y_dn_val, device,
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]), learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=n_jobs,
            objective="binary:logistic",
        )
        val = float(np.nanmean([float(mu.get("auprc", float("nan"))), float(md.get("auprc", float("nan")))]))
        return up_m, dn_m, val

    if int(n_trials) and n_trials > 0:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler
        smp = RandomSampler(seed=int(seed)) if sampler.lower() == "random" else TPESampler(seed=int(seed))
        def objective(trial: "optuna.trial.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            _, _, score = _fit_eval(params)
            return score
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna Impulse trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        up_model, dn_model, best_score = _fit_eval(best_params)
        typer.echo(f"Best Impulse mean AUPRC: {best_score:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate), subsample=float(subsample), colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)
        )
        up_model, dn_model, _ = _fit_eval(base_params)

    os.makedirs(outdir, exist_ok=True)
    up_path = os.path.join(outdir, "best_impulse_up.json")
    dn_path = os.path.join(outdir, "best_impulse_down.json")
    up_model.save_model(_ensure_outdir(up_path))
    dn_model.save_model(_ensure_outdir(dn_path))
    _save_feature_columns(up_path, list(X.columns))
    _save_feature_columns(dn_path, list(X.columns))
    typer.echo(json.dumps({"impulse_up": {"path": up_path}, "impulse_down": {"path": dn_path}}, indent=2))

    # Time-aware CV (optional)
    if int(cv_splits) > 0:
        splits = _make_time_series_splits(
            num_rows=valid,
            n_splits=int(cv_splits),
            min_train=max(600, int(valid * 0.3)),
            val_size=int(cv_val_size),
            scheme=str(cv_scheme),
        )
        cv_metrics: List[Dict[str, float]] = []
        from sklearn.metrics import average_precision_score
        for tr_idx, va_idx in splits:
            X_tr = X.iloc[tr_idx, :].values
            X_va = X.iloc[va_idx, :].values
            yu_tr = y_up[tr_idx]
            yu_va = y_up[va_idx]
            yd_tr = y_dn[tr_idx]
            yd_va = y_dn[va_idx]
            um, _ = _train_xgb_classifier(X_tr, yu_tr, X_va, yu_va, device, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=n_jobs, objective="binary:logistic")
            dm, _ = _train_xgb_classifier(X_tr, yd_tr, X_va, yd_va, device, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=n_jobs, objective="binary:logistic")
            pu = um.predict_proba(X_va)[:, 1]
            pdn = dm.predict_proba(X_va)[:, 1]
            cv_metrics.append({
                "ap_up": float(average_precision_score(yu_va, pu)),
                "ap_dn": float(average_precision_score(yd_va, pdn)),
            })
        typer.echo(json.dumps({"cv": cv_metrics}, indent=2))

    # Permutation test (optional) on the held-out validation slice used above
    if int(perm_test) > 0:
        from sklearn.metrics import average_precision_score
        pu = up_model.predict_proba(X_val)[:, 1]
        pdn = dn_model.predict_proba(X_val)[:, 1]
        ap_u = float(average_precision_score(y_up_val, pu))
        ap_d = float(average_precision_score(y_dn_val, pdn))
        rng = np.random.RandomState(42)
        null_u: List[float] = []
        null_d: List[float] = []
        for _ in range(int(perm_test)):
            yu_perm = _permute_labels_time_aware(y_up_val, mode="shift", rng=rng)
            yd_perm = _permute_labels_time_aware(y_dn_val, mode="shift", rng=rng)
            null_u.append(float(average_precision_score(yu_perm, pu)))
            null_d.append(float(average_precision_score(yd_perm, pdn)))
        pval_u = float((np.sum(np.array(null_u) >= ap_u) + 1) / (len(null_u) + 1))
        pval_d = float((np.sum(np.array(null_d) >= ap_d) + 1) / (len(null_d) + 1))
        typer.echo(json.dumps({"perm_test": {"ap_up": ap_u, "pval_up": pval_u, "ap_dn": ap_d, "pval_dn": pval_d}}, indent=2))


@app.command("impulse_train")
def impulse_train_cmd(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    ae_path: str = typer.Option(""),
    horizon: int = typer.Option(8),
    label_mode: str = typer.Option("vol", help="vol|abs"),
    alpha_up: float = typer.Option(2.0),
    alpha_dn: float = typer.Option(2.0),
    vol_lookback: int = typer.Option(256),
    thr_up_bps: float = typer.Option(30.0),
    thr_dn_bps: float = typer.Option(30.0),
    device: str = typer.Option("auto"),
    n_estimators: int = typer.Option(600),
    max_depth: int = typer.Option(6),
    min_child_weight: float = typer.Option(1.0),
    learning_rate: float = typer.Option(0.05),
    subsample: float = typer.Option(0.8),
    colsample_bytree: float = typer.Option(0.8),
    reg_alpha: float = typer.Option(0.0),
    reg_lambda: float = typer.Option(1.0),
    n_jobs: int = typer.Option(0),
    n_trials: int = typer.Option(0, "--n-trials", "--n_trials", help="If >0, run Optuna tuning"),
    sampler: str = typer.Option("tpe"),
    seed: int = typer.Option(42),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    # validation
    cv_splits: int = typer.Option(0),
    cv_scheme: str = typer.Option("expanding"),
    cv_val_size: int = typer.Option(2000),
    perm_test: int = typer.Option(0),
):
    _impulse_train_impl(**locals())


@app.command("impulse-train")
def impulse_train_cmd_dash(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    ae_path: str = typer.Option(""),
    horizon: int = typer.Option(8),
    label_mode: str = typer.Option("vol", help="vol|abs"),
    alpha_up: float = typer.Option(2.0),
    alpha_dn: float = typer.Option(2.0),
    vol_lookback: int = typer.Option(256),
    thr_up_bps: float = typer.Option(30.0),
    thr_dn_bps: float = typer.Option(30.0),
    device: str = typer.Option("auto"),
    n_estimators: int = typer.Option(600),
    max_depth: int = typer.Option(6),
    min_child_weight: float = typer.Option(1.0),
    learning_rate: float = typer.Option(0.05),
    subsample: float = typer.Option(0.8),
    colsample_bytree: float = typer.Option(0.8),
    reg_alpha: float = typer.Option(0.0),
    reg_lambda: float = typer.Option(1.0),
    n_jobs: int = typer.Option(0),
    n_trials: int = typer.Option(0, "--n-trials", "--n_trials", help="If >0, run Optuna tuning"),
    sampler: str = typer.Option("tpe"),
    seed: int = typer.Option(42),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    # validation
    cv_splits: int = typer.Option(0),
    cv_scheme: str = typer.Option("expanding"),
    cv_val_size: int = typer.Option(2000),
    perm_test: int = typer.Option(0),
):
    _impulse_train_impl(**locals())

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


@app.command()
def meta_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    # Model paths (defaults to stack dir)
    bot_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_bottom.json")),
    top_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_top.json")),
    logret_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_logret.json")),
    ae_path: str = typer.Option("", help="Optional AE manifest to append embeddings to meta inputs"),
    # Signal thresholds
    p_buy_thr: float = typer.Option(0.6),
    p_sell_thr: float = typer.Option(0.6),
    # Triple barrier
    pt_mult: float = typer.Option(2.0),
    sl_mult: float = typer.Option(1.0),
    max_holding: int = typer.Option(24),
    # MLP
    epochs: int = typer.Option(80),
    lr: float = typer.Option(1e-3),
    batch_size: int = typer.Option(512),
    device: str = typer.Option("auto"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
):
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        for tf in ["4h", "1d", "1w"]:
            try:
                _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
            except Exception:
                pass
    # Load data and features
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]
    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    bot_path = str(_coerce_opt(bot_path, str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_bottom.json")))
    top_path = str(_coerce_opt(top_path, str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_top.json")))
    logret_path = str(_coerce_opt(logret_path, str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_logret.json")))
    p_buy_thr = float(_coerce_opt(p_buy_thr, 0.6))
    p_sell_thr = float(_coerce_opt(p_sell_thr, 0.6))
    pt_mult = float(_coerce_opt(pt_mult, 2.0))
    sl_mult = float(_coerce_opt(sl_mult, 1.0))
    max_holding = int(_coerce_opt(max_holding, 24))
    epochs = int(_coerce_opt(epochs, 80))
    lr = float(_coerce_opt(lr, 1e-3))
    batch_size = int(_coerce_opt(batch_size, 512))
    device = str(_coerce_opt(device, "auto"))
    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None))
    feats = feats.reset_index(drop=True)
    if str(ae_path).strip():
        try:
            import json as _json
            with open(ae_path, "r") as _f:
                _man = _json.load(_f)
            if bool(_man.get("raw_htf", False)):
                ae_df = compute_embeddings_from_raw(raw, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae")
                ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
            else:
                ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=int(basic_lookback) if int(basic_lookback) > 0 else 128)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (Meta), proceeding without: {e}")
    T = len(feats)

    # Load TopBot models and predict probabilities
    bot_clf, bot_cols = _load_xgb(bot_path, device=device)
    top_clf, top_cols = _load_xgb(top_path, device=device)
    p_bot = _predict_with_cols(bot_clf, feats, bot_cols)
    p_top = _predict_with_cols(top_clf, feats, top_cols)
    p_bottom = p_bot[:, 1] if (p_bot is not None and p_bot.ndim == 2 and p_bot.shape[1] >= 2) else np.zeros(T)
    p_topp = p_top[:, 1] if (p_top is not None and p_top.ndim == 2 and p_top.shape[1] >= 2) else np.zeros(T)

    # Load Logret classifier and predict class probabilities
    logret_clf, logret_cols = _load_xgb(logret_path, device=device)
    pr = _predict_with_cols(logret_clf, feats, logret_cols)
    if pr is None or pr.ndim != 2 or pr.shape[1] != 5:
        pr = np.zeros((T, 5), dtype=float)
    # reg_direction proxy from probabilities
    class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    reg_dir = np.dot(pr, class_vals)

    # Assemble L0 outputs frame
    l0 = pd.DataFrame({
        "p_bottom": p_bottom,
        "p_top": p_topp,
        "p_up": np.zeros(T),
        "p_dn": np.zeros(T),
        "reg_direction": reg_dir,
        "logret_p_-2": pr[:, 0],
        "logret_p_-1": pr[:, 1],
        "logret_p_0": pr[:, 2],
        "logret_p_1": pr[:, 3],
        "logret_p_2": pr[:, 4],
    }, index=feats.index)

    # Build meta dataset using triple-barrier on primary signals
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    high = feats["high"].astype(float).to_numpy() if "high" in feats.columns else raw["high"].astype(float).to_numpy()
    low = feats["low"].astype(float).to_numpy() if "low" in feats.columns else raw["low"].astype(float).to_numpy()
    # ATR normalized from features; fallback compute
    if "atr" in feats.columns:
        atrn = feats["atr"].astype(float).to_numpy()
    else:
        tr1 = feats["high"].astype(float).to_numpy() - feats["low"].astype(float).to_numpy()
        tr2 = np.abs(feats["high"].astype(float).to_numpy() - np.roll(close, 1))
        tr3 = np.abs(feats["low"].astype(float).to_numpy() - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        tr[0] = tr1[0]
        atrn = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().to_numpy() / (close + 1e-12)

    buy_entries = (p_bottom >= float(p_buy_thr)).astype(int)
    sell_entries = (p_topp >= float(p_sell_thr)).astype(int)
    y_long, _ = triple_barrier_labels(close, high, low, atrn, entries=buy_entries, direction=+1, pt_mult=float(pt_mult), sl_mult=float(sl_mult), max_holding=int(max_holding))
    y_short, _ = triple_barrier_labels(close, high, low, atrn, entries=sell_entries, direction=-1, pt_mult=float(pt_mult), sl_mult=float(sl_mult), max_holding=int(max_holding))

    # Collect samples and build X,y
    rows: List[List[float]] = []
    labels: List[int] = []
    feat_cols = list(l0.columns)
    for i in range(T):
        if buy_entries[i] == 1 and y_long[i] >= 0:
            rows.append([float(l0.iloc[i][c]) for c in feat_cols])
            labels.append(int(y_long[i]))
        if sell_entries[i] == 1 and y_short[i] >= 0:
            rows.append([float(l0.iloc[i][c]) for c in feat_cols])
            labels.append(int(y_short[i]))
    if not rows:
        raise ValueError("No meta samples generated. Adjust thresholds/barrier params.")

    X_meta = pd.DataFrame(rows, columns=feat_cols)
    y_meta = np.asarray(labels, dtype=int)

    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "best_meta.json")
    res = train_meta_mlp_manager(X_meta, y_meta, out_json_path=_ensure_outdir(out_json), epochs=int(epochs), lr=float(lr), batch_size=int(batch_size), device=device)
    typer.echo(json.dumps(res, indent=2))


# -----------------------------
# Evaluation commands
# -----------------------------


@app.command("topbot-eval")
def topbot_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    bot_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_bottom.json")),
    top_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_top.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
):
    # Load data
    path = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange) or _find_data_file(userdir, pair, timeframe, prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found for evaluation.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf_str = str(_coerce_opt(extra_timeframes, "4H,1D,1W"))
    etf = [s.strip() for s in etf_str.split(",") if s.strip()]
    feats = make_features(
        raw,
        mode=_coerce_opt(feature_mode, "full"),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=(etf or None),
    )
    feats = feats.reset_index(drop=True)
    idx = feats.index
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    # Load models and predict
    bot_clf, bot_cols = _load_xgb(str(_coerce_opt(bot_path, bot_path)), device=str(_coerce_opt(device, "auto")))
    top_clf, top_cols = _load_xgb(str(_coerce_opt(top_path, top_path)), device=str(_coerce_opt(device, "auto")))
    p_bot = _predict_with_cols(bot_clf, feats, bot_cols)
    p_top = _predict_with_cols(top_clf, feats, top_cols)
    T = len(feats)
    p_bottom = p_bot[:, 1] if (p_bot is not None and p_bot.ndim == 2 and p_bot.shape[1] >= 2) else np.zeros(T)
    p_topp = p_top[:, 1] if (p_top is not None and p_top.ndim == 2 and p_top.shape[1] >= 2) else np.zeros(T)

    # Visuals
    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parent / "plot" / "xgb_eval"))), prefix="topbot")
    # Probabilities
    _plot_prob_series(idx, {"p_bottom": p_bottom, "p_top": p_topp}, thresholds=None, title=f"Top/Bottom Probabilities {pair} {timeframe}", out_path=os.path.join(root, "topbot_probs.png"))
    # Events at p>=0.6 as default view
    ev = {
        "bottom": (p_bottom >= 0.6).astype(int),
        "top": (p_topp >= 0.6).astype(int),
    }
    _plot_price_with_events(idx, close, ev, title=f"Price with Top/Bottom signals {pair} {timeframe}", out_path=os.path.join(root, "topbot_events.png"))
    # Feature importance
    _plot_feature_importance(bot_clf, bot_cols or list(feats.columns), out_path=os.path.join(root, "fi_bottom.png"), title="Feature importance - Bottom")
    _plot_feature_importance(top_clf, top_cols or list(feats.columns), out_path=os.path.join(root, "fi_top.png"), title="Feature importance - Top")
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))


@app.command("logret-eval")
def logret_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    logret_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_logret.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
):
    path = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange) or _find_data_file(userdir, pair, timeframe, prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found for evaluation.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf_str = str(_coerce_opt(extra_timeframes, "4H,1D,1W"))
    etf = [s.strip() for s in etf_str.split(",") if s.strip()]
    feats = make_features(
        raw,
        mode=_coerce_opt(feature_mode, "full"),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=(etf or None),
    )
    feats = feats.reset_index(drop=True)
    idx = feats.index
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()

    logret_clf, logret_cols = _load_xgb(str(_coerce_opt(logret_path, logret_path)), device=str(_coerce_opt(device, "auto")))
    pr = _predict_with_cols(logret_clf, feats, logret_cols)
    T = len(feats)
    if pr is None or pr.ndim != 2 or pr.shape[1] != 5:
        pr = np.zeros((T, 5), dtype=float)
    class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    reg_dir = pr @ class_vals

    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parent / "plot" / "xgb_eval"))), prefix="logret")
    _plot_prob_series(idx, {
        "p_-2": pr[:, 0],
        "p_-1": pr[:, 1],
        "p_0": pr[:, 2],
        "p_1": pr[:, 3],
        "p_2": pr[:, 4],
    }, thresholds=None, title=f"Logret class probabilities {pair} {timeframe}", out_path=os.path.join(root, "logret_probs.png"))
    # Direction proxy overlay
    _plot_regdir_shading(idx, close, reg_dir, title=f"Price with reg_direction shading {pair} {timeframe}", out_path=os.path.join(root, "logret_regdir.png"))
    # Class band with strong-signal markers
    _plot_logret_classes(idx, close, pr, classes=[-2,-1,0,1,2], title=f"Logret class band + strong signals {pair} {timeframe}", out_path=os.path.join(root, "logret_class_band.png"), strong_thr=0.55)
    _plot_feature_importance(logret_clf, logret_cols or list(feats.columns), out_path=os.path.join(root, "fi_logret.png"), title="Feature importance - Logret")
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))


@app.command("meta-eval")
def meta_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    bot_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_bottom.json")),
    top_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_topbot_top.json")),
    logret_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_logret.json")),
    meta_json_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_meta.json")),
    p_buy_thr: float = typer.Option(0.6),
    p_sell_thr: float = typer.Option(0.6),
    meta_thr: float = typer.Option(0.5),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    device: str = typer.Option("auto"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "plot" / "xgb_eval")),
):
    # Load data and features
    path = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange) or _find_data_file(userdir, pair, timeframe, prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found for evaluation.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf_str = str(_coerce_opt(extra_timeframes, "4H,1D,1W"))
    etf = [s.strip() for s in etf_str.split(",") if s.strip()]
    feats = make_features(
        raw,
        mode=_coerce_opt(feature_mode, "full"),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=(etf or None),
    ).reset_index(drop=True)
    idx = feats.index
    T = len(feats)
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()

    # Load L0 models
    bot_clf, bot_cols = _load_xgb(str(_coerce_opt(bot_path, bot_path)), device=str(_coerce_opt(device, "auto")))
    top_clf, top_cols = _load_xgb(str(_coerce_opt(top_path, top_path)), device=str(_coerce_opt(device, "auto")))
    logret_clf, logret_cols = _load_xgb(str(_coerce_opt(logret_path, logret_path)), device=str(_coerce_opt(device, "auto")))
    p_bot = _predict_with_cols(bot_clf, feats, bot_cols)
    p_top = _predict_with_cols(top_clf, feats, top_cols)
    pr = _predict_with_cols(logret_clf, feats, logret_cols)
    if pr is None or pr.ndim != 2 or pr.shape[1] != 5:
        pr = np.zeros((T, 5), dtype=float)
    p_bottom = p_bot[:, 1] if (p_bot is not None and p_bot.ndim == 2 and p_bot.shape[1] >= 2) else np.zeros(T)
    p_topp = p_top[:, 1] if (p_top is not None and p_top.ndim == 2 and p_top.shape[1] >= 2) else np.zeros(T)
    class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    reg_dir = pr @ class_vals

    # Build meta inputs in the same order as training manifest
    from rl_lib.meta import load_meta_mlp_from_json  # local import to avoid cycle
    meta_model, meta_cols = load_meta_mlp_from_json(str(_coerce_opt(meta_json_path, meta_json_path)))
    l0 = pd.DataFrame({
        "p_bottom": p_bottom,
        "p_top": p_topp,
        "p_up": np.zeros(T),
        "p_dn": np.zeros(T),
        "reg_direction": reg_dir,
        "logret_p_-2": pr[:, 0],
        "logret_p_-1": pr[:, 1],
        "logret_p_0": pr[:, 2],
        "logret_p_1": pr[:, 3],
        "logret_p_2": pr[:, 4],
    })
    for c in meta_cols:
        if c not in l0.columns:
            l0[c] = 0.0
    X_meta = l0.reindex(columns=meta_cols).values.astype(np.float32)
    try:
        import torch
        with torch.no_grad():
            logits = meta_model(torch.from_numpy(X_meta))
            meta_p = torch.sigmoid(logits).cpu().numpy()
    except Exception:
        meta_p = np.zeros((T,), dtype=float)

    # Define signals per simple gate
    buy_sig = (p_bottom >= float(p_buy_thr)) & (meta_p >= float(meta_thr))
    sell_sig = (p_topp >= float(p_sell_thr)) & (meta_p >= float(meta_thr))
    meta_score = (buy_sig.astype(int) - sell_sig.astype(int)).astype(int)  # +1 long, -1 short, 0 flat

    # Shift-aware horizon for sign accuracy
    H = _load_logret_horizon(str(_coerce_opt(logret_path, logret_path)))
    sign_acc = None
    try:
        # Forward returns over H bars
        logp = np.log(close + 1e-12)
        if H > 0 and H < len(close):
            fwd = np.zeros(T)
            fwd[:-H] = logp[H:] - logp[:-H]
            fwd_sign = np.sign(fwd)
            pred_sign = meta_score.copy()
            # Evaluate only where a non-zero prediction exists
            mask = pred_sign != 0
            hits = np.sum((fwd_sign[mask] * pred_sign[mask]) > 0)
            total = int(np.sum(mask))
            sign_acc = float(hits / max(1, total))
    except Exception:
        pass

    # Visuals
    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parent / "plot" / "xgb_eval"))), prefix="meta")
    # Meta probability series
    _plot_prob_series(idx, {"meta_p": meta_p, "p_bottom": p_bottom, "p_top": p_topp}, thresholds={"meta": meta_thr, "buy": p_buy_thr, "sell": p_sell_thr}, title=f"Meta score & primary probs {pair} {timeframe}", out_path=os.path.join(root, "meta_probs.png"))

    # Background shading from meta_score (+1/-1), aligned at prediction time
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
        fig, ax = plt.subplots(figsize=(14, 5))
        Tn = len(close)
        band = _np.clip(meta_score.astype(float), -1.0, 1.0).reshape(1, Tn)
        ax.imshow(band, aspect='auto', cmap='RdYlGn', alpha=0.18,
                  extent=[0, Tn, float(_np.nanmin(close)), float(_np.nanmax(close))])
        ax.plot(range(Tn), close, color='#1f77b4', linewidth=1.0)
        ax.set_title(f"Meta regime shading (+1 long / -1 short) {pair} {timeframe}")
        ax.grid(alpha=0.25)
        _safe_savefig(fig, os.path.join(root, "meta_regime.png"))
    except Exception:
        pass

    out_info = {"outdir": root, "samples": int(T)}
    if sign_acc is not None:
        out_info["sign_accuracy_over_H"] = {"H": int(H), "acc": float(sign_acc)}
    typer.echo(json.dumps(out_info, indent=2))


@app.command("impulse-eval")
def impulse_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    up_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_impulse_up.json")),
    dn_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "best_impulse_down.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
):
    path = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange) or _find_data_file(userdir, pair, timeframe, prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found for evaluation.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, timerange)
    etf_str = str(_coerce_opt(extra_timeframes, "4H,1D,1W"))
    etf = [s.strip() for s in etf_str.split(",") if s.strip()]
    feats = make_features(
        raw,
        mode=_coerce_opt(feature_mode, "full"),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=(etf or None),
    )
    feats = feats.reset_index(drop=True)
    idx = feats.index
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()

    up_clf, up_cols = _load_xgb(str(_coerce_opt(up_path, up_path)), device=str(_coerce_opt(device, "auto")))
    dn_clf, dn_cols = _load_xgb(str(_coerce_opt(dn_path, dn_path)), device=str(_coerce_opt(device, "auto")))
    p_up = _predict_with_cols(up_clf, feats, up_cols)
    p_dn = _predict_with_cols(dn_clf, feats, dn_cols)
    T = len(feats)
    p_up1 = p_up[:, 1] if (p_up is not None and p_up.ndim == 2 and p_up.shape[1] >= 2) else np.zeros(T)
    p_dn1 = p_dn[:, 1] if (p_dn is not None and p_dn.ndim == 2 and p_dn.shape[1] >= 2) else np.zeros(T)

    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parent / "plot" / "xgb_eval"))), prefix="impulse")
    _plot_prob_series(idx, {"p_up": p_up1, "p_dn": p_dn1}, thresholds=None, title=f"Impulse probabilities {pair} {timeframe}", out_path=os.path.join(root, "impulse_probs.png"))
    ev = {
        "up_sig": (p_up1 >= 0.6).astype(int),
        "dn_sig": (p_dn1 >= 0.6).astype(int),
    }
    _plot_price_with_events(idx, close, ev, title=f"Price with impulse signals {pair} {timeframe}", out_path=os.path.join(root, "impulse_events.png"))
    _plot_feature_importance(up_clf, up_cols or list(feats.columns), out_path=os.path.join(root, "fi_impulse_up.png"), title="Feature importance - Impulse Up")
    _plot_feature_importance(dn_clf, dn_cols or list(feats.columns), out_path=os.path.join(root, "fi_impulse_dn.png"), title="Feature importance - Impulse Down")
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))


@app.command()
def train_all(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack")),
    optuna_trials: int = typer.Option(40, "--optuna-trials", "--optuna_trials", help="Trials for Optuna when tuning XGB models"),
    ae_enable: bool = typer.Option(False, help="If true, train AE and include embeddings"),
    ae_out: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_stack" / "ae_conv1d.json"), help="AE manifest output path"),
    # global validation flags passed into sub-trainers
    cv_splits: int = typer.Option(0, help="If >0, run time-aware CV in sub-trainers"),
    cv_scheme: str = typer.Option("expanding"),
    cv_val_size: int = typer.Option(2000),
    perm_test: int = typer.Option(0, help="If >0, run permutation tests in sub-trainers"),
    auto_backtest: bool = typer.Option(True, "--auto-backtest", "--auto_backtest"),
    backtest_timerange: str = typer.Option("20240101-", "--backtest-timerange", "--backtest_timerange"),
):
    # Ensure datasets across TFs
    for tf in [timeframe, "4h", "1d", "1w"]:
        try:
            _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
        except Exception as e:
            typer.echo(f"Auto-download failed for {pair} {tf}: {e}")
    # Optional: Train AE
    ae_path_used = ""
    if bool(ae_enable):
        try:
            params = AETrainParams(
                pair=pair,
                timeframe=timeframe,
                userdir=userdir,
                timerange=timerange,
                prefer_exchange=prefer_exchange,
                feature_mode="full",
                basic_lookback=64,
                extra_timeframes="4H,1D,1W",
                raw_htf=True,
                raw_extra_timeframes="4H,1D,1W",
                ae_cols="close,volume",
                window=128,
                embed_dim=16,
                base_channels=32,
                batch_size=256,
                epochs=30,
                lr=1e-3,
                weight_decay=1e-5,
                device="auto",
                out_path=str(ae_out),
            )
            _ = train_autoencoder(params)
            ae_path_used = str(ae_out)
        except Exception as e:
            typer.echo(f"AE training failed, continuing without embeddings: {e}")

    # Train TopBot with Optuna
    topbot_train(
        pair=pair,
        timeframe=timeframe,
        userdir=userdir,
        timerange=timerange,
        prefer_exchange=prefer_exchange,
        extra_timeframes="4H,1D,1W",
        n_trials=int(optuna_trials),
        outdir=outdir,
        autodownload=False,
        cv_splits=int(cv_splits),
        cv_scheme=str(cv_scheme),
        cv_val_size=int(cv_val_size),
        perm_test=int(perm_test),
        ae_path=ae_path_used,
    )
    # Train Logret with Optuna
    logret_train(
        pair=pair,
        timeframe=timeframe,
        userdir=userdir,
        timerange=timerange,
        prefer_exchange=prefer_exchange,
        extra_timeframes="4H,1D,1W",
        n_trials=int(optuna_trials),
        outdir=outdir,
        autodownload=False,
        cv_splits=int(cv_splits),
        cv_scheme=str(cv_scheme),
        cv_val_size=int(cv_val_size),
        perm_test=int(perm_test),
        ae_path=ae_path_used,
    )
    # Train Impulse with Optuna
    try:
        impulse_train_cmd(
            pair=pair,
            timeframe=timeframe,
            userdir=userdir,
            timerange=timerange,
            prefer_exchange=prefer_exchange,
            extra_timeframes="4H,1D,1W",
            n_trials=int(optuna_trials),
            outdir=outdir,
            autodownload=False,
            cv_splits=int(cv_splits),
            cv_scheme=str(cv_scheme),
            cv_val_size=int(cv_val_size),
            perm_test=int(perm_test),
            ae_path=ae_path_used,
        )
    except Exception as e:
        typer.echo(f"Impulse training skipped/failed: {e}")
    # Train Meta (uses outputs)
    meta_train(
        pair=pair,
        timeframe=timeframe,
        userdir=userdir,
        timerange=timerange,
        prefer_exchange=prefer_exchange,
        extra_timeframes="4H,1D,1W",
        outdir=outdir,
        autodownload=False,
        ae_path=ae_path_used,
    )
    # Generate evaluation charts bundle
    try:
        bundle_root = _ts_outdir(str(Path(__file__).resolve().parent / "plot" / "xgb_eval"), prefix="bundle")
        # Run evals and then move their outputs under bundle_root
        import shutil
        # TopBot
        _tb_before = _ts_outdir(str(Path(__file__).resolve().parent / "plot" / "xgb_eval"), prefix="tmp_topbot")
        # re-evaluate into tmp dir by temporarily overriding outdir
        topbot_eval(
            pair=pair,
            timeframe=timeframe,
            userdir=userdir,
            timerange=timerange,
            prefer_exchange=prefer_exchange,
            bot_path=str(Path(outdir) / "best_topbot_bottom.json"),
            top_path=str(Path(outdir) / "best_topbot_top.json"),
            outdir=_tb_before,
        )
        # Logret
        _lr_before = _ts_outdir(str(Path(__file__).resolve().parent / "plot" / "xgb_eval"), prefix="tmp_logret")
        logret_eval(
            pair=pair,
            timeframe=timeframe,
            userdir=userdir,
            timerange=timerange,
            prefer_exchange=prefer_exchange,
            logret_path=str(Path(outdir) / "best_logret.json"),
            outdir=_lr_before,
        )
        # Impulse (optional)
        up_path = Path(outdir) / "best_impulse_up.json"
        dn_path = Path(outdir) / "best_impulse_down.json"
        if up_path.exists() and dn_path.exists():
            _im_before = _ts_outdir(str(Path(__file__).resolve().parent / "plot" / "xgb_eval"), prefix="tmp_impulse")
            impulse_eval(
                pair=pair,
                timeframe=timeframe,
                userdir=userdir,
                timerange=timerange,
                prefer_exchange=prefer_exchange,
                up_path=str(up_path),
                dn_path=str(dn_path),
                outdir=_im_before,
            )
            # move impulse dir into bundle
            for p in Path(_im_before).glob("**/*"):
                # ensure flat copy into bundle/impulse
                dest_dir = Path(bundle_root) / "impulse"
                dest_dir.mkdir(parents=True, exist_ok=True)
                if p.is_file():
                    shutil.copy2(str(p), str(dest_dir / p.name))
        # move topbot and logret into bundle
        for src_dir, name in [(_tb_before, "topbot"), (_lr_before, "logret")]:
            d = Path(bundle_root) / name
            d.mkdir(parents=True, exist_ok=True)
            for p in Path(src_dir).glob("**/*"):
                if p.is_file():
                    shutil.copy2(str(p), str(d / p.name))
        typer.echo(json.dumps({"charts_bundle": bundle_root}, indent=2))
    except Exception as e:
        typer.echo(f"Chart bundle generation failed: {e}")

    # Optional backtest validation
    if bool(auto_backtest):
        try:
            _run_backtest_stacked(
                userdir=userdir,
                pair=pair,
                timeframe=timeframe,
                timerange=backtest_timerange,
                model_dir=outdir,
                extra_tfs="4h,1d,1w",
                exchange=prefer_exchange,
            )
        except Exception as e:
            typer.echo(f"Auto backtest failed: {e}")


if __name__ == "__main__":
    app()



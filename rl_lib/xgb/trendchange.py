from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from rl_lib.features import make_features
from rl_lib.autoencoder import compute_embeddings, compute_embeddings_from_raw
from .common import (
    _ensure_dataset,
    _find_data_file,  # type: ignore
    _load_ohlcv,      # type: ignore
    _slice_timerange_df,  # type: ignore
    _coerce_opt,
    _save_feature_columns,
    _ensure_outdir,
    _train_xgb_classifier,
    _plot_prob_series,
    _plot_price_with_events,
    _plot_feature_importance,
    _save_feature_importance_text,
    _ts_outdir,
    _predict_with_cols,
    _label_pivots,
    _compute_atr_norm_fast,
    _label_trendchange_strong,
)


app = typer.Typer(add_completion=False)


@app.command("trendchange-train")
def trendchange_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    ae_path: str = typer.Option("", help="Optional path to AE manifest (.json) to append embeddings"),
    left_bars: int = typer.Option(3),
    right_bars: int = typer.Option(3),
    min_gap_bars: int = typer.Option(4),
    horizon: int = typer.Option(8, help="Bars ahead to look for strong pivot reaction"),
    label_mode: str = typer.Option("anchor", help="anchor|window: sparse anchor labels or window labels"),
    thr_mult: float = typer.Option(1.8, help="ATR-multiple for post-pivot reaction strength"),
    pre_h: int = typer.Option(16, help="Bars to measure pre-move build-up before pivot"),
    pre_mult: float = typer.Option(1.0, help="ATR-multiple for pre-move build-up"),
    anchor_offset: int = typer.Option(2, help="Bars before pivot to place positive label in anchor mode"),
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
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    cv_splits: int = typer.Option(0, help="If >0, run time-aware CV with given splits"),
    cv_scheme: str = typer.Option("expanding", help="expanding|rolling"),
    cv_val_size: int = typer.Option(2000, help="Validation fold size (bars)"),
    perm_test: int = typer.Option(0, help="If >0, run permutation test with N shuffles"),
):
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        for tf in ["4h", "1d", "1w"]:
            try:
                _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
            except Exception:
                pass
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)  # type: ignore
    if not path:
        raise FileNotFoundError("Dataset not found. Run download first.")
    raw = _load_ohlcv(path)  # type: ignore
    raw = _slice_timerange_df(raw, timerange)  # type: ignore
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]

    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    left_bars = int(_coerce_opt(left_bars, 3))
    right_bars = int(_coerce_opt(right_bars, 3))
    min_gap_bars = int(_coerce_opt(min_gap_bars, 4))
    horizon = int(_coerce_opt(horizon, 8))
    device = str(_coerce_opt(device, "auto"))

    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None)).reset_index(drop=True)
    if str(ae_path).strip():
        try:
            import json as _json
            with open(ae_path, "r") as _f:
                _man = _json.load(_f)
            if bool(_man.get("raw_htf", False)):
                ae_df = compute_embeddings_from_raw(raw, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae")
                ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
            else:
                ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=None)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (TrendChange), proceeding without: {e}")

    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    atrn = _compute_atr_norm_fast(feats, raw)
    yb, yt = _label_pivots(close, int(left_bars), int(right_bars), int(min_gap_bars))
    y_change = _label_trendchange_strong(
        close=close,
        atr_norm=atrn,
        y_top=yt,
        y_bot=yb,
        horizon=int(horizon),
        pre_h=int(pre_h),
        thr_mult=float(thr_mult),
        pre_mult=float(pre_mult),
        anchor_offset=int(anchor_offset),
        mode=str(label_mode),
    )

    T = len(feats)
    valid = max(0, T - int(horizon))
    if valid <= 400:
        raise ValueError("Not enough data to train TrendChange.")
    X = feats.iloc[:valid, :].copy()
    y = y_change[:valid]

    cut = int(max(200, min(valid - 50, int(valid * 0.8))))
    X_tr = X.iloc[:cut, :].values
    X_val = X.iloc[cut:, :].values
    y_tr = y[:cut]
    y_val = y[cut:]

    def _fit_eval(params: Dict[str, Any]) -> Tuple[Any, float]:
        clf, m = _train_xgb_classifier(
            X_tr, y_tr, X_val, y_val, device,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=int(n_jobs),
            objective="binary:logistic",
        )
        from sklearn.metrics import average_precision_score
        p = clf.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, p))
        return clf, ap

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
            _, ap = _fit_eval(params)
            return ap
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna TrendChange trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        clf, best_score = _fit_eval(best_params)
        typer.echo(f"Best TrendChange AUPRC: {best_score:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate), subsample=float(subsample), colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)
        )
        clf, _ = _fit_eval(base_params)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "best_trendchange.json")
    clf.save_model(_ensure_outdir(out_path))
    _save_feature_columns(out_path, list(feats.columns))


@app.command("trendchange-eval")
def trendchange_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    trendchange_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_trendchange.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
    p_thr: float = typer.Option(0.6, help="Threshold for marking predicted trend changes"),
    min_gap_bars: int = typer.Option(8, help="Min separation between displayed events (peak picking)"),
    peak_window: int = typer.Option(3, help="Local max window for eventization of probabilities"),
    candles: bool = typer.Option(False, help="Plot OHLC candles instead of line"),
):
    path = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange) or _find_data_file(userdir, pair, timeframe, prefer_exchange)  # type: ignore
    if not path:
        raise FileNotFoundError("Dataset not found for evaluation.")
    raw = _load_ohlcv(path)  # type: ignore
    raw = _slice_timerange_df(raw, timerange)  # type: ignore
    etf = [s.strip() for s in str(_coerce_opt(extra_timeframes, "4H,1D,1W")).split(",") if s.strip()]
    feats = make_features(
        raw,
        mode=_coerce_opt(feature_mode, "full"),
        basic_lookback=int(_coerce_opt(basic_lookback, 64)),
        extra_timeframes=(etf or None),
    ).reset_index(drop=True)
    idx = feats.index
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()

    from .common import _load_xgb
    clf, cols = _load_xgb(str(_coerce_opt(trendchange_path, trendchange_path)), device=str(_coerce_opt(device, "auto")))
    pr = _predict_with_cols(clf, feats, cols)
    T = len(feats)
    p_change = pr[:, 1] if (pr is not None and pr.ndim == 2 and pr.shape[1] >= 2) else np.zeros(T)

    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval"))), prefix="trendchange")
    thr_val = float(_coerce_opt(p_thr, 0.6))
    _plot_prob_series(idx, {"p_change": p_change}, thresholds={"thr": thr_val}, title=f"Trend change probability {pair} {timeframe}", out_path=os.path.join(root, "trendchange_probs.png"))
    try:
        gap = int(_coerce_opt(min_gap_bars, 8))
        win = int(max(1, _coerce_opt(peak_window, 3)))
        Tn = len(p_change)
        peaks = np.zeros(Tn, dtype=int)
        last = -10**9
        for i in range(Tn):
            j0 = max(0, i - win)
            j1 = min(Tn, i + win + 1)
            pc = p_change[i]
            if pc >= thr_val and i - last >= gap and pc == np.max(p_change[j0:j1]):
                peaks[i] = 1
                last = i
        slope = np.gradient(close.astype(float))
        tc_up = np.zeros(Tn, dtype=int)
        tc_dn = np.zeros(Tn, dtype=int)
        for i in np.flatnonzero(peaks == 1):
            prev_idx = max(0, i-1)
            next_idx = min(Tn-1, i+1)
            d = close[next_idx] - close[prev_idx]
            if d >= 0:
                tc_up[i] = 1
            else:
                tc_dn[i] = 1
        ev = {"tc_up": tc_up, "tc_dn": tc_dn}
    except Exception:
        mask = (p_change >= thr_val).astype(int)
        Tn = len(mask)
        tc_up = np.zeros(Tn, dtype=int)
        tc_dn = np.zeros(Tn, dtype=int)
        for i in range(Tn):
            if mask[i] == 1:
                prev_idx = max(0, i-1)
                next_idx = min(Tn-1, i+1)
                d = close[next_idx] - close[prev_idx]
                if d >= 0:
                    tc_up[i] = 1
                else:
                    tc_dn[i] = 1
        ev = {"tc_up": tc_up, "tc_dn": tc_dn}
    o = feats["open"].astype(float).to_numpy() if "open" in feats.columns else raw["open"].astype(float).to_numpy()
    hi = feats["high"].astype(float).to_numpy() if "high" in feats.columns else raw["high"].astype(float).to_numpy()
    lo = feats["low"].astype(float).to_numpy() if "low" in feats.columns else raw["low"].astype(float).to_numpy()
    _plot_price_with_events(idx, close, ev, title=f"Price with predicted trend changes {pair} {timeframe}", out_path=os.path.join(root, "trendchange_events.png"), o=o, h=hi, l=lo, use_candles=bool(_coerce_opt(candles, False)))
    _plot_feature_importance(clf, cols or list(feats.columns), out_path=os.path.join(root, "fi_trendchange.png"), title="Feature importance - TrendChange")
    try:
        _save_feature_importance_text(clf, cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_trendchange.txt"), top_k=500)
    except Exception:
        pass
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))



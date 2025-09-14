from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from rl_lib.features import make_features
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
)
from rl_lib.autoencoder import compute_embeddings, compute_embeddings_from_raw


app = typer.Typer(add_completion=False)


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


@app.command("topbot-train")
def topbot_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
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
            typer.echo(f"AE embeddings failed (TopBot), proceeding without: {e}")

    c = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    yb, yt = _label_pivots(c, int(left_bars), int(right_bars), int(min_gap_bars))
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
            n_jobs=int(n_jobs),
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
            n_jobs=int(n_jobs),
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

    # Validation metrics
    try:
        from sklearn.metrics import average_precision_score as _aps
        p_b = bot_model.predict_proba(X_val)[:, 1]
        p_t = top_model.predict_proba(X_val)[:, 1]
        m_bot = {"auprc": float(_aps(yb_val, p_b))}
        m_top = {"auprc": float(_aps(yt_val, p_t))}
    except Exception:
        m_bot = {}; m_top = {}
    typer.echo(json.dumps({"bottom": {"path": bot_path, **m_bot}, "top": {"path": top_path, **m_top}}, indent=2))


@app.command("topbot-eval")
def topbot_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    bot_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_topbot_bottom.json")),
    top_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_topbot_top.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
    candles: bool = typer.Option(False, help="Plot OHLC candles instead of line"),
    bot_thr: float = typer.Option(0.6, help="Probability threshold for bottom trigger"),
    top_thr: float = typer.Option(0.6, help="Probability threshold for top trigger"),
    autodownload: bool = typer.Option(True, help="Auto-download required datasets including HTFs"),
):
    if autodownload:
        _ = _ensure_dataset(userdir, pair, timeframe, prefer_exchange, timerange)
        try:
            etf_opt = str(_coerce_opt(extra_timeframes, "4H,1D,1W"))
            for tf in [s.strip().lower() for s in etf_opt.split(",") if s.strip()]:
                try:
                    _ensure_dataset(userdir, pair, tf, prefer_exchange, timerange)
                except Exception:
                    pass
        except Exception:
            pass
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
    bot_clf, bot_cols = _load_xgb(str(_coerce_opt(bot_path, bot_path)), device=str(_coerce_opt(device, "auto")))
    top_clf, top_cols = _load_xgb(str(_coerce_opt(top_path, top_path)), device=str(_coerce_opt(device, "auto")))
    p_bot = _predict_with_cols(bot_clf, feats, bot_cols)
    p_top = _predict_with_cols(top_clf, feats, top_cols)
    T = len(feats)
    p_bottom = p_bot[:, 1] if (p_bot is not None and p_bot.ndim == 2 and p_bot.shape[1] >= 2) else np.zeros(T)
    p_topp = p_top[:, 1] if (p_top is not None and p_top.ndim == 2 and p_top.shape[1] >= 2) else np.zeros(T)

    bt = float(_coerce_opt(bot_thr, 0.6))
    tt = float(_coerce_opt(top_thr, 0.6))
    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval"))), prefix="topbot")
    _plot_prob_series(idx, {"p_bottom": p_bottom, "p_top": p_topp}, thresholds={"p_bottom": bt, "p_top": tt}, title=f"Top/Bottom Probabilities {pair} {timeframe}", out_path=os.path.join(root, "topbot_probs.png"))
    ev = {
        "bottom": (p_bottom >= bt).astype(int),
        "top": (p_topp >= tt).astype(int),
    }
    o = feats["open"].astype(float).to_numpy() if "open" in feats.columns else raw["open"].astype(float).to_numpy()
    hi = feats["high"].astype(float).to_numpy() if "high" in feats.columns else raw["high"].astype(float).to_numpy()
    lo = feats["low"].astype(float).to_numpy() if "low" in feats.columns else raw["low"].astype(float).to_numpy()
    _plot_price_with_events(idx, close, ev, title=f"Price with Top/Bottom signals {pair} {timeframe}", out_path=os.path.join(root, "topbot_events.png"), o=o, h=hi, l=lo, use_candles=bool(_coerce_opt(candles, False)))
    _plot_feature_importance(bot_clf, bot_cols or list(feats.columns), out_path=os.path.join(root, "fi_bottom.png"), title="Feature importance - Bottom")
    _plot_feature_importance(top_clf, top_cols or list(feats.columns), out_path=os.path.join(root, "fi_top.png"), title="Feature importance - Top")
    try:
        _save_feature_importance_text(bot_clf, bot_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_bottom.txt"), top_k=500)
        _save_feature_importance_text(top_clf, top_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_top.txt"), top_k=500)
    except Exception:
        pass
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))




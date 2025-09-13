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
    _plot_regdir_shading,
    _plot_logret_classes,
    _ts_outdir,
    _predict_with_cols,
)


app = typer.Typer(add_completion=False)


@app.command("logret-train")
def logret_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
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
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
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
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)  # type: ignore
    if not path:
        raise FileNotFoundError("Dataset not found. Run download first.")
    raw = _load_ohlcv(path)  # type: ignore
    raw = _slice_timerange_df(raw, timerange)  # type: ignore
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]

    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
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
    raw_labels = np.zeros(valid, dtype=int)
    thr = float(str(strong_mult)) * atr
    raw_labels[fwd > thr] = 2
    raw_labels[(fwd > 0.0) & (fwd <= thr)] = 1
    raw_labels[(fwd < 0.0) & (fwd >= -thr)] = -1
    raw_labels[fwd < -thr] = -2
    classes_ord = np.array([-2, -1, 0, 1, 2], dtype=int)
    label_to_idx = {c: i for i, c in enumerate(classes_ord)}
    labels = np.vectorize(lambda x: label_to_idx.get(int(x), 2))(raw_labels).astype(int)
    X = feats.iloc[:valid, :].copy()
    cut = int(max(100, min(valid - 50, int(valid * 0.8))))
    X_tr = X.iloc[:cut, :].values
    X_val = X.iloc[cut:, :].values
    y_tr = labels[:cut]
    y_val = labels[cut:]

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
            objective="multi:softprob",
            num_class=5,
        )
        from sklearn.metrics import accuracy_score
        pr = clf.predict_proba(X_val)
        y_hat = np.argmax(pr, axis=1)
        acc = float(accuracy_score(y_val, y_hat))
        return clf, acc

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
            _, score = _fit_eval(params)
            return score
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna Logret trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        model, best_score = _fit_eval(best_params)
        typer.echo(f"Best Logret acc: {best_score:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators), max_depth=int(max_depth), min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate), subsample=float(subsample), colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda)
        )
        model, _ = _fit_eval(base_params)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "best_logret.json")
    model.save_model(_ensure_outdir(out_path))
    _save_feature_columns(out_path, list(X.columns))
    with open(str(Path(out_path).with_suffix("").as_posix()) + "_meta.json", "w") as f:
        json.dump({"horizon": int(horizon)}, f)


@app.command("logret-eval")
def logret_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    logret_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_logret.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
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
    logret_clf, logret_cols = _load_xgb(str(_coerce_opt(logret_path, logret_path)), device=str(_coerce_opt(device, "auto")))
    pr = _predict_with_cols(logret_clf, feats, logret_cols)
    T = len(feats)
    if pr is None or pr.ndim != 2 or pr.shape[1] != 5:
        pr = np.zeros((T, 5), dtype=float)
    class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    reg_dir = pr @ class_vals

    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval"))), prefix="logret")
    _plot_prob_series(idx, {
        "p_-2": pr[:, 0],
        "p_-1": pr[:, 1],
        "p_0": pr[:, 2],
        "p_1": pr[:, 3],
        "p_2": pr[:, 4],
    }, thresholds=None, title=f"Logret class probabilities {pair} {timeframe}", out_path=os.path.join(root, "logret_probs.png"))
    _plot_regdir_shading(idx, close, reg_dir, title=f"Price with reg_direction shading {pair} {timeframe}", out_path=os.path.join(root, "logret_regdir.png"))
    _plot_logret_classes(idx, close, pr, classes=[-2,-1,0,1,2], title=f"Logret class band + strong signals {pair} {timeframe}", out_path=os.path.join(root, "logret_class_band.png"), strong_thr=0.55)
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))



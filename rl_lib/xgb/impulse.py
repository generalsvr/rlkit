from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from rl_lib.features import make_features
from rl_lib.autoencoder import compute_embeddings
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


app = typer.Typer(add_completion=False)


def _impulse_labels(logp_s: pd.Series, horizon: int, mode: str, alpha_up: float, alpha_dn: float, vol_lookback: int, thr_up_bps: float, thr_dn_bps: float) -> Tuple[np.ndarray, np.ndarray, int]:
    H = int(max(1, horizon))
    T = len(logp_s)
    valid = T - H
    fwd = [logp_s.shift(-i) - logp_s for i in range(1, H + 1)]
    fwd_mat = np.vstack([s.to_numpy() for s in fwd])
    fwd_valid = fwd_mat[:, :valid]
    fwd_max = np.nanmax(fwd_valid, axis=0)
    fwd_min = np.nanmin(fwd_valid, axis=0)
    logret = logp_s.diff().fillna(0.0).to_numpy()
    sigma = pd.Series(logret).rolling(int(vol_lookback), min_periods=20).std().fillna(0.0).to_numpy()[:valid]
    if mode == "vol":
        thr = sigma * np.sqrt(max(1, H))
        y_up = (fwd_max > (alpha_up * thr)).astype(int)
        y_dn = (fwd_min < (-alpha_dn * thr)).astype(int)
    else:
        bps_to_lr = 1e-4
        y_up = (fwd_max > (thr_up_bps * bps_to_lr)).astype(int)
        y_dn = (fwd_min < (-(thr_dn_bps * bps_to_lr))).astype(int)
    return y_up, y_dn, valid


@app.command("impulse-train")
def impulse_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[3] / "freqtrade_userdir")),
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
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[3] / "models" / "xgb_stack")),
    autodownload: bool = typer.Option(True),
    cv_splits: int = typer.Option(0),
    cv_scheme: str = typer.Option("expanding"),
    cv_val_size: int = typer.Option(2000),
    perm_test: int = typer.Option(0),
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
        raise FileNotFoundError("Dataset not found.")
    raw = _load_ohlcv(path)  # type: ignore
    raw = _slice_timerange_df(raw, timerange)  # type: ignore
    etf = [s.strip() for s in extra_timeframes.split(",") if s.strip()]

    feature_mode = _coerce_opt(feature_mode, "full")
    basic_lookback = int(_coerce_opt(basic_lookback, 64))
    horizon = int(_coerce_opt(horizon, 8))
    label_mode = str(_coerce_opt(label_mode, "vol")).lower()
    device = str(_coerce_opt(device, "auto"))

    feats = make_features(raw, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=(etf or None)).reset_index(drop=True)
    if str(ae_path).strip():
        try:
            ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device), out_col_prefix="ae", window=int(basic_lookback) if int(basic_lookback) > 0 else 128)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (Impulse), proceeding without: {e}")
    close = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    logp_s = pd.Series(np.log(close + 1e-12), index=feats.index)
    y_up, y_dn, valid = _impulse_labels(logp_s, horizon, label_mode, float(alpha_up), float(alpha_dn), int(vol_lookback), float(thr_up_bps), float(thr_dn_bps))

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
            reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=int(n_jobs),
            objective="binary:logistic",
        )
        dn_m, md = _train_xgb_classifier(
            X_tr, y_dn_tr, X_val, y_dn_val, device,
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]), learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=int(n_jobs),
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


@app.command("impulse-eval")
def impulse_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[3] / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    up_path: str = typer.Option(str(Path(__file__).resolve().parents[3] / "models" / "xgb_stack" / "best_impulse_up.json")),
    dn_path: str = typer.Option(str(Path(__file__).resolve().parents[3] / "models" / "xgb_stack" / "best_impulse_down.json")),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parents[3] / "plot" / "xgb_eval")),
    device: str = typer.Option("auto"),
    up_thr: float = typer.Option(0.6, help="Probability threshold for up trigger"),
    dn_thr: float = typer.Option(0.6, help="Probability threshold for down trigger"),
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
    up_clf, up_cols = _load_xgb(str(_coerce_opt(up_path, up_path)), device=str(_coerce_opt(device, "auto")))
    dn_clf, dn_cols = _load_xgb(str(_coerce_opt(dn_path, dn_path)), device=str(_coerce_opt(device, "auto")))
    p_up = _predict_with_cols(up_clf, feats, up_cols)
    p_dn = _predict_with_cols(dn_clf, feats, dn_cols)
    T = len(feats)
    p_up1 = p_up[:, 1] if (p_up is not None and p_up.ndim == 2 and p_up.shape[1] >= 2) else np.zeros(T)
    p_dn1 = p_dn[:, 1] if (p_dn is not None and p_dn.ndim == 2 and p_dn.shape[1] >= 2) else np.zeros(T)

    ut = float(_coerce_opt(up_thr, 0.6))
    dt = float(_coerce_opt(dn_thr, 0.6))
    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parents[3] / "plot" / "xgb_eval"))), prefix="impulse")
    _plot_prob_series(idx, {"p_up": p_up1, "p_dn": p_dn1}, thresholds={"p_up": ut, "p_dn": dt}, title=f"Impulse probabilities {pair} {timeframe}", out_path=os.path.join(root, "impulse_probs.png"))
    ev = {
        "up_sig": (p_up1 >= ut).astype(int),
        "dn_sig": (p_dn1 >= dt).astype(int),
    }
    _plot_price_with_events(idx, close, ev, title=f"Price with impulse signals {pair} {timeframe}", out_path=os.path.join(root, "impulse_events.png"))
    _plot_feature_importance(up_clf, up_cols or list(feats.columns), out_path=os.path.join(root, "fi_impulse_up.png"), title="Feature importance - Impulse Up")
    _plot_feature_importance(dn_clf, dn_cols or list(feats.columns), out_path=os.path.join(root, "fi_impulse_dn.png"), title="Feature importance - Impulse Down")
    try:
        _save_feature_importance_text(up_clf, up_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_impulse_up.txt"), top_k=500)
        _save_feature_importance_text(dn_clf, dn_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_impulse_dn.txt"), top_k=500)
    except Exception:
        pass
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))



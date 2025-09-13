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
)


app = typer.Typer(add_completion=False)


def _label_global_cycle_extrema(
    close: np.ndarray,
    window: int,
    horizon: int,
    down_frac: float,
    up_frac: float,
    breakout_eps: float,
    min_persist: int,
    min_separation: int,
    smooth_window: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(close, dtype=float)
    T = int(len(c))
    w = int(max(1, window))
    H = int(max(1, horizon))
    L = int(max(0, min_persist))
    sep = int(max(1, min_separation))
    eps = float(max(0.0, breakout_eps))

    x = np.log(c + 1e-12)
    ref = x.copy()
    if int(smooth_window) > 0:
        try:
            r = pd.Series(x).rolling(int(smooth_window), center=True, min_periods=1).mean().to_numpy()
            ref = np.where(np.isfinite(r), r, x)
        except Exception:
            ref = x

    y_top = np.zeros(T, dtype=int)
    y_bot = np.zeros(T, dtype=int)
    top_amp = np.zeros(T, dtype=float)
    bot_amp = np.zeros(T, dtype=float)

    valid = max(0, T - H)
    for t in range(valid):
        l0 = max(0, t - w)
        r1 = min(T, t + w + 1)
        rv = ref[l0:r1]
        if rv.size == 0:
            continue
        f1 = t + 1
        f2 = min(T, t + H + 1)
        if f1 >= f2:
            continue
        c0 = float(c[t])
        seg = c[f1:f2]
        mn = float(np.min(seg))
        mx = float(np.max(seg))

        if ref[t] == np.max(rv):
            drawdown = (c0 - mn) / (c0 + 1e-12)
            breakout = (mx - c0) / (c0 + 1e-12)
            if (drawdown >= float(down_frac)) and (breakout <= eps):
                thr_price = c0 * (1.0 - float(down_frac))
                hit_idx_rel = None
                for k, v in enumerate(seg, start=1):
                    if v <= thr_price:
                        hit_idx_rel = k
                        break
                if hit_idx_rel is not None and int(hit_idx_rel) >= L:
                    y_top[t] = 1
                    top_amp[t] = float(drawdown)

        if ref[t] == np.min(rv):
            rise = (mx - c0) / (c0 + 1e-12)
            breakdown = (c0 - mn) / (c0 + 1e-12)
            if (rise >= float(up_frac)) and (breakdown <= eps):
                thr_price = c0 * (1.0 + float(up_frac))
                hit_idx_rel = None
                for k, v in enumerate(seg, start=1):
                    if v >= thr_price:
                        hit_idx_rel = k
                        break
                if hit_idx_rel is not None and int(hit_idx_rel) >= L:
                    y_bot[t] = 1
                    bot_amp[t] = float(rise)

    amb = (y_top == 1) & (y_bot == 1)
    y_top[amb] = 0
    y_bot[amb] = 0

    def _nms(mask: np.ndarray, strength: np.ndarray, min_gap: int) -> np.ndarray:
        idx = np.flatnonzero(mask == 1)
        if idx.size == 0:
            return mask
        kept: List[int] = []
        for i in idx:
            if not kept:
                kept.append(int(i))
                continue
            if int(i - kept[-1]) > min_gap:
                kept.append(int(i))
            else:
                j = kept[-1]
                if float(strength[i]) > float(strength[j]):
                    kept[-1] = int(i)
        out = np.zeros_like(mask)
        out[np.asarray(kept, dtype=int)] = 1
        return out

    y_top = _nms(y_top, top_amp, sep)
    y_bot = _nms(y_bot, bot_amp, sep)
    return y_bot, y_top


@app.command("globalcycle-train")
def globalcycle_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("4H,1D,1W"),
    ae_path: str = typer.Option("", help="Optional AE manifest (.json) to append embeddings"),
    window: int = typer.Option(14, help="Local extremum window (bars)"),
    horizon: int = typer.Option(200, help="Look-ahead horizon (bars)"),
    down_frac: float = typer.Option(0.30, help="Required future drawdown from top (fraction)"),
    up_frac: float = typer.Option(0.30, help="Required future rise from bottom (fraction)"),
    breakout_eps: float = typer.Option(0.01, help="Tolerance for breakout against the turn (fraction)"),
    min_persist: int = typer.Option(20, help="Minimum bars until threshold hit"),
    min_separation: int = typer.Option(45, help="Minimum bars between same-type events"),
    smooth_window: int = typer.Option(0, help="Centered smoothing window for local extrema (0=off)"),
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
    tune_labels: bool = typer.Option(True, help="When tuning, also search labeling params"),
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

    feature_mode_v = _coerce_opt(feature_mode, "full")
    basic_lookback_v = int(_coerce_opt(basic_lookback, 64))
    device_v = str(_coerce_opt(device, "auto"))

    feats = make_features(raw, mode=feature_mode_v, basic_lookback=basic_lookback_v, extra_timeframes=(etf or None)).reset_index(drop=True)
    if str(ae_path).strip():
        try:
            import json as _json
            with open(ae_path, "r") as _f:
                _man = _json.load(_f)
            if bool(_man.get("raw_htf", False)):
                ae_df = compute_embeddings_from_raw(raw, ae_manifest_path=str(ae_path), device=str(device_v), out_col_prefix="ae")
                ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
            else:
                ae_df = compute_embeddings(feats, ae_manifest_path=str(ae_path), device=str(device_v), out_col_prefix="ae", window=None)
            feats = feats.join(ae_df, how="left")
        except Exception as e:
            typer.echo(f"AE embeddings failed (GlobalCycle), proceeding without: {e}")

    c = feats["close"].astype(float).to_numpy() if "close" in feats.columns else raw["close"].astype(float).to_numpy()
    yb_full, yt_full = _label_global_cycle_extrema(
        c,
        window=int(window),
        horizon=int(horizon),
        down_frac=float(down_frac),
        up_frac=float(up_frac),
        breakout_eps=float(breakout_eps),
        min_persist=int(min_persist),
        min_separation=int(min_separation),
        smooth_window=int(smooth_window),
    )

    T = len(feats)
    valid_len = max(0, T - int(horizon))
    if valid_len < 300:
        raise ValueError("Not enough data to train GlobalCycle.")
    X = feats.iloc[:valid_len, :]
    yb = yb_full[:valid_len]
    yt = yt_full[:valid_len]

    cut = int(max(150, min(valid_len - 50, int(valid_len * 0.8))))
    X_tr = X.iloc[:cut, :].values
    X_val = X.iloc[cut:, :].values
    yb_tr, yb_val = yb[:cut], yb[cut:]
    yt_tr, yt_val = yt[:cut], yt[cut:]

    def _fit_eval(params: Dict[str, Any]) -> Tuple[Any, Any, float]:
        bot_m, m1 = _train_xgb_classifier(
            X_tr, yb_tr, X_val, yb_val, device_v,
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
            X_tr, yt_tr, X_val, yt_val, device_v,
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
            if bool(tune_labels):
                # sample label parameters around seeds
                win_t = trial.suggest_int("window", max(8, int(window) - 8), min(60, int(window) + 12))
                hor_t = trial.suggest_int("horizon", max(50, int(horizon * 0.5)), min(2880, int(horizon * 2)))
                dfrac_t = trial.suggest_float("down_frac", max(0.10, float(down_frac) - 0.10), min(0.50, float(down_frac) + 0.15))
                ufrac_t = trial.suggest_float("up_frac", max(0.10, float(up_frac) - 0.10), min(0.50, float(up_frac) + 0.15))
                beps_t = trial.suggest_float("breakout_eps", 0.003, 0.02)
                mp_t = trial.suggest_int("min_persist", max(10, int(min_persist) - 10), min(60, int(min_persist) + 20))
                ms_t = trial.suggest_int("min_separation", max(24, int(min_separation) - 48), min(240, int(min_separation) + 96))
                sm_t = trial.suggest_int("smooth_window", 0, max(144, int(smooth_window) + 72))
                # rebuild labels
                yb_t, yt_t = _label_global_cycle_extrema(
                    c,
                    window=int(win_t), horizon=int(hor_t),
                    down_frac=float(dfrac_t), up_frac=float(ufrac_t),
                    breakout_eps=float(beps_t), min_persist=int(mp_t),
                    min_separation=int(ms_t), smooth_window=int(sm_t),
                )
                valid_t = max(0, len(feats) - int(hor_t))
                if valid_t < 300:
                    return float('-inf')
                X_t = feats.iloc[:valid_t, :]
                cut_t = int(max(150, min(valid_t - 50, int(valid_t * 0.8))))
                X_tr_t = X_t.iloc[:cut_t, :].values
                X_va_t = X_t.iloc[cut_t:, :].values
                yb_tr_t, yb_va_t = yb_t[:cut_t], yb_t[cut_t:]
                yt_tr_t, yt_va_t = yt_t[:cut_t], yt_t[cut_t:]
                bm, m1 = _train_xgb_classifier(
                    X_tr_t, yb_tr_t, X_va_t, yb_va_t, device_v,
                    n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                    min_child_weight=float(params["min_child_weight"]), learning_rate=float(params["learning_rate"]),
                    subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
                    reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=int(n_jobs),
                    objective="binary:logistic",
                )
                tm, m2 = _train_xgb_classifier(
                    X_tr_t, yt_tr_t, X_va_t, yt_va_t, device_v,
                    n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                    min_child_weight=float(params["min_child_weight"]), learning_rate=float(params["learning_rate"]),
                    subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
                    reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]), n_jobs=int(n_jobs),
                    objective="binary:logistic",
                )
                from sklearn.metrics import average_precision_score as _aps
                try:
                    ap_b = float(_aps(yb_va_t, bm.predict_proba(X_va_t)[:, 1]))
                    ap_t = float(_aps(yt_va_t, tm.predict_proba(X_va_t)[:, 1]))
                except Exception:
                    ap_b = float('nan'); ap_t = float('nan')
                score = float(np.nanmean([ap_b, ap_t]))
            else:
                _, _, score = _fit_eval(params)
            return score
            return score
        study = optuna.create_study(direction="maximize", sampler=smp)
        typer.echo(f"Optuna GlobalCycle trials: {n_trials}")
        study.optimize(objective, n_trials=int(n_trials))
        best_params = study.best_params
        # propagate best label params if tuned
        if bool(tune_labels):
            window = int(best_params.get("window", window))
            horizon = int(best_params.get("horizon", horizon))
            down_frac = float(best_params.get("down_frac", down_frac))
            up_frac = float(best_params.get("up_frac", up_frac))
            breakout_eps = float(best_params.get("breakout_eps", breakout_eps))
            min_persist = int(best_params.get("min_persist", min_persist))
            min_separation = int(best_params.get("min_separation", min_separation))
            smooth_window = int(best_params.get("smooth_window", smooth_window))
            yb_full, yt_full = _label_global_cycle_extrema(
                c,
                window=int(window), horizon=int(horizon),
                down_frac=float(down_frac), up_frac=float(up_frac),
                breakout_eps=float(breakout_eps), min_persist=int(min_persist),
                min_separation=int(min_separation), smooth_window=int(smooth_window),
            )
            valid_len = max(0, len(feats) - int(horizon))
            X = feats.iloc[:valid_len, :]
            yb = yb_full[:valid_len]
            yt = yt_full[:valid_len]
            cut = int(max(150, min(valid_len - 50, int(valid_len * 0.8))))
            X_tr = X.iloc[:cut, :].values
            X_val = X.iloc[cut:, :].values
            yb_tr, yb_val = yb[:cut], yb[cut:]
            yt_tr, yt_val = yt[:cut], yt[cut:]
        bot_model, top_model, best_score = _fit_eval(best_params)
        typer.echo(f"Best GlobalCycle mean AUPRC: {best_score:.6f}")
    else:
        base_params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_child_weight=float(min_child_weight),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
        )
        bot_model, top_model, _ = _fit_eval(base_params)

    os.makedirs(outdir, exist_ok=True)
    bot_path = os.path.join(outdir, "best_global_bottom.json")
    top_path = os.path.join(outdir, "best_global_top.json")
    bot_model.save_model(_ensure_outdir(bot_path))
    top_model.save_model(_ensure_outdir(top_path))
    _save_feature_columns(bot_path, list(feats.columns))
    _save_feature_columns(top_path, list(feats.columns))
    # Save label meta for eval reproducibility
    try:
        meta = {
            "window": int(window),
            "horizon": int(horizon),
            "down_frac": float(down_frac),
            "up_frac": float(up_frac),
            "breakout_eps": float(breakout_eps),
            "min_persist": int(min_persist),
            "min_separation": int(min_separation),
            "smooth_window": int(smooth_window),
        }
        with open(os.path.join(outdir, "globalcycle_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    try:
        from sklearn.metrics import average_precision_score as _aps
        p_b = bot_model.predict_proba(X_val)[:, 1]
        p_t = top_model.predict_proba(X_val)[:, 1]
        m_bot = {"auprc": float(_aps(yb_val, p_b))}
        m_top = {"auprc": float(_aps(yt_val, p_t))}
    except Exception:
        m_bot = {}
        m_top = {}
    typer.echo(json.dumps({"bottom": {"path": bot_path, **m_bot}, "top": {"path": top_path, **m_top}}, indent=2))


@app.command("globalcycle-eval")
def globalcycle_eval(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parents[2] / "freqtrade_userdir")),
    timerange: str = typer.Option("20240101-"),
    prefer_exchange: str = typer.Option("bybit", "--prefer-exchange", "--prefer_exchange"),
    bot_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_global_bottom.json")),
    top_path: str = typer.Option(str(Path(__file__).resolve().parents[2] / "models" / "xgb_stack" / "best_global_top.json")),
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
    bot_clf, bot_cols = _load_xgb(str(_coerce_opt(bot_path, bot_path)), device=str(_coerce_opt(device, "auto")))
    top_clf, top_cols = _load_xgb(str(_coerce_opt(top_path, top_path)), device=str(_coerce_opt(device, "auto")))
    p_bot = _predict_with_cols(bot_clf, feats, bot_cols)
    p_top = _predict_with_cols(top_clf, feats, top_cols)
    T = len(feats)
    p_bottom = p_bot[:, 1] if (p_bot is not None and p_bot.ndim == 2 and p_bot.shape[1] >= 2) else np.zeros(T)
    p_topp = p_top[:, 1] if (p_top is not None and p_top.ndim == 2 and p_top.shape[1] >= 2) else np.zeros(T)

    root = _ts_outdir(str(_coerce_opt(outdir, str(Path(__file__).resolve().parents[2] / "plot" / "xgb_eval"))), prefix="globalcycle")
    _plot_prob_series(idx, {"p_bottom": p_bottom, "p_top": p_topp}, thresholds=None, title=f"GlobalCycle Probabilities {pair} {timeframe}", out_path=os.path.join(root, "globalcycle_probs.png"))
    ev = {
        "bottom": (p_bottom >= 0.6).astype(int),
        "top": (p_topp >= 0.6).astype(int),
    }
    o = feats["open"].astype(float).to_numpy() if "open" in feats.columns else raw["open"].astype(float).to_numpy()
    hi = feats["high"].astype(float).to_numpy() if "high" in feats.columns else raw["high"].astype(float).to_numpy()
    lo = feats["low"].astype(float).to_numpy() if "low" in feats.columns else raw["low"].astype(float).to_numpy()
    _plot_price_with_events(idx, close, ev, title=f"Price with GlobalCycle signals {pair} {timeframe}", out_path=os.path.join(root, "globalcycle_events.png"), o=o, h=hi, l=lo, use_candles=bool(_coerce_opt(candles, False)))
    _plot_feature_importance(bot_clf, bot_cols or list(feats.columns), out_path=os.path.join(root, "fi_global_bottom.png"), title="Feature importance - Global Bottom")
    _plot_feature_importance(top_clf, top_cols or list(feats.columns), out_path=os.path.join(root, "fi_global_top.png"), title="Feature importance - Global Top")
    try:
        _save_feature_importance_text(bot_clf, bot_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_global_bottom.txt"), top_k=500)
        _save_feature_importance_text(top_clf, top_cols or list(feats.columns), out_txt_path=os.path.join(root, "fi_global_top.txt"), top_k=500)
    except Exception:
        pass
    typer.echo(json.dumps({"outdir": root, "samples": int(T)}, indent=2))



from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
import inspect
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

from .features import make_features
from .train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df


@dataclass
class TimesFMForecastParams:
    """Configuration for running TimesFM 2.5 market forecasts."""

    userdir: str
    pair: str
    timeframe: str = "1h"
    timerange: Optional[str] = None
    prefer_exchange: Optional[str] = None

    feature_mode: str = "full"
    basic_lookback: int = 64
    extra_timeframes: Optional[Sequence[str]] = None
    target_columns: Optional[Sequence[str]] = None

    context_length: int = 1024
    horizon: int = 64
    stride: int = 1
    max_windows: int = 256

    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    quantile_levels: Optional[Sequence[float]] = None

    compile_flags: Optional[Dict[str, Any]] = None

    outdir: Optional[str] = None
    save_csv: bool = True
    save_json: bool = True
    make_plots: bool = False
    plot_windows: int = 3
    autodownload: bool = True
    calibrator_path: Optional[str] = None


@dataclass
class ResidualCalibratorParams(TimesFMForecastParams):
    train_ratio: float = 0.7
    alpha: float = 1.0
    model_out_path: str = "models/timesfm_calibrator.json"
    calibrator_feature_columns: Optional[Sequence[str]] = None
    save_val_csv: bool = True


class TimesFMNotAvailableError(RuntimeError):
    pass


def _ensure_timesfm_import():
    try:
        import timesfm  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep not always installed
        raise TimesFMNotAvailableError(
            "timesfm>=2.5.0 is required. Install with `pip install timesfm==2.5.0` in your environment."
        ) from exc
    return timesfm


def _build_windows(
    series: np.ndarray,
    context: int,
    horizon: int,
    stride: int,
    max_windows: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    n = int(series.shape[0])
    context = int(max(1, context))
    horizon = int(max(1, horizon))
    stride = int(max(1, stride))
    if n <= context + horizon:
        return [], [], []

    start = context
    end = n - horizon
    idxs = list(range(start, end, stride))
    if max_windows > 0 and len(idxs) > max_windows:
        idxs = idxs[-max_windows:]

    contexts: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    anchors: List[int] = []
    for i in idxs:
        ctx = series[i - context : i]
        if ctx.shape[0] < context:
            continue
        fut = series[i : i + horizon]
        contexts.append(ctx.astype(np.float32, copy=False))
        targets.append(fut.astype(np.float32, copy=False))
        anchors.append(i)
    return contexts, targets, anchors


def _quantile_labels(levels: Optional[Sequence[float]], count: int) -> List[str]:
    if count <= 0:
        return []
    if levels:
        lv = list(levels)
        if len(lv) == count - 1:
            labels = ["mean"]
            labels.extend(f"p{int(round(q * 100)):02d}" for q in lv)
            return labels
        if len(lv) == count:
            return [f"p{int(round(q * 100)):02d}" for q in lv]
    # Fallback: assume first slot is mean, rest unnamed
    labels = ["mean"]
    labels.extend(f"q{i+1}" for i in range(max(0, count - 1)))
    return labels[:count]


def _filter_forecast_kwargs(config_cls: Any, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Drop kwargs not accepted by ForecastConfig for backward compatibility."""
    try:
        params = set(inspect.signature(config_cls).parameters.keys())
    except (TypeError, ValueError):
        # Fallback: assume everything allowed when signature unavailable
        return dict(kwargs), []
    filtered = {}
    ignored: List[str] = []
    for key, value in kwargs.items():
        if key in params:
            filtered[key] = value
        else:
            ignored.append(key)
    return filtered, ignored


def _augment_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2D for augmentation.")
    if X.size == 0:
        return np.zeros((0, X.shape[1] + 1), dtype=X.dtype)
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, ones])


def _compute_adjustments(
    weights: np.ndarray,
    feature_matrix: np.ndarray,
    anchors: Sequence[int],
) -> np.ndarray:
    if weights.ndim != 2:
        raise ValueError("Weights must be a 2D matrix (features+1, horizon).")
    num_samples = len(anchors)
    horizon = weights.shape[1]
    adjustments = np.zeros((num_samples, horizon), dtype=float)
    W = weights.astype(float, copy=False)
    beta = W[:-1, :]
    b = W[-1, :]
    for i, anchor in enumerate(anchors):
        feat_idx = int(anchor) - 1
        if 0 <= feat_idx < feature_matrix.shape[0]:
            vec = feature_matrix[feat_idx].astype(float, copy=False)
            adjustments[i, :] = vec @ beta + b
    return adjustments


def _plot_forecast_windows(
    target: str,
    series: np.ndarray,
    time_index: pd.Index,
    anchors: Sequence[int],
    context: int,
    horizon: int,
    point_arr: np.ndarray,
    target_arr: np.ndarray,
    q_arr: Optional[np.ndarray],
    outdir: str,
    max_windows: int,
) -> List[str]:
    if not anchors:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(outdir, exist_ok=True)
    limit = max(1, int(max_windows))
    sel = list(range(max(0, len(anchors) - limit), len(anchors)))
    saved: List[str] = []
    for idx in sel:
        anchor = int(anchors[idx])
        ctx_start = max(0, anchor - context)
        ctx_end = anchor
        fut_end = min(len(series), anchor + horizon)
        ctx_times = time_index[ctx_start:ctx_end]
        fut_times = time_index[anchor:fut_end]
        ctx_vals = series[ctx_start:ctx_end]
        fut_actual = target_arr[idx][: len(fut_times)]
        fut_pred = point_arr[idx][: len(fut_times)]

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(ctx_times) > 0:
            ax.plot(ctx_times, ctx_vals, color="#7f7f7f", linewidth=1.0, linestyle="--", label="Context")
        ax.plot(fut_times, fut_actual, color="#1f77b4", linewidth=1.2, label="Actual")
        ax.plot(fut_times, fut_pred, color="#d62728", linewidth=1.2, label="Prediction")

        if q_arr is not None and q_arr.shape[-1] >= 3:
            lower = q_arr[idx][: len(fut_times), 1]
            upper = q_arr[idx][: len(fut_times), -1]
            ax.fill_between(fut_times, lower, upper, color="#ff9896", alpha=0.3, label="Quantile band")

        ax.set_title(f"TimesFM forecast for {target} (anchor={anchor})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fname = f"timesfm_{target}_win{idx}_anchor{anchor}.png"
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)
    return saved


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    err = pred - target
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(np.maximum(mse, 0.0)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(target), 1e-8)
    mape = float(np.mean(np.abs(err) / denom))
    bias = float(np.mean(err))
    per_h_rmse = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
    per_h_mae = np.mean(np.abs(err), axis=0).tolist()
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
        "rmse_per_horizon": per_h_rmse,
        "mae_per_horizon": per_h_mae,
    }


def _fit_linear_residual(X_aug: np.ndarray, residuals: np.ndarray, alpha: float) -> np.ndarray:
    X_aug = np.asarray(X_aug, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if X_aug.shape[0] == 0:
        raise ValueError("No samples available to fit residual calibrator.")
    XtX = X_aug.T @ X_aug
    reg = np.eye(X_aug.shape[1], dtype=float) * float(alpha)
    # Do not overly regularize intercept (last row)
    reg[-1, -1] = min(float(alpha), 1e-6)
    try:
        weights = np.linalg.solve(XtX + reg, X_aug.T @ residuals)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(X_aug, residuals, rcond=None)[0]
    return weights


def _prepare_features(params: TimesFMForecastParams) -> Tuple[pd.DataFrame, List[str]]:
    path = _ensure_dataset(params)
    if not path:
        raise FileNotFoundError(
            f"No dataset found for {params.pair} {params.timeframe} within {params.userdir}/data"
        )
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, params.timerange)
    feats = make_features(
        raw,
        mode=params.feature_mode,
        basic_lookback=params.basic_lookback,
        extra_timeframes=list(params.extra_timeframes) if params.extra_timeframes else None,
    )
    target_cols = list(params.target_columns) if params.target_columns else [c for c in ("close", "logret") if c in feats.columns][:1]
    if not target_cols:
        raise ValueError("Requested target columns not found and defaults unavailable.")
    for col in target_cols:
        if col not in feats.columns:
            raise ValueError(f"Target column '{col}' not present in feature dataframe.")
    return feats, target_cols


def _ensure_dataset(params: TimesFMForecastParams) -> Optional[str]:
    path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
    if path and os.path.exists(path):
        return path
    if not params.autodownload:
        return path

    exchange = params.prefer_exchange or "bybit"
    timerange = params.timerange or "20190101-"
    userdir = params.userdir
    pair = params.pair
    timeframe = params.timeframe

    os.makedirs(userdir, exist_ok=True)
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
            print(f"[TimesFM] Downloading dataset via Freqtrade: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as exc:
            last_err = exc
        found = _find_data_file(userdir, pv, timeframe, prefer_exchange=exchange)
        if found and os.path.exists(found):
            return found

    if last_err is not None:
        print(f"[TimesFM] Dataset download attempts failed for variants {variants}: {last_err}")
    return _find_data_file(userdir, pair, timeframe, prefer_exchange=exchange)


def run_timesfm_forecast(params: TimesFMForecastParams) -> Dict[str, Any]:
    timesfm = _ensure_timesfm_import()

    model = timesfm.TimesFM_2p5_200M_torch()
    model.load_checkpoint()

    compile_kwargs: Dict[str, Any] = {
        "max_context": int(max(params.context_length, 16)),
        "max_horizon": int(max(params.horizon, 1)),
        "normalize_inputs": bool(params.normalize_inputs),
        "use_continuous_quantile_head": bool(params.use_continuous_quantile_head),
        "force_flip_invariance": bool(params.force_flip_invariance),
        "infer_is_positive": bool(params.infer_is_positive),
        "fix_quantile_crossing": bool(params.fix_quantile_crossing),
    }
    if params.quantile_levels is not None:
        compile_kwargs["quantile_levels"] = list(params.quantile_levels)
    if params.compile_flags:
        for k, v in params.compile_flags.items():
            if v is not None:
                compile_kwargs[k] = v
    filtered_kwargs, ignored_keys = _filter_forecast_kwargs(timesfm.ForecastConfig, compile_kwargs)
    model.compile(timesfm.ForecastConfig(**filtered_kwargs))

    feats, target_cols = _prepare_features(params)
    time_index = feats.index
    calibration_bundle: Optional[Dict[str, Any]] = None
    calibration_feature_matrix: Optional[np.ndarray] = None
    if params.calibrator_path:
        try:
            calibration_bundle = load_residual_calibrator(params.calibrator_path)
            feature_cols = calibration_bundle.get("feature_columns", [])
            if feature_cols:
                missing = [c for c in feature_cols if c not in feats.columns]
                if missing:
                    print(f"[TimesFM] Calibrator feature columns missing: {missing}")
                    calibration_bundle = None
                else:
                    calibration_feature_matrix = feats[feature_cols].astype(float).to_numpy(copy=False)
        except Exception as exc:
            print(f"[TimesFM] Failed to load calibrator '{params.calibrator_path}': {exc}")
            calibration_bundle = None

    reports: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    plot_paths: Dict[str, List[str]] = {}
    for col in target_cols:
        series = feats[col].astype(float).to_numpy(copy=False)
        contexts, targets, anchors = _build_windows(
            series,
            context=params.context_length,
            horizon=params.horizon,
            stride=params.stride,
            max_windows=params.max_windows,
        )
        if not contexts:
            reports[col] = {
                "num_windows": 0,
                "metrics": {},
            }
            continue
        point_forecast, quantile_forecast = model.forecast(
            horizon=params.horizon,
            inputs=contexts,
        )
        point_arr = np.asarray(point_forecast, dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.float32)
        base_metrics = _compute_metrics(point_arr, target_arr)
        reports[col] = {
            "num_windows": int(point_arr.shape[0]),
            "metrics": base_metrics,
        }
        if calibration_bundle is not None and calibration_feature_matrix is not None:
            target_info = calibration_bundle.get("targets", {}).get(col)
            if target_info:
                weights_arr = np.asarray(target_info.get("weights", []), dtype=float)
                if weights_arr.size:
                    adjustments = _compute_adjustments(weights_arr, calibration_feature_matrix, anchors)
                    calibrated_point = point_arr + adjustments
                    cal_metrics = _compute_metrics(calibrated_point, target_arr)
                    reports[col]["calibrated_metrics"] = cal_metrics
                    point_arr = calibrated_point
                else:
                    reports[col]["calibration_warning"] = "Calibrator missing weights"
        q_labels: List[str] = []
        q_arr: Optional[np.ndarray] = None
        if quantile_forecast is not None:
            q_arr = np.asarray(quantile_forecast, dtype=np.float32)
            q_labels = _quantile_labels(params.quantile_levels, q_arr.shape[-1])
            reports[col]["quantile_labels"] = q_labels
        reports[col]["anchors"] = anchors

        for row_idx, anchor in enumerate(anchors):
            horizon_times = time_index[anchor : anchor + params.horizon]
            for h_step in range(params.horizon):
                ts = horizon_times[h_step] if h_step < len(horizon_times) else horizon_times[-1]
                rec = {
                    "target": col,
                    "window_index": row_idx,
                    "anchor_index": int(anchor),
                    "context_start_index": int(anchor - params.context_length),
                    "timestamp": str(ts),
                    "horizon_step": h_step + 1,
                    "actual": float(target_arr[row_idx, h_step]),
                    "prediction": float(point_arr[row_idx, h_step]),
                }
                if q_arr is not None:
                    for q_idx in range(q_arr.shape[-1]):
                        label = q_labels[q_idx] if q_idx < len(q_labels) else f"q{q_idx}"
                        rec[f"quantile_{label}"] = float(q_arr[row_idx, h_step, q_idx])
                records.append(rec)

        if params.make_plots and params.outdir:
            plot_files = _plot_forecast_windows(
                target=col,
                series=series,
                time_index=time_index,
                anchors=anchors,
                context=params.context_length,
                horizon=params.horizon,
                point_arr=point_arr,
                target_arr=target_arr,
                q_arr=q_arr,
                outdir=params.outdir,
                max_windows=params.plot_windows,
            )
            if plot_files:
                plot_paths[col] = plot_files
                reports[col]["plot_paths"] = plot_files

    outdir = params.outdir
    preds_path = None
    meta_path = None
    ignored_map = {k: compile_kwargs[k] for k in ignored_keys} if ignored_keys else {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if params.save_csv and records:
            preds_path = os.path.join(outdir, "timesfm_predictions.csv")
            pd.DataFrame.from_records(records).to_csv(preds_path, index=False)
        if params.save_json:
            meta_path = os.path.join(outdir, "timesfm_summary.json")
            payload = {"params": asdict(params), "reports": reports}
            if ignored_map:
                payload["ignored_compile_kwargs"] = ignored_map
            with open(meta_path, "w") as f:
                json.dump(payload, f, indent=2)

    return {
        "params": asdict(params),
        "reports": reports,
        "num_records": len(records),
        "predictions_path": preds_path,
        "summary_path": meta_path,
        "plot_paths": plot_paths,
        "ignored_compile_kwargs": ignored_map,
    }


def train_residual_calibrator(params: ResidualCalibratorParams) -> Dict[str, Any]:
    timesfm = _ensure_timesfm_import()

    model = timesfm.TimesFM_2p5_200M_torch()
    model.load_checkpoint()

    compile_kwargs: Dict[str, Any] = {
        "max_context": int(max(params.context_length, 16)),
        "max_horizon": int(max(params.horizon, 1)),
        "normalize_inputs": bool(params.normalize_inputs),
        "use_continuous_quantile_head": bool(params.use_continuous_quantile_head),
        "force_flip_invariance": bool(params.force_flip_invariance),
        "infer_is_positive": bool(params.infer_is_positive),
        "fix_quantile_crossing": bool(params.fix_quantile_crossing),
    }
    if params.quantile_levels is not None:
        compile_kwargs["quantile_levels"] = list(params.quantile_levels)
    if params.compile_flags:
        for k, v in params.compile_flags.items():
            if v is not None:
                compile_kwargs[k] = v
    filtered_kwargs, ignored_keys = _filter_forecast_kwargs(timesfm.ForecastConfig, compile_kwargs)
    model.compile(timesfm.ForecastConfig(**filtered_kwargs))

    feats, target_cols = _prepare_features(params)
    feature_cols = list(params.calibrator_feature_columns) if params.calibrator_feature_columns else list(feats.columns)
    feature_matrix = feats[feature_cols].astype(float).to_numpy(copy=False)
    time_index = feats.index

    total_summary: Dict[str, Any] = {}
    ignored_map = {k: compile_kwargs[k] for k in ignored_keys} if ignored_keys else {}

    model_dir = os.path.dirname(params.model_out_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if params.outdir:
        os.makedirs(params.outdir, exist_ok=True)

    for col in target_cols:
        series = feats[col].astype(float).to_numpy(copy=False)
        contexts, targets, anchors = _build_windows(
            series,
            context=params.context_length,
            horizon=params.horizon,
            stride=params.stride,
            max_windows=params.max_windows,
        )
        if not contexts:
            total_summary[col] = {"warning": "Not enough windows to train calibrator."}
            continue

        point_forecast, _ = model.forecast(horizon=params.horizon, inputs=contexts)
        point_arr = np.asarray(point_forecast, dtype=float)
        target_arr = np.asarray(targets, dtype=float)

        anchor_feats: List[np.ndarray] = []
        sel_point: List[np.ndarray] = []
        sel_target: List[np.ndarray] = []
        sel_anchors: List[int] = []
        for idx, anchor in enumerate(anchors):
            feat_idx = int(anchor) - 1
            if feat_idx < 0 or feat_idx >= feature_matrix.shape[0]:
                continue
            anchor_feats.append(feature_matrix[feat_idx])
            sel_point.append(point_arr[idx])
            sel_target.append(target_arr[idx])
            sel_anchors.append(int(anchor))

        if not anchor_feats:
            total_summary[col] = {"warning": "No valid anchor features available."}
            continue

        X = np.vstack(anchor_feats)
        base_preds = np.vstack(sel_point)
        target_sel = np.vstack(sel_target)
        n = X.shape[0]
        if n < 2:
            total_summary[col] = {"warning": "Need at least two samples to split train/val."}
            continue

        train_size = int(n * float(params.train_ratio))
        train_size = max(1, min(n - 1, train_size))

        X_train = X[:train_size]
        X_val = X[train_size:]
        base_train = base_preds[:train_size]
        base_val = base_preds[train_size:]
        target_train = target_sel[:train_size]
        target_val = target_sel[train_size:]

        residual_train = target_train - base_train
        X_train_aug = _augment_features(X_train)
        weights = _fit_linear_residual(X_train_aug, residual_train, float(params.alpha))

        train_adjust = X_train_aug @ weights
        train_calibrated = base_train + train_adjust
        train_metrics = _compute_metrics(train_calibrated, target_train)

        val_adjust = _augment_features(X_val) @ weights if len(X_val) else np.zeros_like(base_val)
        calibrated_val = base_val + val_adjust
        base_val_metrics = _compute_metrics(base_val, target_val)
        cal_val_metrics = _compute_metrics(calibrated_val, target_val)

        target_summary: Dict[str, Any] = {
            "weights": weights.tolist(),
            "train_samples": int(train_size),
            "val_samples": int(len(X_val)),
            "train_metrics": train_metrics,
            "base_val_metrics": base_val_metrics,
            "calibrated_val_metrics": cal_val_metrics,
        }

        if params.outdir and params.save_val_csv and len(X_val):
            rows: List[Dict[str, Any]] = []
            for i in range(len(X_val)):
                anchor_idx = sel_anchors[train_size + i]
                ts = time_index[min(anchor_idx, len(time_index) - 1)]
                for h_step in range(params.horizon):
                    rows.append({
                        "timestamp": str(ts),
                        "target": col,
                        "horizon_step": h_step + 1,
                        "actual": float(target_val[i, h_step]),
                        "base_prediction": float(base_val[i, h_step]),
                        "calibrated_prediction": float(calibrated_val[i, h_step]),
                        "adjustment": float(val_adjust[i, h_step]),
                    })
            if rows:
                val_path = os.path.join(params.outdir, f"calibrator_val_{col}.csv")
                pd.DataFrame(rows).to_csv(val_path, index=False)
                target_summary["val_csv_path"] = val_path

        total_summary[col] = target_summary

    bundle = {
        "version": 1,
        "feature_columns": feature_cols,
        "alpha": float(params.alpha),
        "train_ratio": float(params.train_ratio),
        "horizon": int(params.horizon),
        "context_length": int(params.context_length),
        "targets": total_summary,
    }
    if params.quantile_levels is not None:
        bundle["quantile_levels"] = list(params.quantile_levels)
    if ignored_map:
        bundle["ignored_compile_kwargs"] = ignored_map

    with open(params.model_out_path, "w") as f:
        json.dump(bundle, f, indent=2)

    return {
        "calibrator_path": params.model_out_path,
        "targets": total_summary,
        "feature_columns": feature_cols,
        "ignored_compile_kwargs": ignored_map,
    }


def load_residual_calibrator(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if "feature_columns" not in data or "targets" not in data:
        raise ValueError("Invalid residual calibrator bundle.")
    return data

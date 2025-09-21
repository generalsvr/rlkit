from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
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


def _prepare_features(params: TimesFMForecastParams) -> Tuple[pd.DataFrame, List[str]]:
    path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
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
    model.compile(timesfm.ForecastConfig(**compile_kwargs))

    feats, target_cols = _prepare_features(params)
    time_index = feats.index

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
        metrics = _compute_metrics(point_arr, target_arr)
        reports[col] = {
            "num_windows": int(point_arr.shape[0]),
            "metrics": metrics,
        }
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
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if params.save_csv and records:
            preds_path = os.path.join(outdir, "timesfm_predictions.csv")
            pd.DataFrame.from_records(records).to_csv(preds_path, index=False)
        if params.save_json:
            meta_path = os.path.join(outdir, "timesfm_summary.json")
            with open(meta_path, "w") as f:
                json.dump({"params": asdict(params), "reports": reports}, f, indent=2)

    return {
        "params": asdict(params),
        "reports": reports,
        "num_records": len(records),
        "predictions_path": preds_path,
        "summary_path": meta_path,
        "plot_paths": plot_paths,
    }

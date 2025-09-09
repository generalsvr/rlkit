from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from .train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df
from .features import make_features

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


def _compute_atr_norm(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1 / max(1, period), adjust=False).mean().to_numpy()
    return atr / (close + 1e-12)


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_norm: np.ndarray,
    entries: np.ndarray,
    direction: int,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute triple-barrier outcomes per entry.

    Returns (labels, horizon) where labels are 1 if PT first, 0 otherwise. Horizon is bars until decision.
    direction: +1 for long, -1 for short
    entries: binary vector length T with 1 when entry occurs
    """
    T = len(close)
    labels = np.full(T, -1, dtype=int)
    horizon = np.full(T, 0, dtype=int)
    entry_idx = np.flatnonzero(entries == 1)
    if entry_idx.size == 0:
        return labels, horizon
    for idx in entry_idx:
        if idx >= T - 2:
            continue
        c0 = float(close[idx])
        # Use ATR at entry
        atr0 = float(atr_norm[idx])
        if not np.isfinite(atr0) or atr0 <= 0.0:
            atr0 = float(np.nanmedian(atr_norm[max(0, idx-200):idx+1]) or 0.005)
        # Barriers
        if direction > 0:
            pt_level = c0 * (1.0 + pt_mult * atr0)
            sl_level = c0 * (1.0 - sl_mult * atr0)
        else:
            pt_level = c0 * (1.0 - pt_mult * atr0)
            sl_level = c0 * (1.0 + sl_mult * atr0)
        decided = False
        max_h = int(max(1, max_holding))
        for h in range(1, min(max_h + 1, T - idx)):
            hi = float(high[idx + h])
            lo = float(low[idx + h])
            if direction > 0:
                hit_pt = hi >= pt_level
                hit_sl = lo <= sl_level
            else:
                hit_pt = lo <= pt_level
                hit_sl = hi >= sl_level
            if hit_pt or hit_sl:
                labels[idx] = 1 if hit_pt and not hit_sl else 0
                horizon[idx] = h
                decided = True
                break
        if not decided:
            # Time barrier -> label 0
            labels[idx] = 0
            horizon[idx] = max_h
    return labels, horizon


def triple_barrier_multi_outcomes(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_norm: np.ndarray,
    entries: np.ndarray,
    direction: int,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triple-barrier with 3 outcomes per entry: 0=TP, 1=SL, 2=TIME.

    Returns (labels, horizon). labels is -1 where not an entry.
    """
    T = len(close)
    labels = np.full(T, -1, dtype=int)
    horizon = np.full(T, 0, dtype=int)
    idxs = np.flatnonzero(entries == 1)
    if idxs.size == 0:
        return labels, horizon
    for idx in idxs:
        if idx >= T - 2:
            continue
        c0 = float(close[idx])
        atr0 = float(atr_norm[idx])
        if not np.isfinite(atr0) or atr0 <= 0.0:
            atr0 = float(np.nanmedian(atr_norm[max(0, idx-200):idx+1]) or 0.005)
        if direction > 0:
            pt_level = c0 * (1.0 + pt_mult * atr0)
            sl_level = c0 * (1.0 - sl_mult * atr0)
        else:
            pt_level = c0 * (1.0 - pt_mult * atr0)
            sl_level = c0 * (1.0 + sl_mult * atr0)
        decided = False
        max_h = int(max(1, max_holding))
        for h in range(1, min(max_h + 1, T - idx)):
            hi = float(high[idx + h])
            lo = float(low[idx + h])
            if direction > 0:
                hit_pt = hi >= pt_level
                hit_sl = lo <= sl_level
            else:
                hit_pt = lo <= pt_level
                hit_sl = hi >= sl_level
            if hit_pt or hit_sl:
                labels[idx] = 0 if hit_pt and not hit_sl else 1  # 0=TP,1=SL
                horizon[idx] = h
                decided = True
                break
        if not decided:
            labels[idx] = 2  # TIME
            horizon[idx] = max_h
    return labels, horizon


def build_meta_outcome_dataset_from_signals(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    primary_scores: Dict[str, np.ndarray],
    score_thresholds: Dict[str, float],
    direction_map: Dict[str, int],
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    extra_feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build meta outcome dataset: X rows at candidate entries; y in {0=TP,1=SL,2=TIME}."""
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    atrn = _compute_atr_norm(high, low, close, period=14)

    rows: List[List[float]] = []
    labels: List[int] = []
    idx_list: List[int] = []

    base_cols: List[str] = []
    default_cols = [
        "atr","ret_std_14","adx_14","volatility_regime","risk_gate",
        "ema_fast_ratio","ema_slow_ratio","sma_ratio","macd_line","macd_signal",
    ]
    if extra_feature_cols:
        default_cols = list(dict.fromkeys(default_cols + list(extra_feature_cols)))
    for c in default_cols:
        if c in feature_df.columns and c not in base_cols:
            base_cols.append(c)

    for name, scores in primary_scores.items():
        thr = float(score_thresholds.get(name, 0.6))
        direction = int(direction_map.get(name, 0))
        if direction == 0:
            continue
        candidates = (scores >= thr).astype(int)
        y_out, _h = triple_barrier_multi_outcomes(close, high, low, atrn, entries=candidates, direction=direction, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
        idxs = np.flatnonzero(candidates == 1)
        for i in idxs:
            if y_out[i] < 0:
                continue
            row: List[float] = []
            # primary scores
            for n2, sc in primary_scores.items():
                v = float(sc[i]) if (isinstance(sc, np.ndarray) and i < sc.shape[0]) else 0.0
                if not np.isfinite(v):
                    v = 0.0
                row.append(v)
            # base context
            for c in base_cols:
                v = float(feature_df.iloc[i][c]) if c in feature_df.columns else 0.0
                if not np.isfinite(v):
                    v = 0.0
                row.append(v)
            # direction flag
            row.append(float(direction))
            rows.append(row)
            labels.append(int(y_out[i]))
            idx_list.append(i)

    if not rows:
        return pd.DataFrame(), np.array([], dtype=int)
    feat_cols = list(primary_scores.keys()) + base_cols + ["dir"]
    X = pd.DataFrame(rows, columns=feat_cols, index=pd.Index(idx_list))
    y = np.asarray(labels, dtype=int)
    return X, y


class MetaOutcomeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] | None = None, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        if hidden is None:
            hidden = [128, 64]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_meta_outcome_mlp(
    X: pd.DataFrame,
    y: np.ndarray,
    out_json_path: str,
    hidden: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    weight_decay: float = 0.0,
    device: str = "auto",
) -> Dict[str, Any]:
    if not _TORCH_OK:
        raise ImportError("PyTorch is required for meta outcome MLP.")
    import json as _json
    from sklearn.metrics import accuracy_score

    cols = list(X.columns)
    Xn = np.asarray(X.values, dtype=np.float32)
    yn = np.asarray(y, dtype=np.int64)
    n = Xn.shape[0]
    cut = int(max(50, min(n - 20, int(n * 0.8))))
    X_tr = Xn[:cut]; y_tr = yn[:cut]
    X_va = Xn[cut:]; y_va = yn[cut:]

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_ld = DataLoader(tr_ds, batch_size=int(max(1, batch_size)), shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False)

    dev = torch.device("cuda" if (device in ("auto","cuda") and torch.cuda.is_available()) else "cpu")
    model = MetaOutcomeMLP(input_dim=Xn.shape[1], hidden=hidden or [128, 64], dropout=0.1, num_classes=3).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss()

    best_acc = float('-inf')
    best_state: Dict[str, Any] | None = None
    for ep in range(int(max(1, epochs))):
        model.train()
        for xb, yb in tr_ld:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
        # Eval
        model.eval()
        preds: List[int] = []
        with torch.no_grad():
            for xb, _yb in va_ld:
                xb = xb.to(dev)
                logits = model(xb)
                ph = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(ph)
        try:
            acc = float(accuracy_score(y_va, np.asarray(preds, dtype=int)))
        except Exception:
            acc = float('nan')
        if np.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_state = {"model": model.state_dict()}

    if best_state is None:
        best_state = {"model": model.state_dict()}
    weights_path = os.path.splitext(out_json_path)[0] + ".pt"
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    torch.save(best_state, weights_path)
    manifest = {
        "type": "MetaOutcomeMLP",
        "weights_path": weights_path,
        "feature_columns": cols,
        "input_dim": int(Xn.shape[1]),
        "hidden": list(hidden or [128, 64]),
        "dropout": 0.1,
        "val_acc": float(best_acc),
    }
    with open(out_json_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    return {"manifest": out_json_path, "weights": weights_path, "val_acc": float(best_acc)}


def load_meta_outcome_from_json(json_path: str) -> Tuple[MetaOutcomeMLP, List[str]]:
    import json as _json
    with open(json_path, "r") as f:
        man = _json.load(f)
    if man.get("type") != "MetaOutcomeMLP":
        raise ValueError("Manifest is not a MetaOutcomeMLP")
    input_dim = int(man["input_dim"])
    hidden = list(man.get("hidden", [128, 64]))
    dropout = float(man.get("dropout", 0.1))
    model = MetaOutcomeMLP(input_dim=input_dim, hidden=hidden, dropout=dropout, num_classes=3)
    state = torch.load(man["weights_path"], map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, list(man.get("feature_columns", []))

def build_meta_dataset_from_signals(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    primary_scores: Dict[str, np.ndarray],
    score_thresholds: Dict[str, float],
    direction_map: Dict[str, int],
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    extra_feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build meta-label dataset.

    primary_scores: mapping name->score array (e.g., {'p_bottom': arr, 'p_top': arr})
    score_thresholds: mapping name->threshold to define a candidate trade
    direction_map: mapping name->+1/-1 to indicate trade direction for that score
    Returns (X_meta, y_meta)
    """
    # Required market arrays
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    atrn = _compute_atr_norm(high, low, close, period=14)

    # Build candidate entries per score
    all_rows: List[int] = []
    data_rows: List[List[float]] = []
    labels: List[int] = []

    # Meta features base: include the primary scores and selected market state features
    base_cols: List[str] = []
    # Include available columns from feature_df
    default_cols = [
        "atr","ret_std_14","adx_14","volatility_regime","risk_gate",
        "ema_fast_ratio","ema_slow_ratio","sma_ratio","macd_line","macd_signal",
    ]
    if extra_feature_cols:
        default_cols = list(dict.fromkeys(default_cols + list(extra_feature_cols)))
    for c in default_cols:
        if c in feature_df.columns and c not in base_cols:
            base_cols.append(c)

    # Iterate each score source
    for name, scores in primary_scores.items():
        thr = float(score_thresholds.get(name, 0.5))
        direction = int(direction_map.get(name, 0))
        if direction == 0:
            continue
        candidates = (scores >= thr).astype(int)
        y_lbl, _h = triple_barrier_labels(close, high, low, atrn, entries=candidates, direction=direction, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
        idxs = np.flatnonzero(candidates == 1)
        for i in idxs:
            if y_lbl[i] < 0:
                continue
            row: List[float] = []
            # Primary scores as features
            for n2, sc in primary_scores.items():
                v = float(sc[i]) if (isinstance(sc, np.ndarray) and i < sc.shape[0]) else float("nan")
                if not np.isfinite(v):
                    v = 0.0
                row.append(v)
            # Add market state features
            for c in base_cols:
                v = float(feature_df.iloc[i][c]) if c in feature_df.columns else 0.0
                if not np.isfinite(v):
                    v = 0.0
                row.append(v)
            # Optional direction flag for clarity
            row.append(float(direction))
            data_rows.append(row)
            labels.append(int(y_lbl[i]))
            all_rows.append(i)

    if not data_rows:
        return pd.DataFrame(), np.array([], dtype=int)

    # Construct columns: primary score names + base_cols + direction
    feat_cols = list(primary_scores.keys()) + base_cols + ["dir"]
    X = pd.DataFrame(data_rows, columns=feat_cols, index=pd.Index(all_rows))
    y = np.asarray(labels, dtype=int)
    return X, y


@dataclass
class MetaTrainSpec:
    userdir: str
    pair: str = "BTC/USDT"
    timeframe: str = "1h"
    prefer_exchange: Optional[str] = None
    timerange: Optional[str] = None
    feature_mode: str = "full"
    basic_lookback: int = 64
    extra_timeframes: Optional[List[str]] = None
    # Primary model score names and thresholds
    score_names: List[str] = None  # e.g., ["p_bottom","p_top"]
    score_thresholds: Dict[str, float] = None
    direction_map: Dict[str, int] = None  # {"p_bottom": +1, "p_top": -1}
    pt_mult: float = 2.0
    sl_mult: float = 1.0
    max_holding: int = 24
    extra_feature_cols: Optional[List[str]] = None
    out_model_path: str = "models/xgb_meta.json"
    device: str = "auto"
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = 0


def train_meta_labeler(spec: MetaTrainSpec, primary_scores_df: pd.DataFrame) -> Dict[str, Any]:
    """Train XGBoost meta-labeler given primary model scores and feature context.

    primary_scores_df: DataFrame aligned to price index with columns matching spec.score_names
    Returns metadata with paths and metrics.
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    data_path = _find_data_file(spec.userdir, spec.pair, spec.timeframe, prefer_exchange=spec.prefer_exchange)
    if not data_path:
        raise FileNotFoundError("No dataset found for meta training.")
    raw = _load_ohlcv(data_path)
    raw = _slice_timerange_df(raw, spec.timerange)

    feats = make_features(
        raw,
        mode=spec.feature_mode,
        basic_lookback=spec.basic_lookback,
        extra_timeframes=spec.extra_timeframes,
    )
    # Align scores to feats index
    scores = primary_scores_df.reindex(feats.index).fillna(0.0)
    # Build dataset
    X_meta, y_meta = build_meta_dataset_from_signals(
        df=raw if all(col in raw.columns for col in ("open","high","low","close","volume")) else feats,
        feature_df=feats,
        primary_scores={k: scores[k].astype(float).to_numpy() for k in spec.score_names},
        score_thresholds=spec.score_thresholds,
        direction_map=spec.direction_map,
        pt_mult=spec.pt_mult,
        sl_mult=spec.sl_mult,
        max_holding=spec.max_holding,
        extra_feature_cols=spec.extra_feature_cols,
    )
    if X_meta.empty or y_meta.size == 0:
        raise ValueError("No meta training samples generated. Check thresholds and score columns.")

    # Train/validation split by time
    order = np.argsort(X_meta.index.values)
    X_meta = X_meta.iloc[order]
    y_meta = y_meta[order]
    cut = int(max(50, min(len(X_meta) - 20, int(len(X_meta) * 0.8))))
    X_tr = X_meta.iloc[:cut, :].values
    y_tr = y_meta[:cut]
    X_val = X_meta.iloc[cut:, :].values
    y_val = y_meta[cut:]

    dev = "cuda" if str(spec.device).lower() in ("auto","cuda") else "cpu"
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device=dev,
        random_state=42,
        n_jobs=(int(spec.n_jobs) if int(spec.n_jobs) > 0 else -1),
        learning_rate=float(spec.learning_rate),
        max_depth=int(spec.max_depth),
        subsample=float(spec.subsample),
        colsample_bytree=float(spec.colsample_bytree),
        reg_alpha=float(spec.reg_alpha),
        reg_lambda=float(spec.reg_lambda),
        n_estimators=int(spec.n_estimators),
        eval_metric="aucpr",
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p = clf.predict_proba(X_val)[:, 1]
    try:
        auprc = float(average_precision_score(y_val, p))
    except Exception:
        auprc = float("nan")

    # Persist
    os.makedirs(os.path.dirname(spec.out_model_path), exist_ok=True)
    clf.save_model(spec.out_model_path)
    # Save feature columns used
    try:
        import json as _json
        with open(str(os.path.splitext(spec.out_model_path)[0]) + "_feature_columns.json", "w") as f:
            _json.dump(list(X_meta.columns), f)
    except Exception:
        pass

    return {
        "path": spec.out_model_path,
        "val_samples": int(len(y_val)),
        "auprc": float(auprc),
        "n_samples": int(len(y_meta)),
        "feature_columns": list(X_meta.columns),
    }


def build_stacking_dataset(
    feats: pd.DataFrame,
    l0_outputs: pd.DataFrame,
    labels_from_triple_barrier: Tuple[np.ndarray, np.ndarray],
    extra_feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construct Level-1 manager dataset.

    l0_outputs should contain specialist outputs like p_bottom, p_top, p_up, p_dn, reg_direction, predicted_volatility.
    labels_from_triple_barrier: (y, horizon)
    """
    base_cols = list(l0_outputs.columns)
    extra_cols = []
    default_cols = ["ret_std_14","adx_14","volatility_regime","risk_gate"]
    if extra_feature_cols:
        default_cols = list(dict.fromkeys(default_cols + list(extra_feature_cols)))
    for c in default_cols:
        if c in feats.columns and c not in base_cols:
            extra_cols.append(c)
    X = l0_outputs.join(feats[extra_cols], how="left").fillna(0.0)
    y, _h = labels_from_triple_barrier
    # Align to same index positions where y defined (entries only)
    idx = [i for i, v in enumerate(y) if v >= 0]
    if not idx:
        return pd.DataFrame(), np.array([], dtype=int)
    X1 = X.iloc[idx].copy()
    y1 = y[idx]
    return X1, y1


# -----------------------------
# Meta MLP (Level-1 Manager)
# -----------------------------

class MetaMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_meta_mlp_manager(
    X: pd.DataFrame,
    y: np.ndarray,
    out_json_path: str,
    hidden: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    weight_decay: float = 0.0,
    device: str = "auto",
) -> Dict[str, Any]:
    """Train a simple MLP meta-model on (X,y) and save manifest JSON + weights (.pt).

    out_json_path: will save manifest JSON here and weights to same stem with .pt extension.
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch is required for MLP meta training. Ensure 'torch' is installed.")

    import json as _json
    from sklearn.metrics import average_precision_score

    cols = list(X.columns)
    Xn = np.asarray(X.values, dtype=np.float32)
    yn = np.asarray(y, dtype=np.float32)
    # Train/val split by order (time-based)
    n = Xn.shape[0]
    cut = int(max(50, min(n - 20, int(n * 0.8))))
    X_tr = Xn[:cut]
    y_tr = yn[:cut]
    X_val = Xn[cut:]
    y_val = yn[cut:]

    # Dataloaders
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_ld = DataLoader(tr_ds, batch_size=int(max(1, batch_size)), shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False)

    # Device
    dev = torch.device("cuda" if (device in ("auto","cuda") and torch.cuda.is_available()) else "cpu")
    model = MetaMLP(input_dim=Xn.shape[1], hidden=hidden or [128, 64], dropout=0.1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCEWithLogitsLoss()

    best_ap = float('-inf')
    best_state: Dict[str, Any] | None = None
    for ep in range(int(max(1, epochs))):
        model.train()
        for xb, yb in tr_ld:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
        # Eval
        model.eval()
        preds: List[float] = []
        with torch.no_grad():
            for xb, _yb in va_ld:
                xb = xb.to(dev)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds.extend(probs.detach().cpu().numpy().tolist())
        try:
            ap = float(average_precision_score(y_val, np.asarray(preds, dtype=float)))
        except Exception:
            ap = float('nan')
        if np.isfinite(ap) and ap > best_ap:
            best_ap = ap
            best_state = {"model": model.state_dict()}

    # Save
    if best_state is None:
        best_state = {"model": model.state_dict()}
    weights_path = os.path.splitext(out_json_path)[0] + ".pt"
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    torch.save(best_state, weights_path)
    manifest = {
        "type": "MetaMLP",
        "weights_path": weights_path,
        "feature_columns": cols,
        "input_dim": int(Xn.shape[1]),
        "hidden": list(hidden or [128, 64]),
        "dropout": 0.1,
        "val_auprc": float(best_ap),
    }
    with open(out_json_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    return {"manifest": out_json_path, "weights": weights_path, "val_auprc": float(best_ap)}


def load_meta_mlp_from_json(json_path: str) -> Tuple[MetaMLP, List[str]]:
    import json as _json
    with open(json_path, "r") as f:
        man = _json.load(f)
    if man.get("type") != "MetaMLP":
        raise ValueError("Manifest is not a MetaMLP")
    input_dim = int(man["input_dim"])
    hidden = list(man.get("hidden", [128, 64]))
    dropout = float(man.get("dropout", 0.1))
    model = MetaMLP(input_dim=input_dim, hidden=hidden, dropout=dropout)
    state = torch.load(man["weights_path"], map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, list(man.get("feature_columns", []))



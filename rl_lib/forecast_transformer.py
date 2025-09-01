from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .features import make_features
from .train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df
from .transformer_extractor import PositionalEncoding  # reuse same positional enc


class SequenceWindowDataset(Dataset):
    """
    Sliding-window dataset for (encoder context -> decoder horizon) forecasting.

    X: (T, F) features over past window
    Y: (H, D) target columns over future horizon
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        target_columns: List[str],
        window: int,
        horizon: int,
        feature_columns: Optional[List[str]] = None,
        input_scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target_scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        if feature_columns is None:
            feature_columns = list(features_df.columns)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.window = int(window)
        self.horizon = int(horizon)

        self.X = features_df[self.feature_columns].astype(float).to_numpy(copy=False)
        self.Y = features_df[self.target_columns].astype(float).to_numpy(copy=False)

        # Compute scalers if not provided (mean/std per column)
        if input_scaler is None:
            mu_x = self.X.mean(axis=0)
            std_x = self.X.std(axis=0)
            std_x = np.where(np.isfinite(std_x) & (std_x > 0.0), std_x, 1.0)
            self.input_mu = mu_x
            self.input_std = std_x
        else:
            self.input_mu, self.input_std = input_scaler

        if target_scaler is None:
            mu_y = self.Y.mean(axis=0)
            std_y = self.Y.std(axis=0)
            std_y = np.where(np.isfinite(std_y) & (std_y > 0.0), std_y, 1.0)
            self.target_mu = mu_y
            self.target_std = std_y
        else:
            self.target_mu, self.target_std = target_scaler

        self.num_samples = max(0, len(self.X) - self.window - self.horizon + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i0 = idx
        i1 = idx + self.window
        j0 = i1
        j1 = i1 + self.horizon

        x = self.X[i0:i1]
        y = self.Y[j0:j1]

        # Normalize
        x_n = (x - self.input_mu) / self.input_std
        y_n = (y - self.target_mu) / self.target_std

        return {
            "x": torch.as_tensor(x_n, dtype=torch.float32),  # (T, F)
            "y": torch.as_tensor(y_n, dtype=torch.float32),  # (H, D)
        }


class CandleDecoder(nn.Module):
    """
    Encoder-decoder Transformer for multistep forecasting.

    - Encoder ingests past feature sequence (T, F).
    - Decoder autoregressively predicts target sequence (H, D) with causal mask.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.d_model = int(d_model)

        self.enc_in = nn.Linear(self.input_dim, d_model)
        self.dec_in = nn.Linear(self.target_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.pos_dec = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.out = nn.Linear(d_model, self.target_dim)
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # True above diagonal = masked
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, x_ctx: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ctx: (B, T, F) normalized context features
            y_in:  (B, H, D) normalized decoder input (teacher forcing; shifted targets)

        Returns:
            y_pred: (B, H, D) normalized predictions
        """
        b, t, f = x_ctx.shape
        _, h, d = y_in.shape

        mem = self.enc_in(x_ctx)
        mem = self.pos_enc(mem)
        mem = self.encoder(mem, mask=None)
        mem = self.enc_norm(mem)

        tgt = self.dec_in(y_in)
        tgt = self.pos_dec(tgt)
        dec_mask = self._generate_square_subsequent_mask(h, device=tgt.device)
        dec_out = self.decoder(tgt=tgt, memory=mem, tgt_mask=dec_mask)
        dec_out = self.dec_norm(dec_out)
        y_pred = self.out(dec_out)
        return y_pred

    @torch.no_grad()
    def infer_autoregressive(
        self,
        x_ctx: torch.Tensor,
        start_token: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Greedy AR decoding on normalized space.

        Args:
            x_ctx: (B, T, F)
            start_token: (B, 1, D) initial token (e.g., last known target normalized)
            horizon: number of steps to forecast

        Returns:
            y_hat: (B, H, D) normalized predictions
        """
        device = next(self.parameters()).device
        x_ctx = x_ctx.to(device)
        mem = self.enc_in(x_ctx)
        mem = self.pos_enc(mem)
        mem = self.encoder(mem)
        mem = self.enc_norm(mem)

        tokens = [start_token.to(device)]
        for _ in range(horizon):
            tgt = torch.cat(tokens, dim=1)  # (B, L, D)
            h = tgt.size(1)
            tgt_proj = self.dec_in(tgt)
            tgt_proj = self.pos_dec(tgt_proj)
            mask = self._generate_square_subsequent_mask(h, device=device)
            out = self.decoder(tgt=tgt_proj, memory=mem, tgt_mask=mask)
            out = self.dec_norm(out)
            y_step = self.out(out[:, -1:, :])  # last step
            tokens.append(y_step)
        # drop the initial start token
        y_hat = torch.cat(tokens[1:], dim=1)
        return y_hat


@dataclass
class ForecastTrainParams:
    userdir: str
    pair: str
    timeframe: str = "1h"
    feature_mode: str = "full"  # full|basic|ohlcv (ohlcv = candles-only as inputs)
    basic_lookback: int = 64
    extra_timeframes: Optional[List[str]] = None
    timerange: Optional[str] = None
    prefer_exchange: Optional[str] = None

    window: int = 128
    horizon: int = 16
    target_columns: Optional[List[str]] = None  # default OHLCV if None

    # Model
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    seed: int = 42

    model_out_path: str = "models/forecaster.pt"


def _build_datasets(
    raw: pd.DataFrame,
    params: ForecastTrainParams,
) -> Tuple[SequenceWindowDataset, SequenceWindowDataset, List[str], List[str]]:
    mode_l = (params.feature_mode or "full").lower()
    feats = make_features(
        raw,
        feature_columns=None,
        mode=("full" if mode_l == "ohlcv" else mode_l),
        basic_lookback=params.basic_lookback,
        extra_timeframes=params.extra_timeframes,
    )

    # Default targets: OHLCV (if present)
    default_targets = [c for c in ["open", "high", "low", "close", "volume"] if c in feats.columns]
    target_cols = params.target_columns or default_targets
    if not target_cols:
        raise ValueError("No target columns available in features dataframe.")

    # Feature columns
    if mode_l == "ohlcv":
        cand_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in feats.columns]
        if not cand_cols:
            raise ValueError("OHLCV columns not found in features dataframe.")
        feature_cols = cand_cols
        feats = feats[feature_cols]
    else:
        feature_cols = list(feats.columns)

    n = len(feats)
    cut = int(n * 0.8)
    train_df = feats.iloc[:cut].copy()
    valid_df = feats.iloc[cut - max(params.window, 1):].copy()

    # Fit scalers on train
    x_mu = train_df[feature_cols].astype(float).to_numpy().mean(axis=0)
    x_std = train_df[feature_cols].astype(float).to_numpy().std(axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 0.0), x_std, 1.0)
    y_mu = train_df[target_cols].astype(float).to_numpy().mean(axis=0)
    y_std = train_df[target_cols].astype(float).to_numpy().std(axis=0)
    y_std = np.where(np.isfinite(y_std) & (y_std > 0.0), y_std, 1.0)

    train_ds = SequenceWindowDataset(
        train_df, target_columns=target_cols, window=params.window, horizon=params.horizon,
        feature_columns=feature_cols, input_scaler=(x_mu, x_std), target_scaler=(y_mu, y_std)
    )
    valid_ds = SequenceWindowDataset(
        valid_df, target_columns=target_cols, window=params.window, horizon=params.horizon,
        feature_columns=feature_cols, input_scaler=(x_mu, x_std), target_scaler=(y_mu, y_std)
    )
    return train_ds, valid_ds, feature_cols, target_cols


def _prepare_decoder_inputs(y_true: torch.Tensor) -> torch.Tensor:
    """
    Build teacher-forcing decoder inputs by shifting right and prepending zeros.
    y_true: (B, H, D) normalized targets
    Returns y_in: (B, H, D)
    """
    b, h, d = y_true.shape
    start = torch.zeros((b, 1, d), dtype=y_true.dtype, device=y_true.device)
    y_in = torch.cat([start, y_true[:, :-1, :]], dim=1)
    return y_in


def _metrics_denorm(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_true, y_pred: (N, H, D) denormalized
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    # MAPE only on positive targets to avoid division by zero
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(diff) / denom))
    return {"mse": mse, "mae": mae, "mape": mape}


def _save_bundle(
    out_path: str,
    model: CandleDecoder,
    params: ForecastTrainParams,
    feature_cols: List[str],
    target_cols: List[str],
    input_mu: np.ndarray,
    input_std: np.ndarray,
    target_mu: np.ndarray,
    target_std: np.ndarray,
) -> str:
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "input_mu": input_mu,
        "input_std": input_std,
        "target_mu": target_mu,
        "target_std": target_std,
        "window": params.window,
        "horizon": params.horizon,
        "arch": {
            "d_model": params.d_model,
            "nhead": params.nhead,
            "num_encoder_layers": params.num_encoder_layers,
            "num_decoder_layers": params.num_decoder_layers,
            "ff_dim": params.ff_dim,
            "dropout": params.dropout,
        },
        "meta": asdict(params),
    }
    torch.save(ckpt, out_path)
    # Mirror a JSON sidecar (without tensors) for quick reading
    side_json = {
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "input_mu": input_mu.tolist(),
        "input_std": input_std.tolist(),
        "target_mu": target_mu.tolist(),
        "target_std": target_std.tolist(),
        "window": params.window,
        "horizon": params.horizon,
        "arch": {
            "d_model": params.d_model,
            "nhead": params.nhead,
            "num_encoder_layers": params.num_encoder_layers,
            "num_decoder_layers": params.num_decoder_layers,
            "ff_dim": params.ff_dim,
            "dropout": params.dropout,
        },
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(side_json, f)
    return out_path


def train_transformer_forecaster(params: ForecastTrainParams) -> Dict[str, Any]:
    torch.manual_seed(int(params.seed))
    np.random.seed(int(params.seed))

    data_path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
    if not data_path:
        raise FileNotFoundError(
            f"No dataset found for {params.pair} {params.timeframe} in {params.userdir}/data"
        )
    raw = _load_ohlcv(data_path)
    raw = _slice_timerange_df(raw, params.timerange)

    train_ds, valid_ds, feature_cols, target_cols = _build_datasets(raw, params)

    device = torch.device(params.device if torch.cuda.is_available() or params.device == "cpu" else "cpu")
    model = CandleDecoder(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        d_model=params.d_model,
        nhead=params.nhead,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
        ff_dim=params.ff_dim,
        dropout=params.dropout,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=int(params.batch_size), shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=int(params.batch_size), shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(params.learning_rate), weight_decay=float(params.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=(params.device.startswith("cuda") and torch.cuda.is_available()))
    loss_fn = nn.SmoothL1Loss()  # Huber for robustness

    best_val = float("inf")
    best_ckpt: Optional[Dict[str, Any]] = None

    for epoch in range(int(params.epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)  # (B, T, F)
            y = batch["y"].to(device)  # (B, H, D)
            y_in = _prepare_decoder_inputs(y)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                y_hat = model(x, y_in)
                loss = loss_fn(y_hat, y)
            scaler.scale(loss).backward()
            if params.grad_clip_norm and params.grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(params.grad_clip_norm))
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            num_batches += 1
        avg_train = total_loss / max(1, num_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                y_in = _prepare_decoder_inputs(y)
                y_hat = model(x, y_in)
                loss = loss_fn(y_hat, y)
                val_loss += float(loss.item())
                val_batches += 1
        avg_val = val_loss / max(1, val_batches)

        print(f"EPOCH {epoch+1}/{params.epochs} train={avg_train:.6f} valid={avg_val:.6f}")
        if avg_val < best_val:
            best_val = avg_val
            best_ckpt = {
                "state_dict": model.state_dict(),
                "input_mu": train_ds.input_mu,
                "input_std": train_ds.input_std,
                "target_mu": train_ds.target_mu,
                "target_std": train_ds.target_std,
            }

    # Save best
    if best_ckpt is None:
        best_ckpt = {
            "state_dict": model.state_dict(),
            "input_mu": train_ds.input_mu,
            "input_std": train_ds.input_std,
            "target_mu": train_ds.target_mu,
            "target_std": train_ds.target_std,
        }
    model.load_state_dict(best_ckpt["state_dict"])  # ensure best
    out_path = _save_bundle(
        params.model_out_path,
        model,
        params,
        feature_cols,
        target_cols,
        best_ckpt["input_mu"],
        best_ckpt["input_std"],
        best_ckpt["target_mu"],
        best_ckpt["target_std"],
    )

    # Quick validation AR metrics (denormalized)
    model.eval()
    yh_list = []
    yt_list = []
    with torch.no_grad():
        for batch in valid_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            # start token = zeros (normalized)
            start = torch.zeros((x.size(0), 1, y.size(-1)), device=device, dtype=x.dtype)
            y_hat_norm = model.infer_autoregressive(x, start_token=start, horizon=y.size(1))
            # denormalize
            y_dn = (y.cpu().numpy() * valid_ds.target_std) + valid_ds.target_mu
            y_hat_dn = (y_hat_norm.cpu().numpy() * valid_ds.target_std) + valid_ds.target_mu
            yt_list.append(y_dn)
            yh_list.append(y_hat_dn)
            # Limit evaluation size for speed
            if len(yh_list) > 10:
                break
    if yh_list and yt_list:
        yt = np.concatenate(yt_list, axis=0)
        yh = np.concatenate(yh_list, axis=0)
        metrics = _metrics_denorm(yt, yh)
    else:
        metrics = {"mse": float("nan"), "mae": float("nan"), "mape": float("nan")}

    report = {
        "model_path": out_path,
        "valid_best_loss": best_val,
        "ar_metrics": metrics,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "window": params.window,
        "horizon": params.horizon,
    }
    print("FORECAST TRAIN SUMMARY:")
    print(report)
    return report


def load_forecaster(bundle_path: str, device: str = "cuda") -> Tuple[CandleDecoder, Dict[str, Any]]:
    ckpt = torch.load(bundle_path, map_location=device)
    arch = ckpt.get("arch", {})
    feature_cols: List[str] = ckpt["feature_columns"]
    target_cols: List[str] = ckpt["target_columns"]
    model = CandleDecoder(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        d_model=int(arch.get("d_model", 128)),
        nhead=int(arch.get("nhead", 4)),
        num_encoder_layers=int(arch.get("num_encoder_layers", 2)),
        num_decoder_layers=int(arch.get("num_decoder_layers", 2)),
        ff_dim=int(arch.get("ff_dim", 256)),
        dropout=float(arch.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "input_mu": ckpt["input_mu"],
        "input_std": ckpt["input_std"],
        "target_mu": ckpt["target_mu"],
        "target_std": ckpt["target_std"],
        "window": ckpt.get("window"),
        "horizon": ckpt.get("horizon"),
    }
    return model, info


@torch.no_grad()
def forecast_autoregressive(
    model: CandleDecoder,
    features_df: pd.DataFrame,
    info: Dict[str, Any],
    start_index: int,
    horizon: int,
) -> np.ndarray:
    """
    Run AR forecast on a single window starting at start_index.

    Args:
        model: loaded CandleDecoder
        features_df: dataframe with at least feature columns
        info: dict from load_forecaster
        start_index: index in dataframe such that window context ends at start_index (exclusive)
        horizon: steps to predict

    Returns:
        predictions denormalized: (H, D)
    """
    device = next(model.parameters()).device
    feat_cols: List[str] = info["feature_columns"]
    target_mu: np.ndarray = info["target_mu"]
    target_std: np.ndarray = info["target_std"]
    input_mu: np.ndarray = info["input_mu"]
    input_std: np.ndarray = info["input_std"]
    window: int = int(info["window"]) if info.get("window") is not None else 128

    x = features_df[feat_cols].astype(float).to_numpy(copy=False)
    i0 = start_index - window
    i1 = start_index
    if i0 < 0 or i1 > len(x):
        raise IndexError("Invalid start_index/window for forecast context.")
    x_ctx = x[i0:i1]
    x_ctx_n = (x_ctx - input_mu) / input_std
    x_tensor = torch.as_tensor(x_ctx_n, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, F)
    start = torch.zeros((1, 1, len(target_mu)), device=device, dtype=torch.float32)
    y_hat_n = model.infer_autoregressive(x_tensor, start_token=start, horizon=horizon)
    y_hat = (y_hat_n.cpu().numpy() * target_std) + target_mu
    return y_hat[0]


def evaluate_forecaster(
    model_path: str,
    userdir: str,
    pair: str,
    timeframe: str = "1h",
    feature_mode: str = "full",
    basic_lookback: int = 64,
    extra_timeframes: Optional[List[str]] = None,
    timerange: Optional[str] = None,
    device: str = "cuda",
    max_windows: int = 800,
    outdir: Optional[str] = None,
    make_animation: bool = False,
    anim_fps: int = 12,
    animation_mode: str = "next",  # 'next' (rolling next-step) | 'path' (rolling multistep path)
    prefer_exchange: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a trained forecaster and evaluate on a validation slice.
    Saves plots and CSVs with predictions in outdir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    # Load model bundle
    model, info = load_forecaster(model_path, device=device)
    feature_cols: List[str] = info["feature_columns"]
    target_cols: List[str] = info["target_columns"]
    window: int = int(info.get("window") or 128)
    horizon: int = int(info.get("horizon") or 16)

    # Data
    data_path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not data_path:
        raise FileNotFoundError("No dataset found for evaluation.")
    raw = _load_ohlcv(data_path)
    raw = _slice_timerange_df(raw, timerange)
    feats = make_features(
        raw,
        feature_columns=feature_cols,  # enforce training layout
        mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=extra_timeframes,
    )

    # Align evaluation slice: use tail portion
    n = len(feats)
    if n < (window + horizon + 10):
        raise ValueError("Dataset too short for evaluation.")
    start = max(window, n - max_windows - horizon)
    end = n - horizon

    # Select close column if present for visuals
    close_idx = target_cols.index("close") if "close" in target_cols else 0
    times = feats.index

    # Collect predictions and metrics
    y_true_1 = []  # next-step actual close
    y_pred_1 = []  # next-step predicted close
    y_true_all = []  # (H, D)
    y_pred_all = []  # (H, D)
    t_stamps = []

    for i in range(start, end):
        x_ctx_df = feats.iloc[i - window:i]
        y_future_df = feats.iloc[i:i + horizon]
        y_hat = forecast_autoregressive(model, feats, info, start_index=i, horizon=horizon)  # (H, D)
        y_true = y_future_df[target_cols].astype(float).to_numpy(copy=False)

        y_true_all.append(y_true)
        y_pred_all.append(y_hat)
        y_true_1.append(y_true[0, close_idx])
        y_pred_1.append(y_hat[0, close_idx])
        t_stamps.append(times[i])

    y_true_all_arr = np.asarray(y_true_all)  # (N, H, D)
    y_pred_all_arr = np.asarray(y_pred_all)  # (N, H, D)
    y_true_1_arr = np.asarray(y_true_1)
    y_pred_1_arr = np.asarray(y_pred_1)

    # Metrics
    metrics_all = _metrics_denorm(y_true_all_arr, y_pred_all_arr)
    # Direction accuracy on next-step close
    # Compare sign of next close change vs predicted change from last known close
    last_close_series = feats["close"].astype(float).to_numpy()
    last_known = np.asarray([last_close_series[i - 1] for i in range(start, end)])
    true_ret = np.sign(y_true_1_arr - last_known)
    pred_ret = np.sign(y_pred_1_arr - last_known)
    dir_acc = float(np.mean((true_ret == pred_ret).astype(float)))

    # Per-horizon MSE on close
    mse_per_h = []
    for h in range(horizon):
        diff = y_pred_all_arr[:, h, close_idx] - y_true_all_arr[:, h, close_idx]
        mse_per_h.append(float(np.mean(diff ** 2)))

    # Output directory
    base_dir = outdir or os.path.join(os.path.dirname(model_path) or ".", "forecast_eval")
    os.makedirs(base_dir, exist_ok=True)

    # Save CSV for 1-step
    csv_1 = os.path.join(base_dir, "next_step_close.csv")
    pd.DataFrame({
        "time": t_stamps,
        "last_close": last_known,
        "true_next_close": y_true_1_arr,
        "pred_next_close": y_pred_1_arr,
    }).to_csv(csv_1, index=False)

    # Plot 1-step predictions vs actual
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_stamps, y_true_1_arr, label="True next close", color="#2c3e50")
    ax.plot(t_stamps, y_pred_1_arr, label="Pred next close", color="#e67e22", alpha=0.9)
    ax.set_title(f"Next-step close forecast: {pair} {timeframe}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "next_step_close.png"), dpi=150)
    plt.close(fig)

    # Plot last window multi-step path (close)
    last_true = y_true_all_arr[-1, :, close_idx]
    last_pred = y_pred_all_arr[-1, :, close_idx]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(range(1, horizon + 1), last_true, label="True path", marker="o", color="#2ecc71")
    ax2.plot(range(1, horizon + 1), last_pred, label="Pred path", marker="x", color="#c0392b")
    ax2.set_xlabel("Steps ahead")
    ax2.set_ylabel("Close")
    ax2.set_title(f"Multi-step path at tail window: H={horizon}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(base_dir, "tail_multistep_close.png"), dpi=150)
    plt.close(fig2)

    # Optional animation (next-step rolling prediction or rolling path)
    anim_path = None
    if make_animation:
        from matplotlib.animation import FuncAnimation  # type: ignore
        if str(animation_mode).lower() == "path":
            # Rolling multistep path animation over horizon
            figp, axp = plt.subplots(figsize=(10, 5))
            axp.set_title(f"Rolling multistep path (close): H={horizon}")
            axp.set_xlabel("Steps ahead")
            axp.set_ylabel("Close")
            axp.grid(True, alpha=0.3)
            steps = np.arange(1, horizon + 1)
            line_true_p, = axp.plot([], [], label="True path", marker="o", color="#2ecc71")
            line_pred_p, = axp.plot([], [], label="Pred path", marker="x", color="#c0392b")
            axp.legend(loc="upper left")

            def init_p():
                line_true_p.set_data([], [])
                line_pred_p.set_data([], [])
                return line_true_p, line_pred_p

            def update_p(frame: int):
                i = min(frame, y_true_all_arr.shape[0] - 1)
                line_true_p.set_data(steps, y_true_all_arr[i, :, close_idx])
                line_pred_p.set_data(steps, y_pred_all_arr[i, :, close_idx])
                axp.relim()
                axp.autoscale_view()
                return line_true_p, line_pred_p

            ani = FuncAnimation(figp, update_p, frames=y_true_all_arr.shape[0], init_func=init_p, interval=1000.0/anim_fps, blit=True)
            anim_path = os.path.join(base_dir, "rolling_multistep_path.mp4")
            try:
                ani.save(anim_path, fps=anim_fps)
            except Exception:
                anim_path = os.path.join(base_dir, "rolling_multistep_path.gif")
                ani.save(anim_path, fps=anim_fps, writer="pillow")
            plt.close(figp)
        else:
            # Rolling next-step animation over time
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            ax3.set_title(f"Rolling next-step forecast: {pair} {timeframe}")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Close")
            ax3.grid(True, alpha=0.3)

            line_true, = ax3.plot([], [], label="True next close", color="#2c3e50")
            line_pred, = ax3.plot([], [], label="Pred next close", color="#e67e22")
            line_last, = ax3.plot([], [], label="Last close", color="#95a5a6", alpha=0.6)
            ax3.legend(loc="upper left")

            def init():
                line_true.set_data([], [])
                line_pred.set_data([], [])
                line_last.set_data([], [])
                return line_true, line_pred, line_last

            xs = np.asarray(t_stamps)
            def update(frame: int):
                i = max(1, frame)
                line_true.set_data(xs[:i], y_true_1_arr[:i])
                line_pred.set_data(xs[:i], y_pred_1_arr[:i])
                line_last.set_data(xs[:i], last_known[:i])
                ax3.relim()
                ax3.autoscale_view()
                return line_true, line_pred, line_last

            ani = FuncAnimation(fig3, update, frames=len(t_stamps), init_func=init, interval=1000.0/anim_fps, blit=True)
            anim_path = os.path.join(base_dir, "rolling_next_step.mp4")
            try:
                ani.save(anim_path, fps=anim_fps)
            except Exception:
                anim_path = os.path.join(base_dir, "rolling_next_step.gif")
                ani.save(anim_path, fps=anim_fps, writer="pillow")
            plt.close(fig3)

    report = {
        "model_path": model_path,
        "pair": pair,
        "timeframe": timeframe,
        "window": window,
        "horizon": horizon,
        "metrics": {
            **metrics_all,
            "direction_accuracy_next": dir_acc,
            "mse_per_h_close": mse_per_h,
        },
        "artifacts": {
            "next_step_csv": csv_1,
            "next_step_plot": os.path.join(base_dir, "next_step_close.png"),
            "tail_multistep_plot": os.path.join(base_dir, "tail_multistep_close.png"),
            "outdir": base_dir,
            "animation": anim_path,
        },
    }
    print("FORECAST EVAL SUMMARY:")
    print(report)
    return report



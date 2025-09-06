from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from .features import make_features
from .train_sb3 import _find_data_file, _load_ohlcv, _slice_timerange_df


class SequenceDataset(Dataset):
    """
    Produces windows of shape (T, F) from a features dataframe for reconstruction.
    X[i] = features[i-window+1:i+1, feature_cols]
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        window: int = 128,
        feature_columns: Optional[List[str]] = None,
        scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        if feature_columns is None:
            feature_columns = list(features_df.columns)
        self.feature_columns = list(feature_columns)
        self.window = int(window)

        X = features_df[self.feature_columns].astype(float).to_numpy(copy=False)
        if scaler is None:
            mu = X.mean(axis=0)
            std = X.std(axis=0)
            std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
            self.mu = mu
            self.std = std
        else:
            self.mu, self.std = scaler
        Xn = (X - self.mu) / (self.std + 1e-12)
        self.Xn = Xn
        self.num = max(0, len(Xn) - self.window + 1)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> torch.Tensor:
        j0 = idx
        j1 = idx + self.window
        x = self.Xn[j0:j1]
        # (T,F) -> (F,T) for Conv1d input (B,C,L)
        return torch.from_numpy(x.astype(np.float32)).transpose(0, 1)


class Conv1dAutoEncoder(nn.Module):
    """
    1D Conv Autoencoder for time-series feature windows.
    Input: (B, F, T)
    Encoder: Conv1d + ReLU + MaxPool stacks → bottleneck Dense → embedding (E)
    Decoder: Upsampling + Conv1d stacks → reconstruct (B, F, T)
    """

    def __init__(self, num_features: int, window: int, embed_dim: int = 16, base_channels: int = 32):
        super().__init__()
        self.num_features = int(num_features)
        self.window = int(window)
        self.embed_dim = int(embed_dim)

        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(self.num_features, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=max(1, self.window // 8)),
        )

        # Infer flattened size after enc
        with torch.no_grad():
            tmp = torch.zeros(1, self.num_features, self.window)
            h = self.enc(tmp)
            flat = int(h.shape[1] * h.shape[2])
        self._enc_out_flat = flat
        self.fc_mu = nn.Linear(flat, self.embed_dim)
        self.fc_dec = nn.Linear(self.embed_dim, flat)

        # Decoder
        self.dec_c = c3
        self.dec_len = int(h.shape[2])
        self.dec = nn.Sequential(
            nn.Conv1d(c3, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c1, self.num_features, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(h.size(0), self.dec_c, self.dec_len)
        y = self.dec(h)
        # Adjust length to window if needed
        if y.size(2) != self.window:
            if y.size(2) > self.window:
                y = y[:, :, : self.window]
            else:
                pad = self.window - y.size(2)
                y = nn.functional.pad(y, (0, pad))
        return y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return y, z


@dataclass
class AETrainParams:
    pair: str = "BTC/USDT"
    timeframe: str = "1h"
    userdir: str = "freqtrade_userdir"
    timerange: str = "20190101-"
    prefer_exchange: str = "bybit"
    feature_mode: str = "full"
    basic_lookback: int = 64
    extra_timeframes: str = "4H,1D,1W"
    # Raw multi-HTF mode (no indicators): if True, build inputs from raw OHLCV resampled across TFs
    raw_htf: bool = False
    raw_extra_timeframes: str = "4H,1D,1W"
    ae_cols: str = "close,volume"  # which base columns to include across TFs
    window: int = 128
    embed_dim: int = 16
    base_channels: int = 32
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"
    out_path: str = "models/xgb_stack/ae_conv1d.json"


def _resolve_device(dev: str) -> torch.device:
    if dev in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_raw_multitf_features(
    df: pd.DataFrame,
    extra_timeframes: Optional[List[str]] = None,
    base_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a raw OHLCV + HTF OHLCV matrix aligned to base index. No indicators.
    Columns:
      base: 'open','high','low','close','volume' (subset selectable via base_cols)
      for each TF in extra_timeframes: 'TF_open','TF_high','TF_low','TF_close','TF_volume'
    """
    if base_cols is None:
        base_cols = ["open","high","low","close","volume"]
    base = df[[c for c in ["open","high","low","close","volume"] if c in df.columns]].copy()
    base = base.astype(float)
    out = pd.DataFrame(index=base.index)
    for c in base_cols:
        if c in base.columns:
            out[c] = base[c].values
        else:
            out[c] = 0.0
    if extra_timeframes:
        if isinstance(base.index, pd.DatetimeIndex):
            for tf in extra_timeframes:
                tfs = str(tf).upper()
                # Pandas alias normalization: H expects lowercase
                resample_tf = tfs[:-1] + 'h' if tfs.endswith('H') else tfs
                try:
                    agg = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    res = base.resample(resample_tf).agg(agg).dropna(how="all")
                    res_aligned = res.reindex(base.index).ffill().fillna(0.0)
                    for c in ["open","high","low","close","volume"]:
                        nm = f"{tfs}_{c}"
                        if c in res_aligned.columns:
                            out[nm] = res_aligned[c].astype(float).values
                        else:
                            out[nm] = 0.0
                except Exception:
                    # Skip TF on failure
                    for c in ["open","high","low","close","volume"]:
                        out[f"{tfs}_{c}"] = 0.0
        else:
            # Non-datetime index; only base available
            pass
    # Sanitize
    out = out.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return out


def train_autoencoder(params: AETrainParams) -> Dict[str, Any]:
    if not _TORCH_OK:
        raise ImportError("PyTorch required to train Autoencoder")

    path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
    if not path:
        raise FileNotFoundError("Dataset not found. Download via Freqtrade.")
    raw = _load_ohlcv(path)
    raw = _slice_timerange_df(raw, params.timerange)
    # Build feature matrix per mode
    if bool(params.raw_htf):
        etf = [s.strip() for s in (params.raw_extra_timeframes or "").split(",") if s.strip()]
        base_cols = [s.strip() for s in (params.ae_cols or "close,volume").split(",") if s.strip()]
        feats = build_raw_multitf_features(raw, extra_timeframes=etf, base_cols=base_cols)
        feats = feats.reset_index(drop=True)
        # Select training columns: base_cols + TF-prefixed for those columns
        feature_cols = []
        for c in base_cols:
            if c in feats.columns:
                feature_cols.append(c)
        for tf in etf:
            tfs = str(tf).upper()
            for c in ["open","high","low","close","volume"]:
                if c in base_cols:
                    nm = f"{tfs}_{c}"
                    if nm in feats.columns:
                        feature_cols.append(nm)
        if not feature_cols:
            feature_cols = list(feats.columns)
        ds = SequenceDataset(feats, window=int(params.window), feature_columns=feature_cols)
    else:
        etf = [s.strip() for s in (params.extra_timeframes or "").split(",") if s.strip()]
        feats = make_features(
            raw,
            mode=(params.feature_mode or "full"),
            basic_lookback=int(params.basic_lookback),
            extra_timeframes=(etf or None),
        ).reset_index(drop=True)
        # Choose numeric feature columns
        feature_cols = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
        ds = SequenceDataset(feats, window=int(params.window), feature_columns=feature_cols)

    # Train/val split by time
    n = len(ds)
    if n < 200:
        raise ValueError("Not enough windows for AE training")
    cut = int(max(100, min(n - 50, int(n * 0.8))))
    tr_idx = list(range(0, cut))
    va_idx = list(range(cut, n))

    class Subset(Dataset):
        def __init__(self, parent: SequenceDataset, indices: List[int]):
            self.parent = parent
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i: int):
            return self.parent[self.indices[i]]

    tr_ds = Subset(ds, tr_idx)
    va_ds = Subset(ds, va_idx)
    tr_ld = DataLoader(tr_ds, batch_size=int(max(1, params.batch_size)), shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=512, shuffle=False)

    dev = _resolve_device(params.device)
    model = Conv1dAutoEncoder(num_features=len(ds.feature_columns), window=int(params.window), embed_dim=int(params.embed_dim), base_channels=int(params.base_channels)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(params.lr), weight_decay=float(params.weight_decay))
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state: Dict[str, Any] | None = None
    for ep in range(int(max(1, params.epochs))):
        model.train()
        tr_loss = 0.0
        nb = 0
        for xb in tr_ld:
            xb = xb.to(dev)
            opt.zero_grad()
            yb, _zb = model(xb)
            loss = crit(yb, xb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach().cpu().item())
            nb += 1
        tr_loss = tr_loss / max(1, nb)

        model.eval()
        va_loss = 0.0
        nb2 = 0
        with torch.no_grad():
            for xb in va_ld:
                xb = xb.to(dev)
                yb, _zb = model(xb)
                loss = crit(yb, xb)
                va_loss += float(loss.detach().cpu().item())
                nb2 += 1
        va_loss = va_loss / max(1, nb2)
        if np.isfinite(va_loss) and va_loss < best_val:
            best_val = va_loss
            best_state = {"model": model.state_dict()}

    if best_state is None:
        best_state = {"model": model.state_dict()}

    os.makedirs(os.path.dirname(params.out_path) or ".", exist_ok=True)
    weights_path = str(os.path.splitext(params.out_path)[0]) + ".pt"
    torch.save(best_state, weights_path)
    meta = {
        "type": "Conv1dAE",
        "weights_path": weights_path,
        "feature_columns": list(ds.feature_columns),
        "window": int(params.window),
        "embed_dim": int(params.embed_dim),
        "base_channels": int(params.base_channels),
        "input_mu": ds.mu.tolist(),
        "input_std": ds.std.tolist(),
        "val_mse": float(best_val),
        "train_params": asdict(params),
        "raw_htf": bool(params.raw_htf),
        "raw_extra_timeframes": [s.strip() for s in (params.raw_extra_timeframes or "").split(",") if s.strip()],
        "ae_base_cols": [s.strip() for s in (params.ae_cols or "close,volume").split(",") if s.strip()],
    }
    with open(params.out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_autoencoder(manifest_path: str, device: str = "auto") -> Tuple[Conv1dAutoEncoder, Dict[str, Any]]:
    import json as _json
    with open(manifest_path, "r") as f:
        man = _json.load(f)
    if man.get("type") != "Conv1dAE":
        raise ValueError("Manifest is not Conv1dAE")
    window = int(man["window"])
    embed_dim = int(man["embed_dim"])
    base_channels = int(man.get("base_channels", 32))
    feat_cols = list(man.get("feature_columns", []))
    num_features = len(feat_cols)
    dev = _resolve_device(device)
    model = Conv1dAutoEncoder(num_features=num_features, window=window, embed_dim=embed_dim, base_channels=base_channels).to(dev)
    state = torch.load(str(man["weights_path"]), map_location=dev)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, man


def compute_embeddings(
    feats: pd.DataFrame,
    ae_manifest_path: str,
    device: str = "auto",
    out_col_prefix: str = "ae",
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Slide a window over the features and compute embedding for each step (align to last index of window).
    Returns a DataFrame aligned to feats.index with columns f"{out_col_prefix}_{i}".
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch required for embeddings")
    model, man = load_autoencoder(ae_manifest_path, device=device)
    feat_cols = list(man.get("feature_columns", []))
    if not feat_cols:
        feat_cols = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
    W = int(window or man["window"])
    mu = np.asarray(man.get("input_mu", np.zeros(len(feat_cols))), dtype=float)
    std = np.asarray(man.get("input_std", np.ones(len(feat_cols))), dtype=float)
    X = feats[feat_cols].astype(float).to_numpy(copy=False)
    Xn = (X - mu) / (std + 1e-12)
    T = Xn.shape[0]
    if T < W:
        # Not enough rows, return zeros
        cols = [f"{out_col_prefix}_{i}" for i in range(int(model.embed_dim))]
        return pd.DataFrame({c: np.zeros(T, dtype=float) for c in cols}, index=feats.index)
    dev = next(model.parameters()).device
    embs: List[np.ndarray] = []
    bs = 1024
    idxs = list(range(W - 1, T))
    # Batch windows
    windows: List[torch.Tensor] = []
    for i in idxs:
        x = Xn[i - W + 1 : i + 1]
        xt = torch.from_numpy(x.astype(np.float32)).transpose(0, 1).unsqueeze(0)
        windows.append(xt)
        if len(windows) >= bs:
            xb = torch.cat(windows, dim=0).to(dev)
            with torch.no_grad():
                _y, z = model(xb)
            embs.append(z.detach().cpu().numpy())
            windows = []
    if windows:
        xb = torch.cat(windows, dim=0).to(dev)
        with torch.no_grad():
            _y, z = model(xb)
        embs.append(z.detach().cpu().numpy())
    Z = np.concatenate(embs, axis=0) if embs else np.zeros((0, int(model.embed_dim)), dtype=float)
    # Align back to T by padding initial W-1 rows with zeros
    Z_full = np.zeros((T, Z.shape[1] if Z.size else int(model.embed_dim)), dtype=float)
    if Z.size:
        Z_full[W - 1 : , :] = Z
    cols = [f"{out_col_prefix}_{i}" for i in range(Z_full.shape[1])]
    out = pd.DataFrame(Z_full, index=feats.index, columns=cols)
    return out


def compute_embeddings_from_raw(
    raw: pd.DataFrame,
    ae_manifest_path: str,
    device: str = "auto",
    out_col_prefix: str = "ae",
) -> pd.DataFrame:
    """
    Compute embeddings using raw multi-HTF OHLCV per AE manifest. No indicators.
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch required for embeddings")
    model, man = load_autoencoder(ae_manifest_path, device=device)
    etf = list(man.get("raw_extra_timeframes", []))
    base_cols = list(man.get("ae_base_cols", ["close","volume"]))
    feats = build_raw_multitf_features(raw, extra_timeframes=etf, base_cols=base_cols)
    # Ensure we select exactly the trained columns
    feat_cols = list(man.get("feature_columns", list(feats.columns)))
    feats = feats.reindex(columns=feat_cols)
    # Now reuse compute_embeddings logic by passing constructed feats
    return compute_embeddings(feats, ae_manifest_path, device=device, out_col_prefix=out_col_prefix, window=None)



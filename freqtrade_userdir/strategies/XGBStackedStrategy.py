from __future__ import annotations

from typing import Optional, Dict, List
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

# Ensure project root is importable for rl_lib
_userdir_root = Path(__file__).resolve().parents[1]
_project_root = Path(__file__).resolve().parents[2]
for _p in (str(_userdir_root), str(_project_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_lib.features import make_features
from rl_lib.meta import load_meta_mlp_from_json
try:
    # Optional AE embeddings
    from rl_lib.autoencoder import compute_embeddings, compute_embeddings_from_raw
    _AE_OK = True
except Exception:
    _AE_OK = False


class XGBStackedStrategy(IStrategy):
    timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 512
    can_short = True

    minimal_roi: Dict = {"0": 1000}
    stoploss = -0.99
    trailing_stop = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # L0 model paths
        self.bottom_model_path = os.environ.get("XGB_BOTTOM_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_topbot_bottom.json"))
        self.top_model_path = os.environ.get("XGB_TOP_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_topbot_top.json"))
        self.logret_path = os.environ.get("XGB_LOGRET_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_logret.json"))
        self.up_model_path = os.environ.get("XGB_UP_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_impulse_up.json"))
        self.down_model_path = os.environ.get("XGB_DN_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_impulse_down.json"))
        # Meta model manifest (MLP)
        self.meta_json_path = os.environ.get("XGB_META_PATH", str(Path(_project_root) / "models" / "xgb_stack" / "best_meta.json"))

        # Thresholds
        self.p_buy_thr = float(os.environ.get("XGB_P_BUY_THR", "0.6"))
        self.p_sell_thr = float(os.environ.get("XGB_P_SELL_THR", "0.6"))
        self.meta_thr = float(os.environ.get("META_THR", "0.5"))
        self.min_hold_bars = int(os.environ.get("XGB_MIN_HOLD", "8"))
        self.cooldown_bars = int(os.environ.get("XGB_COOLDOWN", "0"))
        # Feature config
        self.feature_mode = os.environ.get("XGB_FEAT_MODE", "full")
        self.extra_timeframes = [s.strip() for s in os.environ.get("XGB_EXTRA_TFS", "4h,1d").split(',') if s.strip()]
        self.device = os.environ.get("XGB_DEVICE", "auto")
        # Mode: which model to backtest: stacked|topbot|logret|impulse
        self.mode = os.environ.get("XGB_MODE", "stacked").strip().lower()
        # Additional thresholds for specific modes
        self.logret_dir_thr = float(os.environ.get("LOGRET_DIR_THR", "0.0"))
        self.p_up_thr = float(os.environ.get("XGB_P_UP_THR", os.environ.get("XGB_P_BUY_THR", str(self.p_buy_thr))))
        self.p_dn_thr = float(os.environ.get("XGB_P_DN_THR", os.environ.get("XGB_P_SELL_THR", str(self.p_sell_thr))))

        # Optional AE manifest for embeddings
        self.ae_path = os.environ.get("XGB_AE_PATH", "").strip()
        self._ae_manifest: Optional[dict] = None
        self.ae_debug = bool(int(os.environ.get("XGB_AE_DEBUG", "0") or "0"))

        # Lazy loaded models
        self._bot = None
        self._top = None
        self._logret = None
        self._upc = None
        self._dnc = None
        self._meta = None
        self._bot_cols: Optional[List[str]] = None
        self._top_cols: Optional[List[str]] = None
        self._logret_cols: Optional[List[str]] = None
        self._up_cols: Optional[List[str]] = None
        self._dn_cols: Optional[List[str]] = None
        self._meta_cols: Optional[List[str]] = None

    def _resolve_device(self) -> str:
        s = str(self.device).lower()
        if s in ("auto", "cuda"):
            try:
                import cupy as _cp  # type: ignore
                _ = _cp.zeros(1)
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"

    def _load_models(self):
        import json
        import xgboost as xgb
        dev = self._resolve_device()
        # Bottom
        if self._bot is None and self.bottom_model_path:
            clf = xgb.XGBClassifier(device=dev)
            clf.load_model(self.bottom_model_path)
            self._bot = clf
            fc = Path(self.bottom_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._bot_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._bot_cols = None
        # Top
        if self._top is None and self.top_model_path:
            clf = xgb.XGBClassifier(device=dev)
            clf.load_model(self.top_model_path)
            self._top = clf
            fc = Path(self.top_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._top_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._top_cols = None
        # Logret
        if self._logret is None and self.logret_path:
            clf = xgb.XGBClassifier(device=dev)
            clf.load_model(self.logret_path)
            self._logret = clf
            fc = Path(self.logret_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._logret_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._logret_cols = None
        # Impulse up
        if self._upc is None and self.up_model_path and os.path.exists(self.up_model_path):
            try:
                upc = xgb.XGBClassifier(device=dev)
                upc.load_model(self.up_model_path)
                self._upc = upc
                fc = Path(self.up_model_path).with_suffix("").as_posix() + "_feature_columns.json"
                if os.path.exists(fc):
                    try:
                        self._up_cols = json.loads(Path(fc).read_text())
                    except Exception:
                        self._up_cols = None
            except Exception:
                self._upc = None
        # Impulse down
        if self._dnc is None and self.down_model_path and os.path.exists(self.down_model_path):
            try:
                dnc = xgb.XGBClassifier(device=dev)
                dnc.load_model(self.down_model_path)
                self._dnc = dnc
                fc = Path(self.down_model_path).with_suffix("").as_posix() + "_feature_columns.json"
                if os.path.exists(fc):
                    try:
                        self._dn_cols = json.loads(Path(fc).read_text())
                    except Exception:
                        self._dn_cols = None
            except Exception:
                self._dnc = None
        # Meta MLP
        if self._meta is None and self.meta_json_path and os.path.exists(self.meta_json_path):
            try:
                meta_model, meta_cols = load_meta_mlp_from_json(self.meta_json_path)
                self._meta = meta_model
                self._meta_cols = meta_cols
            except Exception:
                self._meta = None

    def _view(self, df: pd.DataFrame, cols: Optional[List[str]]) -> Optional[np.ndarray]:
        if cols is None:
            return None
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out.reindex(columns=cols).values

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Load models
        self._load_models()
        # Union feature columns for L0 models
        union_cols: Optional[List[str]] = None
        for cols in (self._bot_cols, self._top_cols, self._logret_cols, self._up_cols, self._dn_cols):
            if cols:
                if union_cols is None:
                    union_cols = list(cols)
                else:
                    for c in cols:
                        if c not in union_cols:
                            union_cols.append(c)
        feats = make_features(df, feature_columns=union_cols, mode=self.feature_mode, extra_timeframes=self.extra_timeframes)
        feats = feats.reset_index(drop=True)

        # Append AE embeddings if configured
        if self.ae_path and _AE_OK:
            try:
                import json as _json
                if self._ae_manifest is None:
                    with open(self.ae_path, 'r') as f:
                        self._ae_manifest = _json.load(f)
                man = self._ae_manifest or {}
                if bool(man.get("raw_htf", False)):
                    ae_df = compute_embeddings_from_raw(df, ae_manifest_path=self.ae_path, device=self._resolve_device(), out_col_prefix="ae")
                    ae_df = ae_df.reindex(index=feats.index).fillna(0.0)
                else:
                    # Build a full feature matrix for AE inputs (may require more columns than union_cols)
                    full_feats = make_features(df, feature_columns=None, mode=self.feature_mode, extra_timeframes=self.extra_timeframes)
                    full_feats = full_feats.reset_index(drop=True)
                    # Normalize TF prefixes to uppercase (e.g., 4h_ -> 4H_) to match AE manifest
                    try:
                        import re as _re
                        rename_map = {}
                        for _c in list(full_feats.columns):
                            m = _re.match(r"^([0-9]+)([hdwHDW])_(.+)$", _c)
                            if m:
                                num, unit, rest = m.group(1), m.group(2), m.group(3)
                                _new = f"{num}{unit.upper()}_{rest}"
                                if _new != _c:
                                    rename_map[_c] = _new
                        if rename_map:
                            full_feats = full_feats.rename(columns=rename_map)
                    except Exception:
                        pass
                    ae_df = compute_embeddings(full_feats, ae_manifest_path=self.ae_path, device=self._resolve_device(), out_col_prefix="ae", window=None)
                feats = feats.join(ae_df, how="left")
                if self.ae_debug:
                    try:
                        import numpy as _np
                        ae_cols = [c for c in feats.columns if c.startswith("ae_")]
                        nz = int(_np.sum(_np.any(feats[ae_cols].to_numpy() != 0.0, axis=1))) if ae_cols else 0
                        print(f"[XGBStackedStrategy] AE applied: cols={len(ae_cols)} nonzero_rows={nz}/{len(feats)}")
                    except Exception:
                        pass
            except Exception as e:
                # Fail silent if AE unavailable
                if self.ae_debug:
                    try:
                        print(f"[XGBStackedStrategy] AE failed: {e}")
                    except Exception:
                        pass

        # Predict L0 outputs
        p_bottom = None
        p_top = None
        pr_logret = None
        p_up = None
        p_dn = None
        if self._bot is not None and self._bot_cols is not None:
            Xb = self._view(feats, self._bot_cols)
            try:
                p = self._bot.predict_proba(Xb)
                p_bottom = p[:, 1]
            except Exception:
                p_bottom = None
        if self._top is not None and self._top_cols is not None:
            Xt = self._view(feats, self._top_cols)
            try:
                p = self._top.predict_proba(Xt)
                p_top = p[:, 1]
            except Exception:
                p_top = None
        if self._logret is not None and self._logret_cols is not None:
            Xr = self._view(feats, self._logret_cols)
            try:
                pr_logret = self._logret.predict_proba(Xr)
            except Exception:
                pr_logret = None
        if self._upc is not None and self._up_cols is not None:
            Xu = self._view(feats, self._up_cols)
            try:
                p_up = self._upc.predict_proba(Xu)[:, 1]
            except Exception:
                p_up = None
        if self._dnc is not None and self._dn_cols is not None:
            Xd = self._view(feats, self._dn_cols)
            try:
                p_dn = self._dnc.predict_proba(Xd)[:, 1]
            except Exception:
                p_dn = None

        T = len(feats)
        if p_bottom is None:
            p_bottom = np.zeros(T)
        if p_top is None:
            p_top = np.zeros(T)
        if pr_logret is None or pr_logret.ndim != 2 or pr_logret.shape[1] != 5:
            pr_logret = np.zeros((T, 5))
        if p_up is None:
            p_up = np.zeros(T)
        if p_dn is None:
            p_dn = np.zeros(T)

        # reg_direction proxy from class probabilities
        class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
        reg_dir = pr_logret @ class_vals

        # Build meta input rows following manifest column order
        meta_X = None
        meta_p = None
        if self._meta is not None and self._meta_cols is not None:
            # Build a dataframe of L0 outputs with expected column names
            l0 = pd.DataFrame({
                "p_bottom": p_bottom,
                "p_top": p_top,
                "p_up": p_up,
                "p_dn": p_dn,
                "reg_direction": reg_dir,
                "logret_p_-2": pr_logret[:, 0],
                "logret_p_-1": pr_logret[:, 1],
                "logret_p_0": pr_logret[:, 2],
                "logret_p_1": pr_logret[:, 3],
                "logret_p_2": pr_logret[:, 4],
            })
            # Align to meta columns (pad missing with zeros)
            for c in self._meta_cols:
                if c not in l0.columns:
                    l0[c] = 0.0
            meta_X = l0.reindex(columns=self._meta_cols).values.astype(np.float32)
            try:
                import torch
                with torch.no_grad():
                    logits = self._meta(torch.from_numpy(meta_X))
                    meta_p = torch.sigmoid(logits).cpu().numpy()
            except Exception:
                meta_p = None

        # Build trading signals according to selected mode
        mode = self.mode
        buy_sig = np.zeros(T, dtype=bool)
        sell_sig = np.zeros(T, dtype=bool)
        if mode in ("stacked", "stack", "meta"):
            buy_sig = (p_bottom >= float(self.p_buy_thr))
            sell_sig = (p_top >= float(self.p_sell_thr))
            if meta_p is not None:
                thr = float(self.meta_thr)
                buy_sig &= (meta_p >= thr).astype(bool)
                sell_sig &= (meta_p >= thr).astype(bool)
        elif mode == "topbot":
            buy_sig = (p_bottom >= float(self.p_buy_thr))
            sell_sig = (p_top >= float(self.p_sell_thr))
        elif mode == "logret":
            thr = float(self.logret_dir_thr)
            buy_sig = (reg_dir >= thr)
            sell_sig = (reg_dir <= -thr)
        elif mode == "impulse":
            buy_sig = (p_up >= float(self.p_up_thr))
            sell_sig = (p_dn >= float(self.p_dn_thr))
        else:
            # Fallback to stacked
            buy_sig = (p_bottom >= float(self.p_buy_thr))
            sell_sig = (p_top >= float(self.p_sell_thr))
            if meta_p is not None:
                thr = float(self.meta_thr)
                buy_sig &= (meta_p >= thr).astype(bool)
                sell_sig &= (meta_p >= thr).astype(bool)

        enter_long = np.zeros(T, dtype=int)
        exit_long = np.zeros(T, dtype=int)
        enter_short = np.zeros(T, dtype=int)
        exit_short = np.zeros(T, dtype=int)

        last_change = -10**9
        cooldown_until = -1
        pos = 0
        for t in range(T):
            desired = 0
            if buy_sig[t] and not sell_sig[t]:
                desired = +1
            elif sell_sig[t] and not buy_sig[t]:
                desired = -1
            else:
                desired = 0

            can_close = (t - last_change) >= max(0, self.min_hold_bars)
            can_open = t >= cooldown_until

            if desired == +1 and pos <= 0 and can_open:
                if pos == -1:
                    exit_short[t] = 1
                enter_long[t] = 1
                pos = +1
                last_change = t
            elif desired == -1 and pos >= 0 and can_open:
                if pos == +1:
                    exit_long[t] = 1
                enter_short[t] = 1
                pos = -1
                last_change = t
            elif desired == 0 and pos != 0 and can_close:
                if pos == +1:
                    exit_long[t] = 1
                elif pos == -1:
                    exit_short[t] = 1
                pos = 0
                last_change = t
                if self.cooldown_bars > 0:
                    cooldown_until = t + self.cooldown_bars

        df["enter_long"] = enter_long
        df["exit_long"] = exit_long
        df["enter_short"] = enter_short
        df["exit_short"] = exit_short

        # Diagnostics
        df["p_bottom"] = p_bottom
        df["p_top"] = p_top
        df["meta_p"] = (meta_p if meta_p is not None else np.zeros(T))
        df["reg_direction"] = reg_dir
        for i, name in enumerate(["lr_p_-2","lr_p_-1","lr_p_0","lr_p_1","lr_p_2"]):
            df[name] = pr_logret[:, i]
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'enter_long'] = (df.get('enter_long', 0) == 1).astype('int')
        df.loc[:, 'enter_short'] = (df.get('enter_short', 0) == 1).astype('int')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'exit_long'] = (df.get('exit_long', 0) == 1).astype('int')
        df.loc[:, 'exit_short'] = (df.get('exit_short', 0) == 1).astype('int')
        return df



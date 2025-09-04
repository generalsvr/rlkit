from __future__ import annotations

from functools import reduce
from typing import Dict, Optional
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


class XGBImpulseStrategy(IStrategy):
    timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 512
    can_short = True

    # Neutral ROI/SL, exits controlled by signals
    minimal_roi: Dict = {"0": 1000}
    stoploss = -0.99
    trailing_stop = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Model paths via env to avoid hardcoding
        self.regressor_path = os.environ.get("XGB_REG_PATH", "")
        self.up_model_path = os.environ.get("XGB_UP_PATH", "")
        self.down_model_path = os.environ.get("XGB_DN_PATH", "")
        # Thresholds and gating
        self.p_up_thr = float(os.environ.get("XGB_P_UP_THR", "0.8"))
        self.p_dn_thr = float(os.environ.get("XGB_P_DN_THR", "0.8"))
        self.p_margin = float(os.environ.get("XGB_P_MARGIN", "0.15"))
        self.min_hold_bars = int(os.environ.get("XGB_MIN_HOLD", "8"))
        self.cooldown_bars = int(os.environ.get("XGB_COOLDOWN", "0"))
        self.impulse_horizon = int(os.environ.get("XGB_IMPULSE_H", "7"))
        # Feature config
        self.feature_mode = os.environ.get("XGB_FEAT_MODE", "full")
        # Default to the HTFs used for training (adjust as needed)
        self.extra_timeframes = [s.strip() for s in os.environ.get("XGB_EXTRA_TFS", "4h,1d").split(',') if s.strip()]

        # Lazy model loading
        self._reg = None
        self._upc = None
        self._dnc = None
        self._reg_cols: Optional[list[str]] = None
        self._up_cols: Optional[list[str]] = None
        self._dn_cols: Optional[list[str]] = None

    def _load_xgb(self):
        import json
        import xgboost as xgb
        device = os.environ.get("XGB_DEVICE", "auto").lower()
        dev = "cuda" if device in ("auto", "cuda") else "cpu"
        # Regressor
        if self.regressor_path and self._reg is None:
            reg = xgb.XGBRegressor(device=dev)
            reg.load_model(self.regressor_path)
            self._reg = reg
            fc = Path(self.regressor_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._reg_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._reg_cols = None
        # Up
        if self.up_model_path and self._upc is None:
            upc = xgb.XGBClassifier(device=dev)
            upc.load_model(self.up_model_path)
            self._upc = upc
            fc = Path(self.up_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._up_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._up_cols = None
        # Down
        if self.down_model_path and self._dnc is None:
            dnc = xgb.XGBClassifier(device=dev)
            dnc.load_model(self.down_model_path)
            self._dnc = dnc
            fc = Path(self.down_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._dn_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._dn_cols = None

    def _view(self, df: pd.DataFrame, cols: Optional[list[str]]) -> Optional[np.ndarray]:
        if cols is None:
            return None
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out.reindex(columns=cols).values

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Load models once
        self._load_xgb()

        # Build features; union of available model cols to avoid missing inputs
        union_cols: Optional[list[str]] = None
        for cols in (self._reg_cols, self._up_cols, self._dn_cols):
            if cols:
                if union_cols is None:
                    union_cols = list(cols)
                else:
                    for c in cols:
                        if c not in union_cols:
                            union_cols.append(c)
        feats = make_features(df, feature_columns=union_cols, mode=self.feature_mode, extra_timeframes=self.extra_timeframes)
        feats = feats.reset_index(drop=True)

        # Predict
        mu = None
        p_up = None
        p_dn = None
        if self._reg is not None and self._reg_cols is not None:
            Xr = self._view(feats, self._reg_cols)
            try:
                mu = self._reg.predict(Xr)
            except Exception:
                mu = None
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

        n = len(feats)
        enter_long = np.zeros(n, dtype=int)
        exit_long = np.zeros(n, dtype=int)
        enter_short = np.zeros(n, dtype=int)
        exit_short = np.zeros(n, dtype=int)

        # Simple gating: thresholds + margin; optional regressor direction confirmation
        reg_gate = True
        if mu is not None:
            reg_gate_up = mu > 0.0
            reg_gate_dn = mu < 0.0
        else:
            reg_gate_up = np.ones(n, dtype=bool)
            reg_gate_dn = np.ones(n, dtype=bool)

        up_ok = np.zeros(n, dtype=bool)
        dn_ok = np.zeros(n, dtype=bool)
        if p_up is not None:
            up_ok = p_up >= float(self.p_up_thr)
        if p_dn is not None:
            dn_ok = p_dn >= float(self.p_dn_thr)
        if p_up is not None and p_dn is not None and float(self.p_margin) > 0.0:
            up_ok &= (p_up - p_dn) >= float(self.p_margin)
            dn_ok &= (p_dn - p_up) >= float(self.p_margin)
        up_ok &= reg_gate_up
        dn_ok &= reg_gate_dn

        # Convert to entry/exit schedule with min-hold and cooldown to avoid churn
        last_change = -10**9
        cooldown_until = -1
        pos = 0
        for t in range(n):
            # Execute desired at bar open: convert to enter/exit flags
            desired = None
            # Close if both fire or none: prefer neutral unless clear signal
            if up_ok[t] and not dn_ok[t]:
                desired = +1
            elif dn_ok[t] and not up_ok[t]:
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

        # Assign signals into dataframe
        df["enter_long"] = enter_long
        df["exit_long"] = exit_long
        df["enter_short"] = enter_short
        df["exit_short"] = exit_short
        # Optional diagnostic columns
        if mu is not None:
            df["mu_pred"] = mu
        if p_up is not None:
            df["p_up"] = p_up
        if p_dn is not None:
            df["p_dn"] = p_dn
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Map indicators to Freqtrade entry flags
        df.loc[:, 'enter_long'] = (df.get('enter_long', 0) == 1).astype('int')
        df.loc[:, 'enter_short'] = (df.get('enter_short', 0) == 1).astype('int')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'exit_long'] = (df.get('exit_long', 0) == 1).astype('int')
        df.loc[:, 'exit_short'] = (df.get('exit_short', 0) == 1).astype('int')
        return df



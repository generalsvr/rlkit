from __future__ import annotations

from typing import Optional, Dict
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


class XGBTopBotStrategy(IStrategy):
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
        # Model paths via env
        self.bottom_model_path = (
            os.environ.get("XGB_BOTTOM_PATH")
            or os.environ.get("XGB_UP_PATH", "")
        )
        self.top_model_path = (
            os.environ.get("XGB_TOP_PATH")
            or os.environ.get("XGB_DN_PATH", "")
        )
        # Thresholds and gating
        self.p_buy_thr = float(os.environ.get("XGB_P_UP_THR", os.environ.get("XGB_P_BUY_THR", "0.6")))
        self.p_sell_thr = float(os.environ.get("XGB_P_DN_THR", os.environ.get("XGB_P_SELL_THR", "0.6")))
        self.p_margin = float(os.environ.get("XGB_P_MARGIN", "0.0"))
        self.min_hold_bars = int(os.environ.get("XGB_MIN_HOLD", "8"))
        self.cooldown_bars = int(os.environ.get("XGB_COOLDOWN", "0"))
        # Feature config
        self.feature_mode = os.environ.get("XGB_FEAT_MODE", "full")
        self.extra_timeframes = [s.strip() for s in os.environ.get("XGB_EXTRA_TFS", "4h,1d").split(',') if s.strip()]

        # Lazy model loading
        self._bot = None
        self._top = None
        self._bot_cols: Optional[list[str]] = None
        self._top_cols: Optional[list[str]] = None

    def _load_xgb(self):
        import json
        import xgboost as xgb
        device = os.environ.get("XGB_DEVICE", "auto").lower()
        dev = "cuda" if device in ("auto", "cuda") else "cpu"

        if self.bottom_model_path and self._bot is None:
            clf = xgb.XGBClassifier(device=dev)
            clf.load_model(self.bottom_model_path)
            self._bot = clf
            fc = Path(self.bottom_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._bot_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._bot_cols = None

        if self.top_model_path and self._top is None:
            clf = xgb.XGBClassifier(device=dev)
            clf.load_model(self.top_model_path)
            self._top = clf
            fc = Path(self.top_model_path).with_suffix("").as_posix() + "_feature_columns.json"
            if os.path.exists(fc):
                try:
                    self._top_cols = json.loads(Path(fc).read_text())
                except Exception:
                    self._top_cols = None

    def _view(self, df: pd.DataFrame, cols: Optional[list[str]]) -> Optional[np.ndarray]:
        if cols is None:
            return None
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out.reindex(columns=cols).values

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Load models
        self._load_xgb()

        # Build features; union of available model cols
        union_cols: Optional[list[str]] = None
        for cols in (self._bot_cols, self._top_cols):
            if cols:
                if union_cols is None:
                    union_cols = list(cols)
                else:
                    for c in cols:
                        if c not in union_cols:
                            union_cols.append(c)
        feats = make_features(df, feature_columns=union_cols, mode=self.feature_mode, extra_timeframes=self.extra_timeframes)
        feats = feats.reset_index(drop=True)

        # Predict probabilities
        p_bot = None
        p_top = None
        if self._bot is not None and self._bot_cols is not None:
            Xb = self._view(feats, self._bot_cols)
            try:
                p_bot = self._bot.predict_proba(Xb)[:, 1]
            except Exception:
                p_bot = None
        if self._top is not None and self._top_cols is not None:
            Xt = self._view(feats, self._top_cols)
            try:
                p_top = self._top.predict_proba(Xt)[:, 1]
            except Exception:
                p_top = None

        n = len(feats)
        enter_long = np.zeros(n, dtype=int)
        exit_long = np.zeros(n, dtype=int)
        enter_short = np.zeros(n, dtype=int)
        exit_short = np.zeros(n, dtype=int)

        buy_ok = np.zeros(n, dtype=bool)
        sell_ok = np.zeros(n, dtype=bool)
        if p_bot is not None:
            buy_ok = p_bot >= float(self.p_buy_thr)
        if p_top is not None:
            sell_ok = p_top >= float(self.p_sell_thr)
        if p_bot is not None and p_top is not None and float(self.p_margin) > 0.0:
            buy_ok &= (p_bot - p_top) >= float(self.p_margin)
            sell_ok &= (p_top - p_bot) >= float(self.p_margin)

        # Convert to entries/exits with min-hold and cooldown
        last_change = -10**9
        cooldown_until = -1
        pos = 0
        for t in range(n):
            desired: int = 0
            if buy_ok[t] and not sell_ok[t]:
                desired = +1
            elif sell_ok[t] and not buy_ok[t]:
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

        # Assign
        df["enter_long"] = enter_long
        df["exit_long"] = exit_long
        df["enter_short"] = enter_short
        df["exit_short"] = exit_short

        # Diagnostics
        if p_bot is not None:
            df["p_bottom"] = p_bot
        if p_top is not None:
            df["p_top"] = p_top
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'enter_long'] = (df.get('enter_long', 0) == 1).astype('int')
        df.loc[:, 'enter_short'] = (df.get('enter_short', 0) == 1).astype('int')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'exit_long'] = (df.get('exit_long', 0) == 1).astype('int')
        df.loc[:, 'exit_short'] = (df.get('exit_short', 0) == 1).astype('int')
        return df



from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import timedelta
import os
from pathlib import Path
import logging
import sys

# Ensure both userdir and project root are on sys.path so rl_lib resolves
_userdir_root = Path(__file__).resolve().parents[1]
_project_root = Path(__file__).resolve().parents[2]
for _p in (str(_userdir_root), str(_project_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rl_lib.signal import compute_rl_signals


class RLStrategy(IStrategy):
    timeframe = '1h'
    # Disable ROI-based exits so RL controls exits exclusively
    minimal_roi = {"0": 1000}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = False
    use_exit_signal = True
    can_short = True

    # Disable TA; RL decides
    process_only_new_candles = True
    startup_candle_count = 256

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_path = os.environ.get("RL_MODEL_PATH", str(Path(__file__).resolve().parents[3] / "models" / "rl_ppo.zip"))
        self.window = int(os.environ.get("RL_WINDOW", "128"))
        self._logger = logging.getLogger(__name__)
        # Align with env trade gating if provided
        self.min_hold_bars = int(os.environ.get("RL_MIN_HOLD_BARS", "0"))
        self.cooldown_bars = int(os.environ.get("RL_COOLDOWN_BARS", "0"))
        self._cooldown_until = None
        # Runtime gating state per pair
        self._last_pos_change_time = {}
        self._last_exit_time = {}
        # Parse timeframe to bar duration
        self._bar_seconds = 3600
        try:
            tf = str(getattr(self, 'timeframe', '1h')).lower()
            if tf.endswith('h'):
                self._bar_seconds = int(tf[:-1] or '1') * 3600
            elif tf.endswith('d'):
                self._bar_seconds = int(tf[:-1] or '1') * 86400
        except Exception:
            self._bar_seconds = 3600

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            enriched = compute_rl_signals(dataframe, self.model_path, window=self.window)
            dataframe["enter_long"] = enriched["enter_long"].values
            dataframe["exit_long"] = enriched["exit_long"].values
            dataframe["enter_short"] = enriched["enter_short"].values
            dataframe["exit_short"] = enriched["exit_short"].values
            # Propagate risk gate for dynamic position sizing
            if "risk_gate" in enriched.columns:
                dataframe["risk_gate"] = enriched["risk_gate"].values
            # Optional: dump signals for audit
            if os.environ.get("RL_DUMP_SIGNALS", "0") in ("1", "true", "True"):
                try:
                    outdir = Path(self.config.get('user_data_dir', '.')) / 'logs'
                    outdir.mkdir(parents=True, exist_ok=True)
                    fp = outdir / f"signals_{metadata.get('pair', 'PAIR').replace('/', '_')}_{self.timeframe}.csv"
                    enriched.to_csv(fp)
                    self._logger.info(f"RLStrategy: dumped signals -> {fp}")
                except Exception:
                    pass
            # Debug summary
            el = int(enriched["enter_long"].sum())
            es = int(enriched["enter_short"].sum())
            xl = int(enriched["exit_long"].sum())
            xs = int(enriched["exit_short"].sum())
            self._logger.info(
                f"RLStrategy: signals summary - enter_long={el}, enter_short={es}, exit_long={xl}, exit_short={xs}, model='{self.model_path}', window={self.window}"
            )
        except Exception as e:
            # On failure default to no-op but log the root cause
            self._logger.exception(f"RLStrategy: compute_rl_signals failed: {e}")
            dataframe["enter_long"] = 0
            dataframe["exit_long"] = 0
            dataframe["enter_short"] = 0
            dataframe["exit_short"] = 0
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = (dataframe['enter_long'] == 1).astype('int')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (dataframe['exit_long'] == 1).astype('int')
        dataframe.loc[:, 'enter_short'] = (dataframe['enter_short'] == 1).astype('int')
        dataframe.loc[:, 'exit_short'] = (dataframe['exit_short'] == 1).astype('int')
        return dataframe

    # New unified API in newer Freqtrade versions
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enforce cooldown gating
        if self.cooldown_bars > 0 and 'date' in dataframe.columns:
            if self._cooldown_until is not None:
                mask_cd = dataframe['date'] < self._cooldown_until
                dataframe.loc[mask_cd, ['enter_long', 'enter_short']] = 0
        dataframe.loc[:, 'enter_long'] = (dataframe.get('enter_long', 0) == 1).astype('int')
        dataframe.loc[:, 'enter_short'] = (dataframe.get('enter_short', 0) == 1).astype('int')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (dataframe.get('exit_long', 0) == 1).astype('int')
        dataframe.loc[:, 'exit_short'] = (dataframe.get('exit_short', 0) == 1).astype('int')
        return dataframe

    # Dynamic stake sizing: scale proposed stake by current risk_gate (0..1)
    def custom_stake_amount(self, pair: str, current_time, current_rate: float, proposed_stake: float, **kwargs) -> float:
        try:
            # Access analyzed dataframe to fetch risk_gate aligned at current_time
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            g = 1.0
            if df is not None and len(df) > 0:
                if current_time in df.index and 'risk_gate' in df.columns:
                    val = df.at[current_time, 'risk_gate']
                else:
                    val = df['risk_gate'].iloc[-1] if 'risk_gate' in df.columns else 1.0
                try:
                    import math
                    g = float(val)
                    if not math.isfinite(g):
                        g = 1.0
                except Exception:
                    g = 1.0
            g = float(max(0.1, min(1.0, g)))
            return float(max(0.0, proposed_stake * g))
        except Exception:
            return float(proposed_stake)

    # Enforce min-hold and cooldown at execution time; prevent same-candle flip
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time, **kwargs):
        # Cooldown check
        if self.cooldown_bars > 0 and self._cooldown_until is not None and current_time < self._cooldown_until:
            return False, "Cooldown active"
        # Min-hold check since last position change
        last = self._last_pos_change_time.get(pair)
        if self.min_hold_bars > 0 and last is not None:
            min_dt = last + timedelta(seconds=self.min_hold_bars * self._bar_seconds)
            if current_time < min_dt:
                return False, "Min-hold not reached"
        # Prevent immediate re-entry on the same candle after exit
        lex = self._last_exit_time.get(pair)
        if lex is not None and getattr(lex, 'ceil', None):
            # Pandas Timestamp: treat same-bar as forbidden
            if lex.floor(f"{self._bar_seconds}s") == getattr(current_time, 'floor', lambda x: current_time)(f"{self._bar_seconds}s"):
                return False, "No same-bar flip"
        # Record entry time as last position change moment
        try:
            self._last_pos_change_time[pair] = current_time
        except Exception:
            pass
        return True, "OK"

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float, rate: float, time_in_force: str, current_time, **kwargs):
        # Min-hold before exit
        last = self._last_pos_change_time.get(pair)
        if self.min_hold_bars > 0 and last is not None:
            min_dt = last + timedelta(seconds=self.min_hold_bars * self._bar_seconds)
            if current_time < min_dt:
                return False, "Min-hold not reached for exit"
        # Record exit time and set cooldown window
        try:
            self._last_exit_time[pair] = current_time
        except Exception:
            pass
        if self.cooldown_bars > 0:
            try:
                self._cooldown_until = current_time + timedelta(seconds=self.cooldown_bars * self._bar_seconds)
            except Exception:
                self._cooldown_until = None
        # Update last position change time to current_time
        self._last_pos_change_time[pair] = current_time
        return True, "OK"



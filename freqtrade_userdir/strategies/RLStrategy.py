from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
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
                    pair_tag = str(metadata.get('pair', 'PAIR')).replace('/', '_').replace(':', '_').replace(' ', '_')
                    fp = outdir / f"signals_{pair_tag}_{self.timeframe}.csv"
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



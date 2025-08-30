from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import os
from pathlib import Path
import logging

# Local import from project
try:
    from rl_lib.signal import compute_rl_signals
except Exception:
    # When running from freqtrade, PYTHONPATH may not include project root
    import sys
    # Ensure user_data root (/freqtrade/user_data) is on path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
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
    startup_candle_count = 128

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_path = os.environ.get("RL_MODEL_PATH", str(Path(__file__).resolve().parents[3] / "models" / "rl_ppo.zip"))
        self.window = int(os.environ.get("RL_WINDOW", "128"))
        self._logger = logging.getLogger(__name__)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            enriched = compute_rl_signals(dataframe, self.model_path, window=self.window)
            dataframe["enter_long"] = enriched["enter_long"].values
            dataframe["exit_long"] = enriched["exit_long"].values
            dataframe["enter_short"] = enriched["enter_short"].values
            dataframe["exit_short"] = enriched["exit_short"].values
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
        dataframe.loc[:, 'enter_long'] = (dataframe.get('enter_long', 0) == 1).astype('int')
        dataframe.loc[:, 'enter_short'] = (dataframe.get('enter_short', 0) == 1).astype('int')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (dataframe.get('exit_long', 0) == 1).astype('int')
        dataframe.loc[:, 'exit_short'] = (dataframe.get('exit_short', 0) == 1).astype('int')
        return dataframe



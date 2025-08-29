from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import os
from pathlib import Path

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
    minimal_roi = {"0": 0}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = False
    use_exit_signal = True

    # Disable TA; RL decides
    process_only_new_candles = True
    startup_candle_count = 128

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_path = os.environ.get("RL_MODEL_PATH", str(Path(__file__).resolve().parents[3] / "models" / "rl_ppo.zip"))
        self.window = int(os.environ.get("RL_WINDOW", "128"))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            enriched = compute_rl_signals(dataframe, self.model_path, window=self.window)
            dataframe["enter_long"] = enriched["enter_long"].values
            dataframe["exit_long"] = enriched["exit_long"].values
        except Exception:
            # On failure default to no-op
            dataframe["enter_long"] = 0
            dataframe["exit_long"] = 0
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = (dataframe['enter_long'] == 1).astype('int')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (dataframe['exit_long'] == 1).astype('int')
        return dataframe



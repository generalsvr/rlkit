from functools import reduce
from typing import Dict

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class MyRLStrategy(IStrategy):
    timeframe = '1h'
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 300

    minimal_roi: Dict = {"0": 10}
    stoploss = -0.99
    trailing_stop = False

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # required raw ohlcv for RL env
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # placeholder target - RL fills &-action during prediction
        dataframe["&-action"] = 0
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [df.get("do_predict", 1) == 1, df["&-action"] == 1]
        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")

        enter_short_conditions = [df.get("do_predict", 1) == 1, df["&-action"] == 3]
        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df.get("do_predict", 1) == 1, df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df.get("do_predict", 1) == 1, df["&-action"] == 4]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1
        return df



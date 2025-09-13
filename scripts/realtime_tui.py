#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# Optional terminal plotting
try:
    import plotext as plt  # pip install plotext
    _PLOTEXT_OK = True
except Exception:
    _PLOTEXT_OK = False

# CCXT for live data
try:
    import ccxt
    _CCXT_OK = True
except Exception:
    _CCXT_OK = False

# Local imports
_CUR = Path(__file__).resolve().parents[1]
if str(_CUR) not in sys.path:
    sys.path.insert(0, str(_CUR))

from rl_lib.features import make_features


def _load_xgb(path: str, device: str = "auto"):
    import xgboost as xgb
    clf = xgb.XGBClassifier(device=("cuda" if device in ("auto","cuda") else "cpu"))
    clf.load_model(path)
    cols_path = str(Path(path).with_suffix("").as_posix()) + "_feature_columns.json"
    cols = None
    if os.path.exists(cols_path):
        try:
            with open(cols_path, "r") as f:
                cols = json.load(f)
        except Exception:
            cols = None
    return clf, cols


def _predict_with_cols(model, feats: pd.DataFrame, cols: Optional[List[str]]):
    if cols is None:
        return None
    X = feats.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    Xv = X.reindex(columns=cols).values
    try:
        return model.predict_proba(Xv)
    except Exception:
        return None


def timeframe_to_seconds(tf: str) -> int:
    s = tf.strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("d"):
        return int(s[:-1]) * 86400
    if s.endswith("w"):
        return int(s[:-1]) * 7 * 86400
    # default 1h
    return 3600


def fetch_ohlcv(exchange: str, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not _CCXT_OK:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = getattr(ccxt, exchange)()
    ex.load_markets()
    data = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=int(limit))
    # ccxt ohlcv: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]).dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def render_chart(df: pd.DataFrame, p_bottom: np.ndarray, p_top: np.ndarray, reg_dir: np.ndarray, p_logret: Optional[np.ndarray], p_thr: float, strong_thr: float, last_n: int = 240, style: str = "line", regdir_thr: float = 0.1):
    if not _PLOTEXT_OK:
        print("plotext not installed. pip install plotext")
        return
    sub = df.tail(last_n)
    close = sub["close"].astype(float).to_numpy()
    # Align predictions to sub length
    T = len(df)
    idx0 = max(0, T - len(sub))
    pb = (p_bottom[idx0:T] if p_bottom is not None else np.zeros(len(sub)))
    pt = (p_top[idx0:T] if p_top is not None else np.zeros(len(sub)))
    # Markers where prob >= thr
    buy_idx = np.where(pb >= float(p_thr))[0]
    sell_idx = np.where(pt >= float(p_thr))[0]

    plt.clear_figure()
    plt.theme('pro')
    # Title with logret summary if available
    title = "Realtime close + signals"
    if p_logret is not None and len(p_logret) == len(df):
        last = p_logret[-1]
        cls_idx = int(np.argmax(last))
        cls_map = {0:-2,1:-1,2:0,3:1,4:2}
        title += f" | logret={cls_map.get(cls_idx, 0)} ({float(np.max(last)):.2f})"
    plt.title(title)
    style_l = str(style).lower()
    if style_l == "candles" and hasattr(plt, "candlestick"):
        sub_o = sub["open"].astype(float).to_list()
        sub_h = sub["high"].astype(float).to_list()
        sub_l = sub["low"].astype(float).to_list()
        sub_c = sub["close"].astype(float).to_list()
        try:
            plt.candlestick(sub_o, sub_h, sub_l, sub_c)
        except Exception:
            plt.plot(close, label="close", color="cyan")
    else:
        plt.plot(close, label="close", color="cyan")
    # overlay markers
    if buy_idx.size > 0:
        yb = close[buy_idx]
        plt.scatter(buy_idx, yb, marker='●', color='green', label='p_bottom>=thr')
    if sell_idx.size > 0:
        ys = close[sell_idx]
        plt.scatter(sell_idx, ys, marker='●', color='red', label='p_top>=thr')
    # Overlay strong logret direction markers if available
    try:
        if p_logret is not None and len(p_logret) == len(df):
            pr_sub = p_logret[-len(sub):]
            strong = (np.max(pr_sub, axis=1) >= float(strong_thr))
            cls = np.argmax(pr_sub, axis=1)
            up_mask = strong & (cls >= 3)
            dn_mask = strong & (cls <= 1)
            up_idx = np.where(up_mask)[0]
            dn_idx = np.where(dn_mask)[0]
            if up_idx.size > 0:
                plt.scatter(up_idx, close[up_idx], marker='˄', color='green', label='logret↑')
            if dn_idx.size > 0:
                plt.scatter(dn_idx, close[dn_idx], marker='˅', color='red', label='logret↓')
    except Exception:
        pass
    plt.canvas_color('black'); plt.axes_color('black')
    plt.ticks_color('white')
    try:
        plt.legend()
    except Exception:
        pass
    plt.show()

    # Regime background emulation (text band): green for up, red for down, gray for neutral
    try:
        band = []
        rd_sub = reg_dir[-len(sub):]
        for val in rd_sub:
            if val >= float(regdir_thr):
                band.append("\033[92m▇\033[0m")
            elif val <= -float(regdir_thr):
                band.append("\033[91m▇\033[0m")
            else:
                band.append("\033[90m▇\033[0m")
        print("regime:", "".join(band))
    except Exception:
        pass

    # Logret info line
    try:
        if p_logret is not None and len(p_logret) == len(df):
            last = p_logret[-1]
            cls = ["-2","-1","0","1","2"]
            s = " | ".join([f"p{c}={last[i]:.2f}" for i, c in enumerate(cls)])
            print(f"{s}  reg_dir={reg_dir[-1]:.3f}")
        else:
            print(f"reg_dir={reg_dir[-1]:.3f}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Realtime terminal UI for XGB stack predictions.")
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--pair", default="BTC/USDT:USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--lookback", type=int, default=512)
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "xgb_stack"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--p-thr", type=float, default=0.6)
    ap.add_argument("--refresh", type=float, default=15.0, help="Refresh seconds between fetches")
    ap.add_argument("--logret-strong-thr", type=float, default=0.55, help="Strong class prob for logret markers")
    ap.add_argument("--style", choices=["line","candles"], default="line", help="Chart style")
    ap.add_argument("--regdir-thr", type=float, default=0.1, help="Regime threshold for background band coloring")
    args = ap.parse_args()

    bot_path = str(Path(args.models_dir) / "best_topbot_bottom.json")
    top_path = str(Path(args.models_dir) / "best_topbot_top.json")
    logret_path = str(Path(args.models_dir) / "best_logret.json")

    bot_clf = top_clf = logret_clf = None
    bot_cols = top_cols = logret_cols = None
    if os.path.exists(bot_path):
        bot_clf, bot_cols = _load_xgb(bot_path, device=args.device)
    if os.path.exists(top_path):
        top_clf, top_cols = _load_xgb(top_path, device=args.device)
    if os.path.exists(logret_path):
        logret_clf, logret_cols = _load_xgb(logret_path, device=args.device)

    tf_secs = timeframe_to_seconds(args.timeframe)

    while True:
        try:
            raw = fetch_ohlcv(args.exchange, args.pair, args.timeframe, limit=max(600, args.lookback + 10))
            feats = make_features(raw, mode='full', basic_lookback=64, extra_timeframes=None).reset_index(drop=True)
            # Predictions
            p_bottom = np.zeros(len(feats)); p_top = np.zeros(len(feats))
            p_logret = None
            if bot_clf is not None and bot_cols is not None:
                pr = _predict_with_cols(bot_clf, feats, bot_cols)
                if pr is not None and pr.ndim == 2 and pr.shape[1] >= 2:
                    p_bottom = pr[:, 1]
            if top_clf is not None and top_cols is not None:
                pr = _predict_with_cols(top_clf, feats, top_cols)
                if pr is not None and pr.ndim == 2 and pr.shape[1] >= 2:
                    p_top = pr[:, 1]
            reg_dir = np.zeros(len(feats))
            if logret_clf is not None and logret_cols is not None:
                pr = _predict_with_cols(logret_clf, feats, logret_cols)
                if pr is not None and pr.ndim == 2 and pr.shape[1] == 5:
                    p_logret = pr
                    class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
                    reg_dir = pr @ class_vals
            render_chart(raw, p_bottom, p_top, reg_dir, p_logret,
                         p_thr=float(args.p_thr), strong_thr=float(args.logret_strong_thr),
                         last_n=min(240, len(raw)), style=str(args.style), regdir_thr=float(args.regdir_thr))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[warn] realtime loop error: {e}")
        # sleep until next refresh (fraction of timeframe)
        time.sleep(max(2.0, float(args.refresh)))


if __name__ == "__main__":
    main()

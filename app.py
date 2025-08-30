from __future__ import annotations

import os
import subprocess
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from rl_lib.signal import compute_rl_signals
from rl_lib.train_sb3 import _find_data_file, _load_ohlcv, _compute_risk_metrics


# -----------------------------
# Utilities
# -----------------------------

def _format_timerange(start: date, end: date) -> str:
    s = start.strftime("%Y%m%d") if start else ""
    e = end.strftime("%Y%m%d") if end else ""
    return f"{s}-{e}" if (s or e) else ""


def _run_freqtrade_download(pair: str, timeframe: str, userdir: str, timerange: str, exchange: str) -> Tuple[bool, str]:
    cmd = [
        "freqtrade", "download-data",
        "--pairs", pair,
        "--timeframes", timeframe,
        "--userdir", userdir,
        "--timerange", timerange,
        "--exchange", exchange,
        "--data-format-ohlcv", "parquet",
    ]
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, cp.stdout or "Download complete."
    except FileNotFoundError:
        # Fallback to python -m freqtrade
        cmd = [
            "python", "-m", "freqtrade", "download-data",
            "--pairs", pair,
            "--timeframes", timeframe,
            "--userdir", userdir,
            "--timerange", timerange,
            "--exchange", exchange,
            "--data-format-ohlcv", "parquet",
        ]
        try:
            cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, cp.stdout or "Download complete."
        except Exception as e:
            return False, f"Freqtrade download failed: {e}"
    except Exception as e:
        return False, f"Freqtrade download failed: {e}"


def _pay_costs(equity: float, from_pos: int, to_pos: int, fee_bps: float, slippage_bps: float) -> Tuple[float, float]:
    if from_pos == to_pos:
        return equity, 0.0
    bps = fee_bps + slippage_bps
    if from_pos * to_pos == -1:
        bps *= 2.0
    cost = bps * 1e-4
    equity *= (1.0 - cost)
    return equity, float(cost)


def simulate_equity_from_actions(
    close: np.ndarray,
    actions: np.ndarray,
    window: int = 128,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
    pnl_on_close: bool = False,
    idle_penalty_bps: float = 0.0,
    turnover_penalty_bps: float = 0.0,
) -> Dict[str, np.ndarray]:
    n = int(close.size)
    if n == 0:
        return {
            "equity": np.asarray([], dtype=float),
            "position": np.asarray([], dtype=int),
        }
    window = int(max(1, window))
    eq = 1.0
    position = 0
    entry_price = None
    equity_curve = np.full(n, np.nan, dtype=float)
    position_series = np.zeros(n, dtype=int)

    # Initialize historical range
    for i in range(min(window, n)):
        equity_curve[i] = eq
        position_series[i] = position

    for t in range(window, n):
        prev_price = float(close[t - 1])
        act = int(actions[t]) if t < actions.size else 0
        prev_position = position

        # Apply action at time t (effective for period [t, t+1])
        if act == 1:  # long
            if position <= 0:
                eq, _ = _pay_costs(eq, position, +1, fee_bps, slippage_bps)
                position = +1
                entry_price = prev_price
        elif act == 2:  # short
            if position >= 0:
                eq, _ = _pay_costs(eq, position, -1, fee_bps, slippage_bps)
                position = -1
                entry_price = prev_price
        elif act == 3:  # close
            if position != 0:
                eq, _ = _pay_costs(eq, position, 0, fee_bps, slippage_bps)
                if pnl_on_close and entry_price is not None:
                    r_close = (prev_price - entry_price) / (entry_price + 1e-12)
                    r_close = r_close if prev_position == +1 else -r_close
                    eq *= (1.0 + r_close)
                position = 0
                entry_price = None

        # Price advance to t (env would update on advance)
        new_price = float(close[t])
        if not pnl_on_close:
            r = (new_price - prev_price) / (prev_price + 1e-12)
            if position == +1:
                eq *= (1.0 + r)
            elif position == -1:
                eq *= (1.0 - r)
            else:
                if idle_penalty_bps > 0.0:
                    idle_cost = idle_penalty_bps * 1e-4
                    eq *= (1.0 - idle_cost)

        if turnover_penalty_bps > 0.0 and prev_position != position:
            tcost = turnover_penalty_bps * 1e-4
            eq *= (1.0 - tcost)

        equity_curve[t] = eq
        position_series[t] = position

    return {
        "equity": equity_curve,
        "position": position_series,
    }


@st.cache_data(show_spinner=False)
def load_data(userdir: str, pair: str, timeframe: str, prefer_exchange: str | None = None) -> pd.DataFrame:
    path = _find_data_file(userdir, pair, timeframe, prefer_exchange=prefer_exchange)
    if not path:
        raise FileNotFoundError("No dataset found. Use Download to fetch data.")
    df = _load_ohlcv(path)
    return df


@st.cache_data(show_spinner=False)
def slice_data(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and (start or end):
        idx = df.index
        try:
            idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
        except Exception:
            idx_cmp = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
        mask = pd.Series(True, index=idx)
        if start:
            mask &= idx_cmp >= pd.to_datetime(start)
        if end:
            mask &= idx_cmp <= pd.to_datetime(end)
        return df.loc[mask]
    return df


@st.cache_data(show_spinner=False)
def run_rl_signals(df: pd.DataFrame, model_path: str, window: int) -> pd.DataFrame:
    os.environ.setdefault("RL_DEVICE", "cpu")
    out = compute_rl_signals(df, model_path=model_path, window=window)
    return out


def plot_results(df: pd.DataFrame, signals: pd.DataFrame, equity: np.ndarray, title: str = "Backtest"):
    idx = signals.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05)

    if all(c in df.columns for c in ("open", "high", "low", "close")):
        fig.add_trace(
            go.Candlestick(
                x=idx,
                open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name="OHLC"
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=idx, y=df["close"], mode="lines", name="Close"),
            row=1, col=1,
        )

    # Signals
    for col, name, symbol, color in [
        ("enter_long", "LONG", "triangle-up", "#2ecc71"),
        ("enter_short", "SHORT", "triangle-down", "#e74c3c"),
        ("exit_long", "EXIT L", "x", "#f1c40f"),
        ("exit_short", "EXIT S", "x", "#f39c12"),
    ]:
        if col in signals.columns:
            mask = signals[col] == 1
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=idx[mask],
                        y=df.loc[mask, "close"],
                        mode="markers",
                        name=name,
                        marker=dict(symbol=symbol, size=10, color=color, line=dict(width=0)),
                    ),
                    row=1, col=1,
                )

    # Equity curve
    fig.add_trace(
        go.Scatter(x=idx, y=equity, mode="lines", name="Equity", line=dict(color="#3498db")),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=800,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="RL Trading Terminal", layout="wide")
st.title("RL Trading Terminal")

with st.sidebar:
    st.header("Data")
    pair = st.text_input("Pair", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", options=["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    userdir = st.text_input("Userdir", value=str(os.path.join(os.path.dirname(__file__), "freqtrade_userdir")))
    exchange = st.selectbox("Exchange", options=["binance", "bybit", "okx", "kraken"], index=0)

    st.markdown("---")
    st.header("Range")
    today = date.today()
    start_date = st.date_input("Start", value=date(today.year - 2, 1, 1))
    end_date = st.date_input("End", value=today)
    timerange_str = _format_timerange(start_date, end_date)

    if st.button("Download via Freqtrade", use_container_width=True):
        ok, msg = _run_freqtrade_download(pair, timeframe, userdir, timerange_str, exchange)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")
    st.header("Model")
    default_model = os.path.join(os.path.dirname(__file__), "models", "rl_ppo.zip")
    model_path = st.text_input("Model path", value=default_model)
    window = st.number_input("Window", min_value=8, max_value=2048, value=128, step=8)

    st.markdown("---")
    st.header("Costs")
    fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=6.0, step=0.1)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    idle_penalty_bps = st.number_input("Idle penalty (bps)", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
    turnover_penalty_bps = st.number_input("Turnover penalty (bps)", min_value=0.0, max_value=10.0, value=0.0, step=0.01)


st.subheader("Backtest")

try:
    raw_df = load_data(userdir, pair, timeframe, prefer_exchange=exchange)
except Exception as e:
    st.info("No local data found. Use Download to fetch dataset.")
    raw_df = pd.DataFrame()

if not raw_df.empty:
    st.caption(f"Loaded rows={len(raw_df)} | start={raw_df.index[0]} | end={raw_df.index[-1]}")

col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    run_clicked = st.button("Run Visual Test", type="primary")
with col_run2:
    export_clicked = st.button("Export Signals CSV")

signals_df = None

if run_clicked and not raw_df.empty:
    df = slice_data(raw_df, start_date, end_date)
    if df.empty:
        st.warning("Selected range returned no rows.")
    else:
        try:
            signals_df = run_rl_signals(df, model_path=model_path, window=int(window))
            sim = simulate_equity_from_actions(
                close=signals_df["close"].to_numpy(dtype=float),
                actions=signals_df["rl_action"].to_numpy(dtype=int),
                window=int(window),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                pnl_on_close=False,
                idle_penalty_bps=float(idle_penalty_bps),
                turnover_penalty_bps=float(turnover_penalty_bps),
            )
            equity = sim["equity"]
            metrics = _compute_risk_metrics(np.asarray([e for e in equity if np.isfinite(e)], dtype=float))

            left, right = st.columns([1, 1])
            with left:
                st.metric("Final Equity", f"{equity[~np.isnan(equity)][-1]:.4f}")
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', float('nan')):.2%}")
            with right:
                st.metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}")
                st.metric("Sortino", f"{metrics.get('sortino', float('nan')):.2f}")

            plot_results(df, signals_df, equity, title=f"{pair} {timeframe}")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)

if export_clicked and signals_df is not None:
    csv = signals_df.to_csv(index=True).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="rl_signals.csv", mime="text/csv")



#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Third-party
try:
    import ccxt
    _CCXT_OK = True
except Exception:
    _CCXT_OK = False

try:
    from dash import Dash, dcc, html, Input, Output
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    _DASH_OK = True
except Exception:
    _DASH_OK = False

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


def fetch_ohlcv(exchange: str, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not _CCXT_OK:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = getattr(ccxt, exchange)()
    ex.load_markets()
    data = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=int(limit))
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]).dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def timeframe_to_seconds(tf: str) -> int:
    s = str(tf).lower().strip()
    if s.endswith('m'):
        return int(s[:-1]) * 60
    if s.endswith('h'):
        return int(s[:-1]) * 3600
    if s.endswith('d'):
        return int(s[:-1]) * 86400
    if s.endswith('w'):
        return int(s[:-1]) * 7 * 86400
    return 3600


def drop_incomplete_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    tf_sec = timeframe_to_seconds(timeframe)
    now = pd.Timestamp.now(tz='UTC')
    last_ts = df.index[-1]
    if (now - last_ts).total_seconds() < (tf_sec - 1):
        return df.iloc[:-1]
    return df


def make_figure(df: pd.DataFrame, p_bottom: np.ndarray, p_top: np.ndarray, pr_logret: Optional[np.ndarray], p_thr: float, strong_thr: float, reg_thr: float) -> go.Figure:
    T = len(df)
    idx = df.index
    # Single pane chart
    fig = make_subplots(rows=1, cols=1)

    # Regime shading ON the main chart using Heatmap (like eval imshow)
    if pr_logret is not None and pr_logret.shape == (T, 5):
        class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
        reg_dir = pr_logret @ class_vals  # [-2,2]
        # Normalize to [0,1] for colorscale mapping
        reg_norm = (reg_dir + 2.0) / 4.0
        y_min = float(df['low'].min())
        y_max = float(df['high'].max())
        z = np.vstack([reg_norm, reg_norm])  # shape (2, T)
        fig.add_trace(
            go.Heatmap(
                x=idx, y=[y_min, y_max], z=z,
                colorscale=[[0.0, '#d62728'], [0.5, '#7f7f7f'], [1.0, '#2ca02c']],
                zmin=0.0, zmax=1.0, opacity=0.18,
                showscale=False, hoverinfo='skip', name='reg_bg'
            ),
            row=1, col=1
        )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=idx, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC',
            increasing_line_color='#9aa0a6', decreasing_line_color='#9aa0a6',
            increasing_fillcolor='#3c4043', decreasing_fillcolor='#202124',
            opacity=0.95
        ),
        row=1, col=1
    )

    # Signals (on top)
    pb = p_bottom if p_bottom is not None and len(p_bottom) == T else np.zeros(T)
    pt = p_top if p_top is not None and len(p_top) == T else np.zeros(T)
    buy_mask = pb >= float(p_thr)
    sell_mask = pt >= float(p_thr)
    fig.add_trace(go.Scatter(
        x=idx[buy_mask], y=df['close'][buy_mask], mode='markers', name='Bottom≥thr',
        marker=dict(color='lime', size=12, symbol='triangle-up', line=dict(color='white', width=1.2)),
        hoverinfo='x+y'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx[sell_mask], y=df['close'][sell_mask], mode='markers', name='Top≥thr',
        marker=dict(color='#ff1744', size=12, symbol='triangle-down', line=dict(color='white', width=1.2)),
        hoverinfo='x+y'), row=1, col=1)

    # Logret quick badges
    if pr_logret is not None and pr_logret.shape == (T, 5):
        class_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
        reg_dir = pr_logret @ class_vals
        last = pr_logret[-1]
        txt = " ".join([f"p{c}={last[i]:.2f}" for i,c in enumerate(["-2","-1","0","1","2"])])
        fig.add_annotation(text=txt, xref='paper', yref='paper', x=0.01, y=0.99, showarrow=False, font=dict(color='white', size=12))
        last_cls = int(np.argmax(last)); last_dir = class_vals[last_cls]
        rec = 'UP' if (last_dir > 0 and reg_dir[-1] >= reg_thr) else ('DOWN' if (last_dir < 0 and reg_dir[-1] <= -reg_thr) else 'NEUTRAL')
        fig.add_annotation(text=f"REC: {rec}", xref='paper', yref='paper', x=0.92, y=0.99, showarrow=False, font=dict(color='white', size=12))

    fig.update_xaxes(rangeslider_visible=True, showspikes=True, spikemode='across', spikesnap='cursor')
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=6, r=6, t=28, b=4),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        dragmode='pan',
        hovermode='x unified',
        uirevision=True,
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description='Realtime Dash GUI for XGB stack')
    ap.add_argument('--exchange', default='bybit')
    ap.add_argument('--pair', default='BTC/USDT:USDT')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--lookback', type=int, default=400)
    ap.add_argument('--models-dir', default=str(Path(__file__).resolve().parents[1] / 'models' / 'xgb_stack'))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--p-thr', type=float, default=0.6)
    ap.add_argument('--logret-strong-thr', type=float, default=0.55)
    ap.add_argument('--extra-timeframes', default='4H,1D,1W')
    ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--refresh', type=float, default=15.0)
    args = ap.parse_args()

    if not _DASH_OK:
        print("Install deps: pip install dash plotly ccxt xgboost")
        sys.exit(1)

    # Load models
    bot_path = str(Path(args.models_dir) / 'best_topbot_bottom.json')
    top_path = str(Path(args.models_dir) / 'best_topbot_top.json')
    logret_path = str(Path(args.models_dir) / 'best_logret.json')
    bot_clf = bot_cols = top_clf = top_cols = logret_clf = logret_cols = None
    if os.path.exists(bot_path):
        bot_clf, bot_cols = _load_xgb(bot_path, device=args.device)
    if os.path.exists(top_path):
        top_clf, top_cols = _load_xgb(top_path, device=args.device)
    if os.path.exists(logret_path):
        logret_clf, logret_cols = _load_xgb(logret_path, device=args.device)

    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label('p_thr'),
                dcc.Slider(id='p-thr', min=0.0, max=1.0, step=0.01, value=float(args.p_thr), tooltip={'always_visible': True})
            ], style={'flex': '1', 'marginRight': '16px'}),
            html.Div([
                html.Label('reg_dir_thr'),
                dcc.Slider(id='reg-thr', min=0.0, max=1.5, step=0.01, value=float(args.logret-strong-thr) if False else float(args.logret_strong_thr), tooltip={'always_visible': True})
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'padding': '6px 12px'}),
        dcc.Graph(
            id='price-graph', figure=go.Figure(),
            style={'height': '96vh', 'width': '100vw'},
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset',
                'responsive': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
            }
        ),
        dcc.Interval(id='timer', interval=int(max(2.0, args.refresh) * 1000), n_intervals=0),
        html.Div(id='status', style={'marginTop': '6px'})
    ])

    @app.callback(Output('price-graph', 'figure'), Output('status', 'children'),
                  Input('timer', 'n_intervals'), Input('p-thr', 'value'), Input('reg-thr', 'value'))
    def update(_, p_thr_val, reg_thr_val):
        try:
            raw = fetch_ohlcv(args.exchange, args.pair, args.timeframe, limit=max(600, args.lookback + 10))
            raw = drop_incomplete_last_bar(raw, args.timeframe)
            etf = [s.strip() for s in str(args.extra_timeframes).split(',') if s.strip()]
            feats = make_features(raw, mode='full', basic_lookback=64, extra_timeframes=(etf or None)).reset_index(drop=True)
            T = len(feats)
            # Predictions
            p_bottom = np.zeros(T); p_top = np.zeros(T); pr = None
            if bot_clf is not None and bot_cols is not None:
                pb = _predict_with_cols(bot_clf, feats, bot_cols)
                if pb is not None and pb.ndim == 2 and pb.shape[1] >= 2:
                    p_bottom = pb[:, 1]
            if top_clf is not None and top_cols is not None:
                pt = _predict_with_cols(top_clf, feats, top_cols)
                if pt is not None and pt.ndim == 2 and pt.shape[1] >= 2:
                    p_top = pt[:, 1]
            if logret_clf is not None and logret_cols is not None:
                pr = _predict_with_cols(logret_clf, feats, logret_cols)
            fig = make_figure(raw, p_bottom, p_top, pr, p_thr=float(p_thr_val or args.p_thr), strong_thr=float(args.logret_strong_thr), reg_thr=float(reg_thr_val or args.logret_strong_thr))
            return fig, f"Last update: {pd.Timestamp.utcnow()}"
        except Exception as e:
            return go.Figure(), f"Error: {e}"

    try:
        # Dash >= 2.17
        app.run(debug=False, host=args.host, port=int(args.port))
    except Exception:
        # Fallback for older Dash
        app.run_server(debug=False, host=args.host, port=int(args.port))


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from rl_lib.timesfm_forecast import (
    TimesFMForecastParams,
    TimesFMNotAvailableError,
    run_timesfm_forecast,
)


app = typer.Typer(add_completion=False)


def _csv_to_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts or None


def _csv_to_float_list(text: Optional[str]) -> Optional[List[float]]:
    parts = _csv_to_list(text)
    if not parts:
        return None
    out: List[float] = []
    for item in parts:
        try:
            val = float(item)
        except ValueError as exc:
            raise typer.BadParameter(f"Could not parse float value '{item}' in quantile levels.") from exc
        if not (0.0 <= val <= 1.0):
            raise typer.BadParameter(f"Quantile level {val} must be within [0, 1].")
        out.append(val)
    return out


def _load_json_flags(payload: Optional[str]) -> Optional[dict]:
    if payload is None or str(payload).strip() == "":
        return None
    try:
        return json.loads(str(payload))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Failed to parse JSON compile flags: {exc}") from exc


@app.command()
def forecast(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: Optional[str] = typer.Option(None),
    prefer_exchange: Optional[str] = typer.Option(None, "--prefer-exchange", "--prefer_exchange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: Optional[str] = typer.Option(None, help="Comma separated HTFs, e.g. '4H,1D'"),
    target_columns: Optional[str] = typer.Option(None, help="Comma separated target columns, default close"),
    context_length: int = typer.Option(2048),
    horizon: int = typer.Option(64),
    stride: int = typer.Option(1),
    max_windows: int = typer.Option(256),
    normalize_inputs: bool = typer.Option(True),
    use_quantile_head: bool = typer.Option(True, help="Enable continuous quantile head"),
    force_flip_invariance: bool = typer.Option(True),
    infer_is_positive: bool = typer.Option(True),
    fix_quantile_crossing: bool = typer.Option(True),
    quantile_levels: Optional[str] = typer.Option(None, help="Comma separated quantile levels e.g. 0.1,0.5,0.9"),
    compile_flags: Optional[str] = typer.Option(None, help="Additional ForecastConfig kwargs as JSON string"),
    outdir: Optional[str] = typer.Option(None, help="Directory to persist predictions/summary"),
    save_csv: bool = typer.Option(True),
    save_json: bool = typer.Option(True),
):
    """Run TimesFM 2.5 forecasts over market data slices."""

    params = TimesFMForecastParams(
        userdir=str(userdir),
        pair=str(pair),
        timeframe=str(timeframe),
        timerange=timerange,
        prefer_exchange=prefer_exchange,
        feature_mode=str(feature_mode),
        basic_lookback=int(basic_lookback),
        extra_timeframes=_csv_to_list(extra_timeframes),
        target_columns=_csv_to_list(target_columns),
        context_length=int(context_length),
        horizon=int(horizon),
        stride=int(stride),
        max_windows=int(max_windows),
        normalize_inputs=bool(normalize_inputs),
        use_continuous_quantile_head=bool(use_quantile_head),
        force_flip_invariance=bool(force_flip_invariance),
        infer_is_positive=bool(infer_is_positive),
        fix_quantile_crossing=bool(fix_quantile_crossing),
        quantile_levels=_csv_to_float_list(quantile_levels),
        compile_flags=_load_json_flags(compile_flags),
        outdir=str(outdir) if outdir else None,
        save_csv=bool(save_csv),
        save_json=bool(save_json),
    )

    try:
        report = run_timesfm_forecast(params)
    except TimesFMNotAvailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()

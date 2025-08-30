"use client";

import React from "react";
import type { IChartApi, UTCTimestamp } from "lightweight-charts";

type Candle = [number, number, number, number, number, number];
type Mark = { time: number; label: string; color: string };

function toUtc(tsMs: number): UTCTimestamp {
  return Math.floor(tsMs / 1000) as UTCTimestamp;
}

export default function Home() {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const seriesRef = React.useRef<any>(null);
  const [loading, setLoading] = React.useState(false);
  const [logs, setLogs] = React.useState<any[]>([]);
  const [pair, setPair] = React.useState("BTC/USDT:USDT");
  const [timeframe, setTimeframe] = React.useState("1h");
  const [timerange, setTimerange] = React.useState<string>("");
  const [modelPath, setModelPath] = React.useState<string>("./models/newrl_ppo.zip");

  async function loadChartsModule() {
    const mod: any = await import("lightweight-charts");
    const create = mod.createChart || (mod.default && mod.default.createChart);
    if (!create) throw new Error("createChart not found in lightweight-charts module");
    return create as (container: HTMLElement, options?: any) => IChartApi;
  }

  React.useEffect(() => {
    if (!containerRef.current || chartRef.current) return;
    let chart: IChartApi | null = null;
    let ro: ResizeObserver | null = null;
    loadChartsModule()
      .then((create) => {
        chart = create(containerRef.current as HTMLElement, {
          height: 560,
          layout: { textColor: "#d1d5db", background: { type: "solid", color: "#0b1220" } },
          grid: { horzLines: { color: "#1f2937" }, vertLines: { color: "#1f2937" } },
          timeScale: { rightOffset: 4, barSpacing: 8, lockVisibleTimeRangeOnResize: true },
          crosshair: { mode: 1 },
        });
        const series = (chart as any).addCandlestickSeries();
        chartRef.current = chart;
        seriesRef.current = series;
        ro = new ResizeObserver(() => chart!.applyOptions({ width: containerRef.current?.clientWidth || 800 }));
        ro.observe(containerRef.current as Element);
      })
      .catch((e) => {
        console.error(e);
      });
    return () => {
      if (ro) ro.disconnect();
      if (chart) (chart as any).remove?.();
      chartRef.current = null;
    };
  }, []);

  const fetchCandles = async () => {
    const r = await fetch("http://127.0.0.1:8501/candles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pair, timeframe, timerange: timerange || null }),
    });
    if (!r.ok) throw new Error("candles fetch failed");
    const j = await r.json();
    const candles: Candle[] = j.candles || [];
    const series = seriesRef.current;
    if (series) {
      series.setData(
        candles.map((c: Candle) => ({ time: toUtc(c[0]), open: c[1], high: c[2], low: c[3], close: c[4] }))
      );
      chartRef.current?.timeScale().fitContent();
    }
  };

  const runAgent = async () => {
    setLoading(true);
    setLogs([]);
    try {
      const r = await fetch("http://127.0.0.1:8501/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pair,
          timeframe,
          model_path: modelPath,
          timerange: timerange || null,
          window: 128,
          reward_type: "vol_scaled",
        }),
      });
      if (!r.ok) throw new Error("run failed");
      const j = await r.json();
      const candles: Candle[] = j.candles || [];
      const marks: Mark[] = j.marks || [];
      const series = seriesRef.current;
      if (series && candles.length) {
        series.setData(
          candles.map((c: Candle) => ({ time: toUtc(c[0]), open: c[1], high: c[2], low: c[3], close: c[4] }))
        );
        if (marks.length) {
          series.setMarkers(
            marks.map((m: Mark) => ({ time: toUtc(m.time), position: "aboveBar", color: m.color, shape: "arrowDown", text: m.label }))
          );
        }
        chartRef.current?.timeScale().fitContent();
      }
      setLogs(j.logs || []);
    } catch (e: any) {
      setLogs([{ level: "error", message: e?.message || String(e) }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#0b1220] text-gray-200 p-6">
      <h1 className="text-xl font-semibold mb-4">Backtest Agent Terminal</h1>
      <div className="flex flex-wrap gap-3 items-end mb-4">
        <div className="flex flex-col">
          <label className="text-xs mb-1">Pair</label>
          <input className="bg-gray-900 border border-gray-700 rounded px-2 py-1" value={pair} onChange={(e) => setPair(e.target.value)} />
        </div>
        <div className="flex flex-col">
          <label className="text-xs mb-1">Timeframe</label>
          <input className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-24" value={timeframe} onChange={(e) => setTimeframe(e.target.value)} />
        </div>
        <div className="flex flex-col">
          <label className="text-xs mb-1">Timerange (YYYYMMDD-YYYYMMDD)</label>
          <input className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-64" value={timerange} onChange={(e) => setTimerange(e.target.value)} placeholder="20220101-20240101" />
        </div>
        <div className="flex flex-col">
          <label className="text-xs mb-1">Model Path</label>
          <input className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-[360px]" value={modelPath} onChange={(e) => setModelPath(e.target.value)} />
        </div>
        <button onClick={fetchCandles} className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded">Load Candles</button>
        <button onClick={runAgent} disabled={loading} className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-3 py-2 rounded">{loading ? "Running..." : "Run Agent"}</button>
      </div>

      <div ref={containerRef} className="w-full h-[560px] rounded border border-gray-700" />

      <div className="mt-4">
        <h2 className="text-lg font-semibold mb-2">Logs</h2>
        <div className="bg-gray-900 border border-gray-700 rounded p-3 max-h-72 overflow-auto text-xs">
          {logs.map((l, i) => (
            <div key={i} className="whitespace-pre">
              {typeof l === "string" ? l : JSON.stringify(l)}
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

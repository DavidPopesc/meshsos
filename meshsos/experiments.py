"""Run the four {payload} x {routing} variants, sweep node count, export results."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import codec, geo, radio, scenario, sim
from .channel import HARDWARE

LABELS = {
    "raw_flood": "Raw text + flooding",
    "raw_agent": "Raw text + agent routing",
    "compressed_flood": "Compressed + flooding",
    "compressed_agent": "Compressed + agent routing",
}
TABLE_COLS = [
    ("delivery_ratio", "Delivery", lambda v: f"{v * 100:5.1f}%"),
    ("latency_avg_s", "Lat avg", lambda v: f"{v:6.1f}s"),
    ("latency_p95_s", "Lat p95", lambda v: f"{v:6.1f}s"),
    ("airtime_s", "Airtime", lambda v: f"{v:6.0f}s"),
    ("collisions", "Collisions", lambda v: f"{v:6d}"),
    ("channel_utilization", "Chan util", lambda v: f"{v * 100:5.1f}%"),
]


def run_id(scenario_name: str, nodes: int, seed: int, sf: int) -> str:
    return f"{scenario_name}_n{nodes}_s{seed}_sf{sf}"


def build_scenario(name: str, nodes: int, seed: int, sf: int, duration: float) -> scenario.Scenario:
    if name != "quake":
        raise ValueError(f"unknown scenario {name!r} (available: quake)")
    return scenario.make_quake(nodes, seed, sf=sf, duration=duration)


def format_table(results: dict[str, dict], sc: scenario.Scenario | None = None) -> str:
    head = f"{'Variant':28s}" + "".join(f"{label:>12s}" for _, label, _ in TABLE_COLS)
    if sc is not None:
        head += f"{'Bytes/msg':>12s}"
    lines = [head, "-" * len(head)]
    for name, res in results.items():
        m = res["metrics"]
        row = f"{LABELS.get(name, name):28s}"
        for key, _, fmt in TABLE_COLS:
            v = m[key]
            row += f"{('n/a' if isinstance(v, float) and math.isnan(v) else fmt(v)):>12s}"
        if sc is not None:
            row += f"{bytes_per_message(sc, res['payload']):>12.1f}"
        lines.append(row)
    return "\n".join(lines)


def markdown_table(results: dict[str, dict], sc: scenario.Scenario) -> str:
    cols = ["Variant", "Delivery", "Latency avg", "Latency p95", "Airtime", "Collisions", "Channel util.", "Payload bytes/msg"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, res in results.items():
        m = res["metrics"]
        out.append("| " + " | ".join([
            LABELS[name], f"{m['delivery_ratio'] * 100:.1f}%", f"{m['latency_avg_s']:.1f} s", f"{m['latency_p95_s']:.1f} s",
            f"{m['airtime_s']:.0f} s", f"{m['collisions']:,}", f"{m['channel_utilization'] * 100:.1f}%",
            f"{bytes_per_message(sc, res['payload']):.1f}",
        ]) + " |")
    return "\n".join(out)


def bytes_per_message(sc: scenario.Scenario, payload: str) -> float:
    if payload == "compressed":
        return float(codec.PACKET_BYTES)
    return float(np.mean([m.raw_bytes for m in sc.messages])) if sc.messages else 0.0


def codec_accuracy(sc: scenario.Scenario) -> dict[str, float]:
    acc: dict[str, list[bool]] = {}
    for m in sc.messages:
        for k, v in scenario.field_accuracy(codec.unpack(m.packet), m.truth).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def export_timeline(sc: scenario.Scenario, results: dict[str, dict], path: Path, rid: str) -> dict:
    """Everything the 3D viewer needs, in one JSON file."""
    ch = sc.channel
    # Typical link range per node: 90th percentile distance of its usable links at t=0.
    ch.update(np.zeros(len(sc.buildings), dtype=bool))
    nbrs = ch.neighbors(sc.sf)
    ranges = []
    for i, js in enumerate(nbrs):
        d = ch.dist[i, js] if js else np.array([80.0])
        ranges.append(_r(np.percentile(d, 90), 1))
    collapse_t = {}
    for t, kind, payload in sc.events:
        if kind == "collapse":
            collapse_t[payload] = _r(t, 2)
    fail_t = {payload: _r(t, 2) for t, kind, payload in sc.events if kind == "fail"}

    data = {
        "meta": {
            "id": rid, "scenario": sc.name, "seed": sc.seed, "nodes": len(sc.nodes), "sf": sc.sf,
            "bandwidth_hz": radio.BW_HZ, "duration": sc.duration, "quake_t": sc.quake_t,
            "width_m": _r(geo.WIDTH_M, 1), "height_m": _r(geo.HEIGHT_M, 1),
            "bbox": {"lat0": geo.LAT0, "lon0": geo.LON0, "lat1": geo.LAT1, "lon1": geo.LON1},
            "grid": {"nx": geo.GRID_NX, "ny": geo.GRID_NY},
            "shoreline_y": _r(geo.SHORELINE_Y, 1),
            "sensitivity_dbm": _r(radio.sensitivity_dbm(sc.sf), 1),
            "event_fields": ["t", "type", "a", "b", "msg", "x"],
            "event_types": sim.EVENT_NAMES,
            "drop_reasons": sim.DROP_REASONS,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "codec_accuracy": codec_accuracy(sc),
            "hardware": {k: v["label"] for k, v in HARDWARE.items()},
        },
        "nodes": [{
            "id": n.id, "kind": n.kind, "x": _r(n.x, 1), "y": _r(n.y, 1),
            "lat": round(geo.to_latlon(n.x, n.y)[0], 6), "lon": round(geo.to_latlon(n.x, n.y)[1], 6),
            "battery0": _r(n.battery0, 1), "building": n.building, "range_m": ranges[n.id],
            "fail_t": fail_t.get(n.id),
        } for n in sc.nodes],
        "buildings": [{
            "id": b.id, "x": _r(b.x, 1), "y": _r(b.y, 1), "w": _r(b.w, 1), "d": _r(b.d, 1), "h": _r(b.h, 1),
            "kind": b.kind, "on_fill": b.on_fill, "collapse_t": collapse_t.get(b.id),
        } for b in sc.buildings],
        "zones": [{"kind": k, "x0": _r(a, 1), "y0": _r(b, 1), "x1": _r(c, 1), "y1": _r(d, 1)} for k, a, b, c, d in geo.ZONES],
        "streets": (
            [{"name": f"{k} St", "axis": "ns", "pos": _r(v, 1)} for k, v in geo.NS_X.items()]
            + [{"name": "Marina Blvd" if k == "Marina" else f"{k} St", "axis": "ew", "pos": _r(v, 1)} for k, v in geo.EW_Y.items()]
        ),
        "landmarks": [{"name": k.title(), "x": _r(x, 1), "y": _r(y, 1)} for k, (x, y) in geo.LANDMARKS.items()
                      if k in ("palace of fine arts", "marina green", "safeway", "marina middle school", "yacht harbor", "presidio", "fort mason")],
        "shocks": sc.shocks,
        "messages": [{
            "id": m.id, "node": m.node, "t": m.t, "text": m.text, "hex": m.packet.hex(),
            "raw_bytes": m.raw_bytes, "packed_bytes": len(m.packet), "fields": m.fields,
            "decoded": codec.decode(m.packet),
        } for m in sc.messages],
        "variants": {},
    }
    for name, res in results.items():
        data["variants"][name] = {
            "label": LABELS[name], "payload": res["payload"], "routing": res["routing"],
            "metrics": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in res["metrics"].items()},
            "delivered": {str(k): v for k, v in res["delivered"].items()},
            "steps": res["steps"],
            "events": res["events"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return data


def point_latest(out_dir: Path, rid: str) -> None:
    latest = out_dir / "latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.is_dir():
            import shutil
            shutil.rmtree(latest)
        latest.symlink_to(rid, target_is_directory=True)
    except OSError:                       # no symlinks (e.g. Windows without dev mode): copy
        import shutil
        shutil.copytree(out_dir / rid, latest)


def run(scenario_name: str = "quake", nodes: int = 50, seed: int = 1, sf: int = 9, duration: float = 900.0,
        out: str = "runs", charts: bool = True, verbose: bool = True) -> tuple[dict, Path]:
    t0 = time.time()
    sc = build_scenario(scenario_name, nodes, seed, sf, duration)
    results = sim.run_all(sc, record=True)
    rid = run_id(scenario_name, nodes, seed, sf)
    out_dir = Path(out)
    run_dir = out_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    export_timeline(sc, results, run_dir / "timeline.json", rid)
    summary = {
        "id": rid, "nodes": nodes, "seed": seed, "sf": sf, "duration": duration,
        "messages": len(sc.messages), "codec_accuracy": codec_accuracy(sc),
        "raw_bytes_avg": bytes_per_message(sc, "raw"), "packed_bytes": codec.PACKET_BYTES,
        "variants": {k: v["metrics"] for k, v in results.items()},
    }
    with open(run_dir / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda v: None)
    with open(run_dir / "results.md", "w") as fh:
        fh.write(markdown_table(results, sc) + "\n")
    if charts:
        from . import plots
        plots.variant_charts(sc, results, run_dir)
    point_latest(out_dir, rid)
    if verbose:
        n_raw = bytes_per_message(sc, "raw")
        print(f"MeshSOS · scenario={scenario_name} nodes={nodes} seed={seed} SF{sf}/125kHz · "
              f"{len(sc.messages)} SOS messages · {sum(1 for e in sc.events if e[1] == 'collapse')} collapses · "
              f"{sum(1 for e in sc.events if e[1] == 'fail')} node failures")
        print(f"payload: raw text avg {n_raw:.1f} B vs packed {codec.PACKET_BYTES} B "
              f"({n_raw / codec.PACKET_BYTES:.1f}x smaller) · max PHY payload under 400 ms dwell at SF{sf}: "
              f"{radio.max_payload(sf)} B\n")
        print(format_table(results, sc))
        acc = codec_accuracy(sc)
        print("\ncodec field accuracy on this run's messages: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in acc.items()))
        print(f"\nwrote {run_dir}/timeline.json, results.json" + (", *.png" if charts else "")
              + f"  ({time.time() - t0:.1f}s)  ->  {out_dir / 'latest'}")
    return results, run_dir


def sweep(node_counts=(20, 30, 40, 50, 60, 70, 80), seeds=(1, 2, 3), sf: int = 9, duration: float = 900.0,
          out: str = "runs", charts: bool = True, verbose: bool = True) -> dict:
    rows = []
    for n in node_counts:
        for s in seeds:
            sc = build_scenario("quake", n, s, sf, duration)
            res = sim.run_all(sc, record=False)
            for name, r in res.items():
                rows.append({"nodes": n, "seed": s, "variant": name, **r["metrics"]})
            if verbose:
                print(f"  nodes={n:3d} seed={s}  " + "  ".join(
                    f"{k}={v['metrics']['delivery_ratio'] * 100:5.1f}%" for k, v in res.items()), flush=True)
    agg: dict[str, dict[int, dict]] = {}
    for name in LABELS:
        agg[name] = {}
        for n in node_counts:
            sub = [r for r in rows if r["variant"] == name and r["nodes"] == n]
            agg[name][n] = {k: float(np.nanmean([r[k] for r in sub]))
                            for k in ("delivery_ratio", "latency_avg_s", "latency_p95_s", "airtime_s",
                                      "collisions", "channel_utilization")}
    sweep_dir = Path(out) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "sweep.json", "w") as fh:
        json.dump({"node_counts": list(node_counts), "seeds": list(seeds), "sf": sf, "rows": rows, "mean": agg},
                  fh, indent=1, default=lambda v: None)
    if charts:
        from . import plots
        plots.sweep_chart(agg, list(node_counts), sweep_dir)
    if verbose:
        print("\nmean delivery ratio by node count")
        print(f"{'nodes':>6s}" + "".join(f"{LABELS[k]:>30s}" for k in LABELS))
        for n in node_counts:
            print(f"{n:6d}" + "".join(f"{agg[k][n]['delivery_ratio'] * 100:29.1f}%" for k in LABELS))
        print(f"\nwrote {sweep_dir}/sweep.json" + (", sweep.png" if charts else ""))
    return agg

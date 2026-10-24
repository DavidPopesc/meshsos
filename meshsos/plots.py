"""Matplotlib charts for a run and for the node-count sweep."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import geo  # noqa: E402

# Reference categorical palette, slots 1-4 in fixed order (one slot per variant).
COLORS = {
    "raw_flood": "#2a78d6",
    "raw_agent": "#eb6834",
    "compressed_flood": "#1baf7a",
    "compressed_agent": "#eda100",
}
SHORT = {
    "raw_flood": "Raw · flood",
    "raw_agent": "Raw · agent",
    "compressed_flood": "Packed · flood",
    "compressed_agent": "Packed · agent",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a988f"
ALERT = "#e34948"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
    })


def _legend(fig, names, y=0.995):
    handles = [Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in names]
    fig.legend(handles, [SHORT[n] for n in names], loc="upper center", ncol=len(names),
               bbox_to_anchor=(0.5, y), handlelength=1.2, columnspacing=1.6)


def variant_charts(sc, results: dict[str, dict], out: Path) -> list[Path]:
    _style()
    names = list(results)
    paths = [_metrics_chart(results, names, out), _latency_cdf(results, names, out),
             _delivery_over_time(sc, results, names, out), map_chart(sc, out)]
    plt.close("all")
    return paths


def _metrics_chart(results, names, out: Path) -> Path:
    panels = [
        ("delivery_ratio", "Delivery ratio", lambda v: v * 100, "{:.0f}%"),
        ("latency_avg_s", "Average latency (s)", lambda v: v, "{:.1f}"),
        ("latency_p95_s", "95th-percentile latency (s)", lambda v: v, "{:.0f}"),
        ("airtime_s", "Total airtime (s)", lambda v: v, "{:.0f}"),
        ("collisions", "Collisions", lambda v: v, "{:,.0f}"),
        ("channel_utilization", "Channel utilization", lambda v: v * 100, "{:.1f}%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (key, title, tf, fmt) in zip(axes.flat, panels):
        vals = [tf(results[n]["metrics"][key]) for n in names]
        vals = [0 if (isinstance(v, float) and math.isnan(v)) else v for v in vals]
        x = np.arange(len(names))
        ax.bar(x, vals, width=0.72, color=[COLORS[n] for n in names], edgecolor=SURFACE, linewidth=2)
        top = max(vals) if max(vals) > 0 else 1
        for xi, v in zip(x, vals):
            ax.text(xi, v + top * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(x, [SHORT[n].replace(" · ", "\n") for n in names])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("")
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "metrics.png"
    fig.savefig(p, dpi=150)
    return p


def _latency_cdf(results, names, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n_msgs = next(iter(results.values()))["metrics"]["messages"]
    xmax = 1.0
    for n in names:
        lat = np.sort([d["latency"] for d in results[n]["delivered"].values()])
        if lat.size == 0:
            continue
        y = np.arange(1, lat.size + 1) / n_msgs * 100       # share of ALL messages, so the plateau = delivery ratio
        ax.step(np.r_[0, lat], np.r_[0, y], where="post", color=COLORS[n], linewidth=2, label=SHORT[n])
        ax.text(lat[-1], y[-1], f"  {SHORT[n]}", color=INK_2, fontsize=8.5, va="center")
        xmax = max(xmax, np.percentile(lat, 98))
    ax.set_xlim(0, xmax * 1.35)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Latency from SOS typed to gateway (s)")
    ax.set_ylabel("Messages delivered (% of all sent)")
    ax.set_title("Delivery vs. latency", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = out / "latency_cdf.png"
    fig.savefig(p, dpi=150)
    return p


def _delivery_over_time(sc, results, names, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.0))
    t = np.linspace(0, sc.duration, 400)
    gen = np.array([m.t for m in sc.messages])
    ax.plot(t / 60, [(gen <= x).sum() for x in t], color=NEUTRAL, linewidth=1.5, linestyle="--", label="SOS sent")
    for n in names:
        dt = np.array([d["t"] for d in results[n]["delivered"].values()])
        ax.plot(t / 60, [(dt <= x).sum() for x in t], color=COLORS[n], linewidth=2, label=SHORT[n])
    for s in sc.shocks:
        ax.axvline(s["t"] / 60, color=NEUTRAL, linewidth=1, alpha=0.6)
        ax.text(s["t"] / 60, ax.get_ylim()[1] * 0.97, f" M{s['mag']} {s['kind']}", color=INK_2, fontsize=8, va="top")
    ax.set_xlabel("Minutes into the simulation")
    ax.set_ylabel("Messages (cumulative)")
    ax.set_title("SOS messages reaching the gateway", loc="left")
    ax.legend(loc="upper left", ncol=5, bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    p = out / "delivery_over_time.png"
    fig.savefig(p, dpi=150)
    return p


def map_chart(sc, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    zone_colors = {"water": "#cfe3f2", "park": "#dcebd2", "harbor": "#cfe3f2", "forest": "#cfe0c4"}
    for kind, x0, y0, x1, y1 in geo.ZONES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=zone_colors[kind], zorder=0, linewidth=0))
    collapsed = {p for _, k, p in sc.events if k == "collapse"}
    for b in sc.buildings:
        c = ALERT if b.id in collapsed else ("#bdbab2" if b.kind != "landmark" else "#a39b8b")
        ax.add_patch(Rectangle((b.x - b.w / 2, b.y - b.d / 2), b.w, b.d, color=c, zorder=1, linewidth=0))
    sc.channel.update(np.zeros(len(sc.buildings), dtype=bool))
    for a, b, rssi in sc.channel.links(sc.sf, margin_db=3.0):
        if a < b:
            na, nb = sc.nodes[a], sc.nodes[b]
            ax.plot([na.x, nb.x], [na.y, nb.y], color="#2a78d6", alpha=0.18, linewidth=0.8, zorder=2)
    for n in sc.nodes:
        if n.kind == "gateway":
            ax.scatter(n.x, n.y, s=180, marker="*", color="#eda100", edgecolor=INK, linewidth=0.8, zorder=4)
            ax.annotate("gateway", (n.x, n.y), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK)
        else:
            ax.scatter(n.x, n.y, s=36 if n.kind == "tdeck" else 18, marker="s" if n.kind == "tdeck" else "o",
                       color=INK, edgecolor=SURFACE, linewidth=1, zorder=3)
    for name in ("palace of fine arts", "marina green", "yacht harbor", "presidio"):
        x, y = geo.LANDMARKS[name]
        ax.text(x, y + 25, name.title(), fontsize=8, color=INK_2, ha="center", zorder=5)
    for k, x in geo.NS_X.items():
        ax.text(x, -14, k, fontsize=6.5, color=INK_2, ha="center", rotation=90, va="top")
    for k, y in geo.EW_Y.items():
        ax.text(geo.WIDTH_M + 8, y, "Marina Blvd" if k == "Marina" else k, fontsize=6.5, color=INK_2, va="center")
    ax.set_xlim(0, geo.WIDTH_M)
    ax.set_ylim(-90, geo.HEIGHT_M)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Marina / Palace of Fine Arts · {len(sc.nodes)} nodes · collapsed buildings in red · links ≥3 dB margin",
                 loc="left")
    fig.tight_layout()
    p = out / "map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def sweep_chart(agg: dict, node_counts: list[int], out: Path) -> Path:
    _style()
    names = list(agg)
    panels = [("delivery_ratio", "Delivery ratio (%)", 100), ("airtime_s", "Total airtime (s)", 1),
              ("latency_p95_s", "95th-percentile latency (s)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title, scale) in zip(axes, panels):
        for n in names:
            ys = [agg[n][c][key] * scale for c in node_counts]
            ax.plot(node_counts, ys, color=COLORS[n], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=SHORT[n])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Nodes in the mesh")
        ax.set_xticks(node_counts)
        if key == "delivery_ratio":
            ax.set_ylim(0, 100)
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out / "sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p

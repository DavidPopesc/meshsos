"""Command line: python -m meshsos {run,sweep,view,encode,decode}."""

from __future__ import annotations

import argparse
import sys

from . import codec, experiments


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m meshsos", description="MeshSOS LoRa mesh simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run all four variants on one scenario")
    r.add_argument("--scenario", default="quake")
    r.add_argument("--nodes", type=int, default=50)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--sf", type=int, default=9, choices=[7, 8, 9, 10], help="spreading factor (SF11/12 break the 400 ms dwell limit)")
    r.add_argument("--duration", type=float, default=900.0, help="simulated seconds")
    r.add_argument("--out", default="runs")
    r.add_argument("--no-charts", action="store_true")
    r.add_argument("--sweep", action="store_true", help="also sweep node count afterwards")
    r.add_argument("--llm", action="store_true", help="re-decode a few delivered packets with Claude (needs ANTHROPIC_API_KEY)")

    s = sub.add_parser("sweep", help="sweep node count to find where flooding falls apart")
    s.add_argument("--nodes", type=int, nargs="+", default=[20, 30, 40, 50, 60, 70, 80])
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--sf", type=int, default=9, choices=[7, 8, 9, 10])
    s.add_argument("--duration", type=float, default=900.0)
    s.add_argument("--out", default="runs")
    s.add_argument("--no-charts", action="store_true")

    v = sub.add_parser("view", help="open the 3D viewer for a run")
    v.add_argument("--run", default="runs/latest")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--no-browser", action="store_true")
    v.add_argument("--demo", action="store_true", help="open in demo mode (auto-play fly-through)")
    v.add_argument("--capture-dir", default=None, help=argparse.SUPPRESS)

    e = sub.add_parser("encode", help="compress a free-text SOS into a packet")
    e.add_argument("text")
    d = sub.add_parser("decode", help="decode a packet (hex) back into text")
    d.add_argument("hex")
    d.add_argument("--llm", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        results, run_dir = experiments.run(a.scenario, a.nodes, a.seed, a.sf, a.duration, a.out, charts=not a.no_charts)
        if a.llm:
            _llm_demo(run_dir)
        if a.sweep:
            print("\nsweeping node count ...")
            experiments.sweep(seeds=(1, 2, 3), sf=a.sf, duration=a.duration, out=a.out, charts=not a.no_charts)
    elif a.cmd == "sweep":
        experiments.sweep(tuple(a.nodes), tuple(range(1, a.seeds + 1)), a.sf, a.duration, a.out, charts=not a.no_charts)
    elif a.cmd == "view":
        from . import server
        server.serve(a.run, a.port, not a.no_browser, "demo=1" if a.demo else "", a.capture_dir)
    elif a.cmd == "encode":
        f = codec.encode_fields(a.text)
        data = codec.pack(f)
        print(f"{len(a.text.encode())} B text -> {len(data)} B packet: {data.hex()}")
        print(codec.describe(f))
    elif a.cmd == "decode":
        llm = codec.anthropic_llm() if a.llm else None
        print(codec.decode(bytes.fromhex(a.hex), llm=llm))
    return 0


def _llm_demo(run_dir) -> None:
    import json
    with open(run_dir / "timeline.json") as fh:
        tl = json.load(fh)
    try:
        llm = codec.anthropic_llm()
    except RuntimeError as exc:
        print(f"\n--llm skipped: {exc}")
        return
    delivered = tl["variants"]["compressed_agent"]["delivered"]
    print("\nLLM gateway decoding (compressed + agent routing, first 5 delivered):")
    for mid in list(delivered)[:5]:
        m = tl["messages"][int(mid)]
        print(f"  sent : {m['text']}\n  wire : {m['hex']}\n  rule : {m['decoded']}\n  llm  : {codec.decode(bytes.fromhex(m['hex']), llm=llm)}\n")


if __name__ == "__main__":
    sys.exit(main())

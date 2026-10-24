import json

from meshsos import experiments, scenario, sim


def test_scenario_is_reproducible():
    a = scenario.make_quake(30, seed=4)
    b = scenario.make_quake(30, seed=4)
    assert [m.text for m in a.messages] == [m.text for m in b.messages]
    assert a.events == b.events


def test_all_variants_run_and_report(tmp_path):
    sc = scenario.make_quake(25, seed=2, duration=400)
    results = sim.run_all(sc)
    assert set(results) == {"raw_flood", "raw_agent", "compressed_flood", "compressed_agent"}
    for r in results.values():
        m = r["metrics"]
        assert 0 <= m["delivery_ratio"] <= 1
        assert m["airtime_s"] > 0
    # compressed frames are shorter, so they use less airtime under the same routing
    assert results["compressed_flood"]["metrics"]["airtime_data_s"] < results["raw_flood"]["metrics"]["airtime_data_s"]
    data = experiments.export_timeline(sc, results, tmp_path / "timeline.json", "test")
    loaded = json.loads((tmp_path / "timeline.json").read_text())
    assert loaded["meta"]["nodes"] == 25
    v = loaded["variants"]["compressed_agent"]
    assert v["steps"][0]["alive"] and "nbrs" in v["steps"][0] and "battery" in v["steps"][0]
    kinds = {sim.EVENT_NAMES[e[1]] for e in v["events"]}
    assert {"send", "hop"} <= kinds
    assert data["messages"][0]["hex"]


def test_frames_respect_dwell_limit():
    sc = scenario.make_quake(20, seed=1, duration=300)
    s = sim.Simulator(sc, sim.Config(payload="raw", routing="flood"))
    s.run()
    assert max(s.toa_cache.values()) <= 0.400

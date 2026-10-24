import numpy as np
import pytest

from meshsos import codec, geo, scenario
from meshsos.codec import Hazard, Injury, MsgType, Need, SosFields


def test_packet_is_ten_bytes_and_fits_layout():
    assert codec.PAYLOAD_BITS == 75
    assert len(codec.encode("help")) == codec.PACKET_BYTES == 10


def test_round_trip_all_field_extremes():
    for vals in (
        dict(type=0, severity=0, people=0, kids=0, injuries=0, needs=0, hazards=0, floor=0, loc_src=0,
             cell_x=0, cell_y=0, minute=0, msg_id=0),
        dict(type=15, severity=3, people=31, kids=7, injuries=255, needs=255, hazards=15, floor=7, loc_src=1,
             cell_x=63, cell_y=31, minute=255, msg_id=0xFFFF),
        dict(type=MsgType.TRAPPED, severity=2, people=3, kids=2, injuries=Injury.BLEEDING, needs=Need.WATER,
             hazards=Hazard.GAS, floor=3, loc_src=1, cell_x=17, cell_y=9, minute=42, msg_id=0xBEEF),
    ):
        f = SosFields(**vals)
        assert codec.unpack(codec.pack(f)) == f


def test_random_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(500):
        f = SosFields(**{name: int(rng.integers(0, 1 << bits)) for name, bits in codec.LAYOUT if name != "version"})
        assert codec.unpack(codec.pack(f)) == f


def test_out_of_range_field_rejected():
    with pytest.raises(ValueError):
        codec.pack(SosFields(people=32))
    with pytest.raises(ValueError):
        codec.unpack(b"\x00" * 9)


def test_spec_example():
    text = "trapped under debris at 3rd and Bay, 2 kids, one bleeding badly, need water"
    f = codec.encode_fields(text, node_xy=(500.0, 400.0), minute=7, msg_id=0x1234)
    assert f.type == MsgType.TRAPPED
    assert f.severity == 3
    assert f.people == 3 and f.kids == 2          # the sender plus two children
    assert f.injuries == Injury.BLEEDING
    assert f.needs & Need.WATER and f.needs & Need.RESCUE and f.needs & Need.MEDIC
    assert f.loc_src == 0                         # "3rd and Bay" is not in the Marina gazetteer: GPS fallback
    assert (f.cell_x, f.cell_y) == geo.cell_of(500.0, 400.0)
    out = codec.decode(codec.pack(f))
    assert "TRAPPED" in out and "2 children" in out and "bleeding" in out and "water" in out and "#1234" in out


def test_intersection_and_landmark_lookup():
    f = codec.encode_fields("fire at Fillmore and Chestnut, 4 people")
    assert f.loc_src == 1 and f.type == MsgType.FIRE and f.hazards & Hazard.FIRE and f.people == 4
    assert (f.cell_x, f.cell_y) == geo.cell_of(geo.NS_X["Fillmore"], geo.EW_Y["Chestnut"])
    g = codec.encode_fields("we are ok at the Palace of Fine Arts, family of 5")
    assert g.type == MsgType.CHECKIN and g.severity == 0 and g.people == 5
    assert "Palace" in codec.describe(g)


def test_decode_round_trip_preserves_meaning_on_generated_corpus():
    rng = np.random.default_rng(3)
    correct = []
    for _ in range(300):
        text, truth = scenario.make_sos(rng, (800.0, 500.0))
        pkt = codec.encode(text, node_xy=(800.0, 500.0))
        f = codec.unpack(pkt)
        correct.append(all(scenario.field_accuracy(f, truth).values()))
        assert codec.decode(pkt)       # always produces text
    assert np.mean(correct) > 0.85


def test_llm_hook_used_and_falls_back():
    pkt = codec.encode("gas leak at Scott and Bay, 2 people")
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "Gas leak reported near Scott & Bay; two people on site."

    assert codec.decode(pkt, llm=fake_llm).startswith("Gas leak reported")
    assert "GAS_LEAK" in seen["prompt"]

    def broken_llm(prompt):
        raise TimeoutError

    assert codec.decode(pkt, llm=broken_llm) == codec.describe(codec.unpack(pkt))

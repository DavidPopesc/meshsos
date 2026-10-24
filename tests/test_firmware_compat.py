"""Compile firmware/meshsos_codec.h on the host and check it against the Python codec."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from meshsos import codec, radio

ROOT = Path(__file__).resolve().parent.parent
CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")

HARNESS = r"""
#include "meshsos_codec.h"
#include <iostream>
#include <string>
int main() {
  std::string cmd;
  while (std::cin >> cmd) {
    if (cmd == "P") {            // pack: 14 field values -> hex
      unsigned v[14];
      for (auto& x : v) std::cin >> x;
      meshsos::SosPacket p;
      p.type = v[0]; p.severity = v[1]; p.people = v[2]; p.kids = v[3]; p.injuries = v[4]; p.needs = v[5];
      p.hazards = v[6]; p.floor = v[7]; p.loc_src = v[8]; p.cell_x = v[9]; p.cell_y = v[10];
      p.minute = v[11]; p.msg_id = v[12]; (void)v[13];
      uint8_t buf[10];
      meshsos::pack(p, buf);
      for (auto b : buf) printf("%02x", b);
      printf("\n");
    } else if (cmd == "U") {     // unpack hex -> fields
      std::string hex; std::cin >> hex;
      uint8_t buf[10];
      for (int i = 0; i < 10; ++i) buf[i] = (uint8_t)std::stoi(hex.substr(2 * i, 2), nullptr, 16);
      meshsos::SosPacket p;
      bool ok = meshsos::unpack(buf, p);
      printf("%d %u %u %u %u %u %u %u %u %u %u %u %u %u\n", ok, p.type, p.severity, p.people, p.kids, p.injuries,
             p.needs, p.hazards, p.floor, p.loc_src, p.cell_x, p.cell_y, p.minute, p.msg_id);
    } else if (cmd == "E") {     // encode a line of text at node (500, 400)
      std::string line; std::getline(std::cin, line);
      meshsos::SosPacket p = meshsos::encodeText(line.c_str() + 1, 500.0f, 400.0f, 0, 0);
      printf("%u %u %u %u %u %u %u %u\n", p.type, p.people, p.kids, p.injuries, p.needs, p.hazards, p.loc_src, p.floor);
    } else if (cmd == "T") {     // time on air
      unsigned n, sf; std::cin >> n >> sf;
      printf("%.4f\n", meshsos::timeOnAirMs(n, sf));
    }
  }
}
"""

FIELDS = ["type", "severity", "people", "kids", "injuries", "needs", "hazards", "floor", "loc_src",
          "cell_x", "cell_y", "minute", "msg_id"]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if CXX is None:
        pytest.skip("no C++ compiler on this machine")
    d = tmp_path_factory.mktemp("fw")
    src = d / "harness.cpp"
    src.write_text(HARNESS)
    exe = d / "harness"
    subprocess.run([CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "firmware"), str(src),
                    "-o", str(exe)], check=True, capture_output=True)

    def call(lines: list[str]) -> list[str]:
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()

    return call


def test_pack_is_bit_exact(harness):
    rng = np.random.default_rng(11)
    cases = []
    for _ in range(300):
        cases.append(codec.SosFields(**{n: int(rng.integers(0, 1 << b)) for n, b in codec.LAYOUT if n != "version"}))
    out = harness(["P " + " ".join(str(getattr(f, n)) for n in FIELDS) + " 0" for f in cases])
    assert out == [codec.pack(f).hex() for f in cases]


def test_unpack_matches(harness):
    pkts = [codec.encode("trapped under debris at Scott and Bay, 2 kids, need water", (400, 300), 12, 0xABCD),
            codec.encode("we are ok", (10, 10), 255, 1)]
    out = harness([f"U {p.hex()}" for p in pkts])
    for line, p in zip(out, pkts):
        vals = list(map(int, line.split()))
        assert vals[0] == 1
        f = codec.unpack(p)
        assert vals[1:] == [getattr(f, n) for n in FIELDS]


def test_keyword_encoder_agrees_on_common_phrasings(harness):
    texts = [
        "trapped under debris at Scott and Bay, 2 kids, one bleeding badly, need water",
        "fire in the building at Fillmore and Chestnut, 4 people, send firefighters",
        "strong gas smell at Divisadero and Lombard, family of 3",
        "grandma is unconscious at Pierce and Francisco, need an ambulance",
        "we are ok at the Palace of Fine Arts, 5 people",
        "power lines down at Webster and Beach, sparking",
        "building collapsed at Baker & North Point, three of us inside, please send rescue",
        "at Laguna and Marina Blvd, out of water, need blankets",
    ]
    out = harness([f"E {t}" for t in texts])
    for t, line in zip(texts, out):
        f = codec.encode_fields(t, (500.0, 400.0))
        cpp = list(map(int, line.split()))
        assert cpp == [f.type, f.people, f.kids, f.injuries, f.needs, f.hazards, f.loc_src, f.floor], t


def test_airtime_matches(harness):
    pairs = [(n, sf) for sf in (7, 8, 9, 10, 11, 12) for n in (0, 10, 20, 50, 100)]
    out = harness([f"T {n} {sf}" for n, sf in pairs])
    for (n, sf), line in zip(pairs, out):
        assert float(line) == pytest.approx(radio.time_on_air(n, sf) * 1e3, abs=0.01)

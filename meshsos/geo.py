"""Map of the SF Marina / Palace of Fine Arts area in local metres.

The street grid is an approximation good to a few tens of metres: enough for
a radio simulation and a recognisable 3D scene, not for navigation.
x grows east, y grows north, origin at the south-west corner of the box.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

LAT0, LON0 = 37.7965, -122.4515   # south-west corner
LAT1, LON1 = 37.8085, -122.4295   # north-east corner
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians((LAT0 + LAT1) / 2))
WIDTH_M = (LON1 - LON0) * M_PER_DEG_LON
HEIGHT_M = (LAT1 - LAT0) * M_PER_DEG_LAT

# Location cell grid carried in the compressed packet: 6 bits x 5 bits.
GRID_NX, GRID_NY = 64, 32


def to_xy(lat: float, lon: float) -> tuple[float, float]:
    return (lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT


def to_latlon(x: float, y: float) -> tuple[float, float]:
    return LAT0 + y / M_PER_DEG_LAT, LON0 + x / M_PER_DEG_LON


def cell_of(x: float, y: float) -> tuple[int, int]:
    gx = int(np.clip(x / WIDTH_M * GRID_NX, 0, GRID_NX - 1))
    gy = int(np.clip(y / HEIGHT_M * GRID_NY, 0, GRID_NY - 1))
    return gx, gy


def cell_center(gx: int, gy: int) -> tuple[float, float]:
    return (gx + 0.5) * WIDTH_M / GRID_NX, (gy + 0.5) * HEIGHT_M / GRID_NY


# North-south streets, west to east (longitude).
NS_STREETS = {
    "Lyon": -122.4494, "Baker": -122.4459, "Broderick": -122.4442, "Divisadero": -122.4426,
    "Scott": -122.4410, "Pierce": -122.4393, "Steiner": -122.4377, "Fillmore": -122.4360,
    "Webster": -122.4343, "Buchanan": -122.4326, "Laguna": -122.4310,
}
# East-west streets, south to north (latitude).
EW_STREETS = {
    "Greenwich": 37.7974, "Lombard": 37.7988, "Chestnut": 37.8006, "Francisco": 37.8020,
    "Bay": 37.8034, "North Point": 37.8044, "Beach": 37.8054, "Marina": 37.8066,
}

NS_X = {k: (v - LON0) * M_PER_DEG_LON for k, v in NS_STREETS.items()}
EW_Y = {k: (v - LAT0) * M_PER_DEG_LAT for k, v in EW_STREETS.items()}
SHORELINE_Y = (37.8076 - LAT0) * M_PER_DEG_LAT

LANDMARKS = {
    "palace of fine arts": to_xy(37.8027, -122.4482),
    "palace": to_xy(37.8027, -122.4482),
    "rotunda": to_xy(37.8027, -122.4482),
    "lagoon": to_xy(37.8037, -122.4469),
    "marina green": to_xy(37.8070, -122.4380),
    "yacht harbor": to_xy(37.8078, -122.4440),
    "harbor": to_xy(37.8078, -122.4440),
    "safeway": to_xy(37.8050, -122.4325),
    "marina middle school": to_xy(37.8030, -122.4365),
    "moscone rec": to_xy(37.8015, -122.4318),
    "moscone": to_xy(37.8015, -122.4318),
    "presidio": to_xy(37.8000, -122.4505),
    "fort mason": to_xy(37.8060, -122.4300),
    "chestnut st shops": to_xy(37.8006, -122.4370),
}

# Open areas drawn on the ground plane and kept free of buildings: (name, xmin, ymin, xmax, ymax).
ZONES = [
    ("water", 0.0, SHORELINE_Y, WIDTH_M, HEIGHT_M),
    ("park", NS_X["Scott"] - 60, EW_Y["Marina"] + 15, NS_X["Laguna"] + 40, SHORELINE_Y),   # Marina Green
    ("park", NS_X["Lyon"] - 20, EW_Y["Bay"] + 10, NS_X["Baker"] + 15, EW_Y["Marina"] + 60),  # Palace grounds
    ("water", LANDMARKS["rotunda"][0] + 24, EW_Y["Bay"] + 25, NS_X["Baker"] + 5, EW_Y["Beach"] + 15),  # lagoon
    ("harbor", NS_X["Baker"] + 25, EW_Y["Marina"] + 15, NS_X["Scott"] - 70, SHORELINE_Y),
    ("forest", 0.0, 0.0, NS_X["Lyon"] - 25, SHORELINE_Y),                                  # Presidio
]


def in_zone(x: float, y: float, kinds=("water", "park", "harbor", "forest")) -> str | None:
    for kind, x0, y0, x1, y1 in ZONES:
        if kind in kinds and x0 <= x <= x1 and y0 <= y <= y1:
            return kind
    return None


@dataclass
class Building:
    id: int
    x: float      # centre
    y: float
    w: float      # east-west size
    d: float      # north-south size
    h: float
    kind: str     # residential | commercial | landmark
    on_fill: bool  # built on 1906 rubble / bay fill: liquefaction risk (see 1989 Loma Prieta)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x - self.w / 2, self.y - self.d / 2, self.x + self.w / 2, self.y + self.d / 2


def build_city(rng: np.random.Generator) -> list[Building]:
    """Procedural 2-4 storey blocks along the real street grid."""
    xs = sorted(NS_X.values())
    ys = sorted(EW_Y.values())
    street_half = 11.0
    out: list[Building] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            bx0, bx1 = xs[i] + street_half, xs[i + 1] - street_half
            by0, by1 = ys[j] + street_half, ys[j + 1] - street_half
            if bx1 - bx0 < 20 or by1 - by0 < 20:
                continue
            commercial = abs(ys[j] - EW_Y["Chestnut"]) < 1 or abs(ys[j + 1] - EW_Y["Chestnut"]) < 1
            # Buildings line the long north/south faces of each block.
            for face_y, depth in ((by0, 1), (by1, -1)):
                x = bx0
                while x < bx1 - 8:
                    w = float(rng.uniform(9, 16))
                    w = min(w, bx1 - x)
                    d = float(rng.uniform(14, 22))
                    d = min(d, (by1 - by0) / 2 - 2)
                    cx, cy = x + w / 2, face_y + depth * d / 2
                    if in_zone(cx, cy) is None and rng.random() > 0.08:
                        h = float(rng.uniform(7, 13)) if not commercial else float(rng.uniform(9, 16))
                        out.append(Building(len(out), cx, cy, w - 1.2, d, h,
                                            "commercial" if commercial else "residential",
                                            on_fill=cy > EW_Y["Chestnut"]))
                    x += w
    # The Palace of Fine Arts rotunda: tall, but a 2016 seismic retrofit makes it unlikely to fall.
    px, py = LANDMARKS["rotunda"]
    out.append(Building(len(out), px, py, 34, 34, 34, "landmark", on_fill=False))
    return out


_STREET_ALIASES = {
    "divis": "Divisadero", "marina blvd": "Marina", "northpoint": "North Point", "north pt": "North Point",
}


def _match_street(word: str) -> str | None:
    word = word.strip().lower()
    for alias, name in _STREET_ALIASES.items():
        if word.endswith(alias):
            return name
    for name in list(NS_STREETS) + list(EW_STREETS):
        if word.endswith(name.lower()):
            return name
    return None


def find_location(text: str) -> tuple[float, float, str] | None:
    """Rule-based gazetteer lookup: landmarks first, then 'X and Y' intersections."""
    t = text.lower()
    for name, (x, y) in LANDMARKS.items():
        if name in t:
            return x, y, name
    for m in re.finditer(r"([a-z]+(?: point| pt| blvd)?)\s*(?:st\.?|street|blvd)?\s*(?:and|&|/|\+)\s*([a-z]+(?: point| pt| blvd)?)", t):
        a, b = _match_street(m.group(1)), _match_street(m.group(2))
        if a and b:
            if a in NS_X and b in EW_Y:
                return NS_X[a], EW_Y[b], f"{a} & {b}"
            if b in NS_X and a in EW_Y:
                return NS_X[b], EW_Y[a], f"{b} & {a}"
    return None


def describe_location(x: float, y: float) -> str:
    """Reverse geocode to the nearest landmark or intersection."""
    best_name, best_d = "", 1e9
    for name, (lx, ly) in LANDMARKS.items():
        d = math.hypot(lx - x, ly - y)
        if d < best_d and d < 90:
            best_name, best_d = name.title(), d
    if best_name:
        return f"near {best_name}"
    ns = min(NS_X, key=lambda k: abs(NS_X[k] - x))
    ew = min(EW_Y, key=lambda k: abs(EW_Y[k] - y))
    ew_name = "Marina Blvd" if ew == "Marina" else f"{ew} St"
    return f"near {ns} St & {ew_name}"

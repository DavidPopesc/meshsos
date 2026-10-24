"""Pairwise link budget between every node, updated as buildings collapse."""

from __future__ import annotations

import numpy as np

from . import radio

# Per-hardware radio parameters: tx power (dBm), antenna gain (dBi), wall factor.
# The wall factor scales building losses: the gateway's mast clears most rooftops.
HARDWARE = {
    "t3": {"tx_dbm": 17.0, "gain_dbi": 0.0, "walls": 1.0, "label": "LilyGO T3 LoRa32 V1.6.1 (ESP32 + SX1276)"},
    "tdeck": {"tx_dbm": 20.0, "gain_dbi": 0.0, "walls": 1.0, "label": "LilyGO T-Deck Plus (ESP32-S3 + SX1262)"},
    "gateway": {"tx_dbm": 20.0, "gain_dbi": 3.0, "walls": 0.6, "label": "Gateway (SX1262, 3 dBi antenna on a 6 m mast)"},
}


class Channel:
    def __init__(self, xy: np.ndarray, kinds: list[str], boxes: np.ndarray,
                 rng: np.random.Generator, model: radio.PathLossModel | None = None):
        self.model = model or radio.PathLossModel()
        self.xy = np.asarray(xy, dtype=float)
        self.n = len(self.xy)
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.tx = np.array([HARDWARE[k]["tx_dbm"] for k in kinds])
        self.gain = np.array([HARDWARE[k]["gain_dbi"] for k in kinds])
        walls = np.array([HARDWARE[k]["walls"] for k in kinds])
        self.wall_factor = np.minimum(walls[:, None], walls[None, :])

        diff = self.xy[:, None, :] - self.xy[None, :, :]
        self.dist = np.hypot(diff[..., 0], diff[..., 1])
        s = rng.normal(0.0, self.model.shadow_sigma_db, (self.n, self.n))
        self.shadow = np.triu(s, 1) + np.triu(s, 1).T          # reciprocal shadowing

        iu, ju = np.triu_indices(self.n, 1)
        self._pairs = (iu, ju)
        self._cross = radio.segment_box_crossings(self.xy[iu], self.xy[ju], self.boxes)   # (M, P)
        self.collapsed = np.zeros(len(self.boxes), dtype=bool)
        self.update(self.collapsed)

    def update(self, collapsed: np.ndarray) -> None:
        self.collapsed = np.asarray(collapsed, dtype=bool).copy()
        iu, ju = self._pairs
        intact_p = (~self.collapsed).astype(float) @ self._cross
        rubble_p = self.collapsed.astype(float) @ self._cross
        intact = np.zeros((self.n, self.n))
        rubble = np.zeros((self.n, self.n))
        intact[iu, ju] = intact_p
        intact[ju, iu] = intact_p
        rubble[iu, ju] = rubble_p
        rubble[ju, iu] = rubble_p
        self.walls_intact, self.walls_rubble = intact, rubble
        pl = self.model.loss_db(self.dist, intact * self.wall_factor, rubble * self.wall_factor, self.shadow)
        # rssi[i, j]: mean power node j receives when node i transmits.
        self.rssi = self.tx[:, None] + self.gain[:, None] + self.gain[None, :] - pl
        np.fill_diagonal(self.rssi, -np.inf)

    def links(self, sf: int, margin_db: float = 0.0, alive=None) -> list[tuple[int, int, float]]:
        """Directed links whose mean RSSI clears sensitivity + margin."""
        ok = self.rssi >= radio.sensitivity_dbm(sf) + margin_db
        if alive is not None:
            alive = np.asarray(alive, dtype=bool)
            ok &= alive[:, None] & alive[None, :]
        i, j = np.nonzero(ok)
        return [(int(a), int(b), float(self.rssi[a, b])) for a, b in zip(i, j)]

    def neighbors(self, sf: int, margin_db: float = 0.0, alive=None) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(self.n)]
        for a, b, _ in self.links(sf, margin_db, alive):
            out[a].append(b)
        return out

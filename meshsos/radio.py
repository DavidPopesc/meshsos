"""LoRa radio model: path loss, time-on-air, link budget and capture.

Everything here is plain numpy so the same numbers can be reproduced on a
laptop or reasoned about on the ESP32 side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FREQ_HZ = 915e6            # US915 ISM band
BW_HZ = 125e3              # 125 kHz channel
NOISE_FIGURE_DB = 6.0      # typical SX127x/SX126x receiver NF
NOISE_FLOOR_DBM = -174.0 + 10.0 * math.log10(BW_HZ) + NOISE_FIGURE_DB   # ~ -117 dBm

# Demodulation SNR floor per spreading factor (Semtech SX1276 datasheet, table 13).
SNR_LIMIT_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

CAPTURE_DB = 6.0           # a frame this much stronger than the interference survives
MAX_DWELL_S = 0.400        # US915 400 ms max dwell time per uplink (FCC 15.247 / LoRaWAN RP002)


class DwellTimeExceeded(ValueError):
    """Raised when a frame would stay on air longer than the US915 dwell limit."""


def sensitivity_dbm(sf: int) -> float:
    """Receiver sensitivity at 125 kHz: thermal noise + NF + SNR floor."""
    return NOISE_FLOOR_DBM + SNR_LIMIT_DB[sf]


def symbol_time(sf: int, bw: float = BW_HZ) -> float:
    return (2 ** sf) / bw


def time_on_air(
    payload_len: int,
    sf: int = 9,
    bw: float = BW_HZ,
    cr: int = 1,
    preamble: int = 8,
    explicit_header: bool = True,
    crc: bool = True,
    low_dr_optimize: bool | None = None,
) -> float:
    """LoRa time-on-air in seconds (Semtech AN1200.13 / SX1276 datasheet 4.1.1.7).

    ``cr`` is the coding-rate index: 1 -> 4/5, 2 -> 4/6, 3 -> 4/7, 4 -> 4/8.
    Low data rate optimisation is switched on automatically when the symbol
    time exceeds 16 ms (SF11/SF12 at 125 kHz), as the radios require.
    """
    if not 7 <= sf <= 12:
        raise ValueError(f"SF must be 7..12, got {sf}")
    if not 0 <= payload_len <= 255:
        raise ValueError(f"payload must be 0..255 bytes, got {payload_len}")
    t_sym = symbol_time(sf, bw)
    if low_dr_optimize is None:
        low_dr_optimize = t_sym > 0.016
    de = 1 if low_dr_optimize else 0
    ih = 0 if explicit_header else 1
    t_preamble = (preamble + 4.25) * t_sym
    num = 8 * payload_len - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym


def check_dwell(toa_s: float, limit_s: float = MAX_DWELL_S) -> float:
    if toa_s > limit_s:
        raise DwellTimeExceeded(f"time-on-air {toa_s * 1e3:.1f} ms exceeds {limit_s * 1e3:.0f} ms dwell limit")
    return toa_s


def max_payload(sf: int, limit_s: float = MAX_DWELL_S, **kw) -> int:
    """Largest PHY payload (bytes) that still fits inside the dwell limit; 0 if none fits."""
    best = 0
    for n in range(256):
        if time_on_air(n, sf, **kw) <= limit_s:
            best = n
        else:
            break
    return best


def fspl_1m_db(freq_hz: float = FREQ_HZ) -> float:
    lam = 299_792_458.0 / freq_hz
    return 20.0 * math.log10(4.0 * math.pi / lam)


@dataclass
class PathLossModel:
    """Log-distance path loss with per-wall attenuation and log-normal shadowing.

    PL(d) = PL(1 m) + 10 n log10(d) + walls * intact_db + rubble * rubble_db + X_sigma
    Antennas sit at body height in a dense 2-4 storey neighbourhood, so the
    exponent is well above free space.
    """

    exponent: float = 3.4
    intact_db: float = 6.0       # per intact building crossed
    rubble_db: float = 11.0      # per collapsed building (rebar, concrete, wet debris)
    shadow_sigma_db: float = 4.0
    fast_fade_sigma_db: float = 2.0
    max_walls: int = 8           # beyond this the signal diffracts over the roofs instead
    pl0_db: float = fspl_1m_db()

    def loss_db(self, d_m, intact, rubble, shadow_db=0.0):
        d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
        intact = np.asarray(intact, dtype=float)
        rubble = np.asarray(rubble, dtype=float)
        total = intact + rubble
        scale = np.where(total > self.max_walls, self.max_walls / np.maximum(total, 1), 1.0)
        walls = (intact * self.intact_db + rubble * self.rubble_db) * scale
        return self.pl0_db + 10.0 * self.exponent * np.log10(d) + walls + shadow_db


def segment_box_crossings(p0: np.ndarray, p1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Which axis-aligned boxes each segment passes through (vectorised Liang-Barsky).

    p0, p1: (N, 2) segment endpoints. boxes: (M, 4) as xmin, ymin, xmax, ymax.
    Returns an (M, N) boolean matrix.
    """
    if len(boxes) == 0 or len(p0) == 0:
        return np.zeros((len(boxes), len(p0)), dtype=bool)
    x0, y0 = p0[:, 0][None, :], p0[:, 1][None, :]
    dx, dy = (p1[:, 0] - p0[:, 0])[None, :], (p1[:, 1] - p0[:, 1])[None, :]
    xmin, ymin, xmax, ymax = (boxes[:, k][:, None] for k in range(4))
    t0 = np.zeros((len(boxes), len(p0)))
    t1 = np.ones((len(boxes), len(p0)))
    ok = np.ones_like(t0, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            p = np.broadcast_to(p, t0.shape)
            q = np.broadcast_to(q, t0.shape)
            parallel = p == 0
            ok &= ~(parallel & (q < 0))
            r = q / p
            t0 = np.where(~parallel & (p < 0), np.maximum(t0, r), t0)
            t1 = np.where(~parallel & (p > 0), np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


def capture_survives(signal_dbm: float, interferers_dbm) -> bool:
    """True if the frame beats the summed interference by the capture margin."""
    interferers_dbm = np.asarray(interferers_dbm, dtype=float)
    if interferers_dbm.size == 0:
        return True
    total = 10.0 * np.log10(np.sum(10.0 ** (interferers_dbm / 10.0)))
    return signal_dbm - total >= CAPTURE_DB

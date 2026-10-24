import math

import numpy as np
import pytest

from meshsos import radio


def test_sanity_points_from_spec():
    # SF9 / 125 kHz, CR 4/5, 8-symbol preamble, explicit header, CRC on.
    assert radio.time_on_air(50, sf=9) == pytest.approx(0.330, abs=0.005)
    assert radio.time_on_air(20, sf=9) == pytest.approx(0.185, abs=0.005)


def test_exact_semtech_values():
    # Hand-computed from the Semtech formula: preamble 12.25 symbols, payload 68 / 33 symbols.
    t_sym = 2 ** 9 / 125e3
    assert radio.time_on_air(50, sf=9) == pytest.approx((12.25 + 68) * t_sym)
    assert radio.time_on_air(20, sf=9) == pytest.approx((12.25 + 33) * t_sym)


def test_monotonic_in_payload_and_sf():
    for sf in range(7, 13):
        toas = [radio.time_on_air(n, sf) for n in range(0, 256, 5)]
        assert all(b >= a for a, b in zip(toas, toas[1:]))
    for n in (10, 50, 200):
        toas = [radio.time_on_air(n, sf) for sf in range(7, 13)]
        assert all(b > a for a, b in zip(toas, toas[1:]))


def test_low_data_rate_optimize_auto_enabled_for_sf11_and_sf12():
    assert radio.time_on_air(20, 11) == radio.time_on_air(20, 11, low_dr_optimize=True)
    assert radio.time_on_air(20, 10) == radio.time_on_air(20, 10, low_dr_optimize=False)


def test_dwell_limit():
    assert radio.max_payload(9) == 66
    assert radio.max_payload(10) == 24
    assert radio.max_payload(11) == 0          # SF11 at 125 kHz cannot fit even an empty frame
    with pytest.raises(radio.DwellTimeExceeded):
        radio.check_dwell(radio.time_on_air(100, 9))
    radio.check_dwell(radio.time_on_air(66, 9))


def test_sensitivity_matches_datasheet():
    assert radio.sensitivity_dbm(7) == pytest.approx(-124.5, abs=1.5)
    assert radio.sensitivity_dbm(12) == pytest.approx(-137, abs=1.5)


def test_capture_effect():
    assert radio.capture_survives(-100, [])
    assert radio.capture_survives(-100, [-106.5])
    assert not radio.capture_survives(-100, [-104])
    assert not radio.capture_survives(-100, [-109, -109])   # two interferers add up to -106 dBm


def test_path_loss_walls_and_distance():
    m = radio.PathLossModel()
    assert m.loss_db(100, 0, 0) > m.loss_db(10, 0, 0)
    assert m.loss_db(100, 2, 0) == pytest.approx(m.loss_db(100, 0, 0) + 2 * m.intact_db)
    assert m.loss_db(100, 0, 1) > m.loss_db(100, 1, 0)
    assert m.pl0_db == pytest.approx(31.7, abs=0.1)


def test_segment_box_crossings():
    boxes = np.array([[4, -1, 6, 1], [20, 20, 30, 30]])
    p0 = np.array([[0, 0], [0, 5], [5, 0]])
    p1 = np.array([[10, 0], [10, 5], [5, 10]])
    hits = radio.segment_box_crossings(p0, p1, boxes)
    assert hits[:, 0].tolist() == [True, False]
    assert hits[:, 1].tolist() == [False, False]
    assert hits[:, 2].tolist() == [True, False]     # starts inside the box

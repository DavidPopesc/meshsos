"""Discrete-event simulation of the LoRa mesh: MAC, collisions, flooding vs agent routing."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import radio
from .scenario import Scenario

BROADCAST = -1
DATA, ACK, BEACON = 0, 1, 2

# Event codes in the exported timeline.
EV_SEND, EV_HOP, EV_BEACON, EV_ACK, EV_COLLISION, EV_DELIVERED, EV_DEAD, EV_DROP = range(8)
EVENT_NAMES = ["send", "hop", "beacon", "ack", "collision", "delivered", "dead", "drop"]
DROP_REASONS = ["queue full", "queue timeout", "no route", "hop limit", "retries exhausted"]

HEADER_FLOOD = 8     # src, dst, next, prev, packet id (2), hop limit/flags, fragment
HEADER_AGENT = 12    # + routing metric (hops|battery), parent id, last two hops for loop avoidance
ACK_BYTES = 6
BEACON_BYTES = HEADER_AGENT + 2   # channel busy, neighbour count
MAX_HOPS = 15


@dataclass
class Config:
    payload: str = "compressed"      # raw | compressed
    routing: str = "agent"           # flood | agent
    hop_limit_flood: int = 7         # Meshtastic max; the default of 3 cannot cross this map
    hop_limit_agent: int = 12
    beacon_period: float = 90.0
    ack_retries_per_nbr: int = 2
    max_attempts: int = 6
    max_queue: int = 40
    frame_ttl: float = 180.0          # a frame stuck this long in a queue is dropped
    idle_drain: float = 0.004         # % per second (time-compressed so a 15 min run shows hours of drain)
    tx_drain: float = 0.6             # % per second on air
    step_s: float = 10.0              # timeline snapshot interval
    record: bool = True               # keep per-event timeline (off for big sweeps)
    weights: tuple = (0.40, 0.30, 0.15, 0.15)   # link, progress, battery, idle channel

    @property
    def name(self) -> str:
        return f"{self.payload}_{self.routing}"


@dataclass(slots=True)
class Frame:
    kind: int
    src: int
    dst: int
    orig: int
    msg: int
    frag: int
    nfrag: int
    key: int
    hop_limit: int
    size: int
    path: tuple
    prev: int = -1
    meta: tuple | None = None        # (hops, battery, busy, parent) piggybacked in agent mode
    cancelled: bool = False
    attempts: int = 0
    tries_nbr: int = 0
    holds: int = 0
    enq: float = 0.0
    excluded: set = field(default_factory=set)
    grad: int = 99                    # next hop must advertise fewer hops than this (loop freedom)
    side: int = 0                     # sideways (equal-hop) moves used, max 2
    resets: int = 0
    ack_for: int = -1                 # ACK frames: tx id being acknowledged


@dataclass(slots=True)
class Tx:
    id: int
    src: int
    t0: float
    t1: float
    frame: Frame
    rssi: np.ndarray


@dataclass(slots=True)
class Nbr:
    snr: float
    rssi: float
    last: float
    hops: int = 99
    battery: float = 50.0
    busy: float = 0.0
    ack: float = 0.8
    parent: int = -1


class NodeState:
    __slots__ = ("id", "kind", "alive", "battery", "bat_t", "gw", "queue", "transmitting", "mac_pending",
                 "backoff", "awaiting", "ack_token", "seen", "pending", "nbrs", "hops", "last_beacon",
                 "heard", "heard_t", "busy", "intervals", "alive_t", "parent", "ack_due", "last_tx")

    def __init__(self, nid: int, kind: str, battery: float):
        self.id, self.kind = nid, kind
        self.alive = True
        self.battery, self.bat_t = battery, 0.0
        self.gw = kind == "gateway"
        self.queue: deque[Frame] = deque()
        self.transmitting = False
        self.mac_pending = False
        self.backoff = 0
        self.awaiting: tuple[Frame, int] | None = None
        self.ack_token = 0
        self.seen: set[int] = set()
        self.pending: dict[int, Frame] = {}
        self.nbrs: dict[int, Nbr] = {}
        self.hops = 0 if self.gw else 99
        self.last_beacon = -1e9
        self.heard, self.heard_t, self.busy = 0.0, 0.0, 0.0
        self.intervals: list[tuple[float, float]] = []
        self.alive_t = None
        self.parent = -1
        self.ack_due = -1.0
        self.last_tx = -1e9


class Simulator:
    def __init__(self, sc: Scenario, cfg: Config, seed: int | None = None):
        self.sc, self.cfg = sc, cfg
        self.sf = sc.sf
        self.rng = np.random.default_rng(sc.seed * 1000 + 17 if seed is None else seed)
        self.sens = radio.sensitivity_dbm(self.sf)
        self.snr_lim = radio.SNR_LIMIT_DB[self.sf]
        self.t_sym = radio.symbol_time(self.sf)
        self.slot = 2.5 * self.t_sym + 0.01
        self.header = HEADER_AGENT if cfg.routing == "agent" else HEADER_FLOOD
        self.max_phy = radio.max_payload(self.sf)
        self.frag_payload = self.max_phy - self.header
        self.ch = sc.channel
        self.gw = sc.gateway
        self.n = len(sc.nodes)
        self.nodes = [NodeState(nd.id, nd.kind, nd.battery0) for nd in sc.nodes]
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self._txid = itertools.count()
        self.active: dict[int, Tx] = {}
        self.recent: list[Tx] = []
        self.toa_cache: dict[int, float] = {}
        self.collapsed = np.zeros(len(sc.buildings), dtype=bool)
        # metrics
        self.airtime = 0.0
        self.airtime_by_kind = [0.0, 0.0, 0.0]
        self.n_tx = 0
        self.collisions = 0
        self.halfduplex = 0
        self.drops = 0
        self.battery_deaths = 0
        self.delivered: dict[int, dict] = {}
        self.gw_frags: dict[int, set[int]] = {}
        self.events: list[list] = []
        self.steps: list[dict] = []

    # ------------------------------------------------------------ plumbing

    def at(self, t: float, fn, *args) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), fn, args))

    def toa(self, size: int) -> float:
        t = self.toa_cache.get(size)
        if t is None:
            t = radio.check_dwell(radio.time_on_air(size, self.sf))
            self.toa_cache[size] = t
        return t

    def log(self, *ev) -> None:
        if self.cfg.record:
            self.events.append(list(ev))

    def drain(self, node: NodeState) -> None:
        if node.gw or not node.alive:
            node.bat_t = self.now
            return
        node.battery -= self.cfg.idle_drain * (self.now - node.bat_t)
        node.bat_t = self.now
        if node.battery <= 0:
            node.battery = 0.0
            self.battery_deaths += 1
            self.kill(node, "battery")

    def kill(self, node: NodeState, reason: str) -> None:
        if not node.alive:
            return
        node.alive = False
        node.alive_t = self.now
        node.queue.clear()
        node.awaiting = None
        for f in node.pending.values():
            f.cancelled = True
        self.log(round(self.now, 3), EV_DEAD, node.id, -1, -1, 0 if reason == "failure" else 1)

    # ------------------------------------------------------------ run

    def run(self) -> dict:
        sc, cfg = self.sc, self.cfg
        self.ch.update(self.collapsed)
        for t, kind, payload in sc.events:
            self.at(t, self._scenario_event, kind, payload)
        for m in sc.messages:
            self.at(m.t, self._originate, m)
        if cfg.routing == "agent":
            for nd in self.nodes:
                first = self.rng.uniform(0.5, 4.0) if nd.gw else self.rng.uniform(1.0, 30.0)
                self.at(first, self._beacon_tick, nd)
        t = 0.0
        while t <= sc.duration:
            self.at(t, self._snapshot)
            t += cfg.step_s
        while self._heap:
            t, _, fn, args = heapq.heappop(self._heap)
            if t > sc.duration:
                break
            self.now = t
            fn(*args)
        self.now = sc.duration
        result = self._results()
        self.ch.update(np.zeros(len(sc.buildings), dtype=bool))
        return result

    def _scenario_event(self, kind: str, payload) -> None:
        if kind == "collapse":
            self.collapsed[payload] = True
            self.ch.update(self.collapsed)
        elif kind == "fail":
            self.kill(self.nodes[payload], "failure")

    # ------------------------------------------------------------ origination

    def _originate(self, m) -> None:
        node = self.nodes[m.node]
        self.drain(node)
        if not node.alive:
            return
        if self.cfg.payload == "compressed":
            sizes = [self.header + len(m.packet)]
        else:
            raw = m.raw_bytes
            nfrag = math.ceil(raw / self.frag_payload)
            sizes = [self.header + min(self.frag_payload, raw - i * self.frag_payload) for i in range(nfrag)]
        for i, size in enumerate(sizes):
            key = m.id * 16 + i
            f = Frame(DATA, node.id, BROADCAST, node.id, m.id, i, len(sizes), key,
                      self.cfg.hop_limit_flood if self.cfg.routing == "flood" else self.cfg.hop_limit_agent,
                      size, (node.id,), enq=self.now)
            node.seen.add(key)
            if self.cfg.routing == "flood":
                self.enqueue(node, f)
            else:
                self._route(node, f)

    # ------------------------------------------------------------ MAC

    def enqueue(self, node: NodeState, f: Frame, front: bool = False) -> None:
        if not node.alive:
            return
        if len(node.queue) >= self.cfg.max_queue:
            self.drops += 1
            self.log(round(self.now, 3), EV_DROP, node.id, -1, f.msg, 0)
            return
        f.enq = self.now
        (node.queue.appendleft if front else node.queue.append)(f)
        self.kick(node)

    def kick(self, node: NodeState, delay: float = 0.0) -> None:
        if node.alive and not node.mac_pending and not node.transmitting:
            node.mac_pending = True
            self.at(self.now + delay + self.rng.uniform(0, 0.005), self._mac, node)

    def channel_busy(self, node: NodeState) -> bool:
        thresh = self.sens - 3.0    # CAD detects preambles slightly below data sensitivity
        for q in self.active.values():
            if q.src != node.id and q.rssi[node.id] >= thresh:
                return True
        return False

    def _mac(self, node: NodeState) -> None:
        node.mac_pending = False
        if not node.alive or node.transmitting or node.awaiting is not None:
            return
        if self.now < node.ack_due:          # an ACK we owe goes out first
            node.mac_pending = True
            self.at(node.ack_due + 0.002, self._mac, node)
            return
        while node.queue and (node.queue[0].cancelled or self.now - node.queue[0].enq > self.cfg.frame_ttl):
            f = node.queue.popleft()
            if not f.cancelled:
                self.drops += 1
                self.log(round(self.now, 3), EV_DROP, node.id, -1, f.msg, 1)
        if not node.queue:
            return
        if self.channel_busy(node):
            node.backoff += 1
            window = min(2.5, self.slot * (2 ** min(node.backoff + 2, 7)))
            node.mac_pending = True
            self.at(self.now + self.rng.uniform(self.slot, window), self._mac, node)
            return
        node.backoff = 0
        self.start_tx(node, node.queue.popleft())

    def start_tx(self, node: NodeState, f: Frame) -> None:
        self.drain(node)
        if not node.alive:
            return
        f.src = node.id
        if self.cfg.routing == "agent":
            f.meta = (node.hops, node.battery, node.busy, node.parent)
        dur = self.toa(f.size)
        rssi = self.ch.rssi[node.id] + self.rng.normal(0.0, self.ch.model.fast_fade_sigma_db, self.n)
        rssi[node.id] = -np.inf
        tx = Tx(next(self._txid), node.id, self.now, self.now + dur, f, rssi)
        self.active[tx.id] = tx
        self.recent.append(tx)
        node.transmitting = True
        if f.kind != BEACON:
            node.last_tx = self.now
        node.intervals.append((tx.t0, tx.t1))
        for j in np.nonzero(rssi >= self.sens)[0]:
            other = self.nodes[j]
            if other.alive:
                other.intervals.append((tx.t0, tx.t1))
                other.heard += dur
        self.airtime += dur
        self.airtime_by_kind[f.kind] += dur
        self.n_tx += 1
        if not node.gw:
            node.battery -= self.cfg.tx_drain * dur
        if f.kind == DATA:
            ev = EV_SEND if f.src == f.orig and len(f.path) == 1 else EV_HOP
        else:
            ev = EV_ACK if f.kind == ACK else EV_BEACON
        if f.kind != BEACON or self.cfg.record:
            self.log(round(tx.t0, 3), ev, node.id, f.dst, f.msg, round(dur, 4))
        self.at(tx.t1, self._end_tx, tx)

    def _end_tx(self, tx: Tx) -> None:
        del self.active[tx.id]
        src = self.nodes[tx.src]
        src.transmitting = False
        f = tx.frame
        # Drop history no longer able to overlap anything still on air.
        horizon = min((q.t0 for q in self.active.values()), default=self.now) - 0.5
        if len(self.recent) > 64:
            self.recent = [q for q in self.recent if q.t1 >= horizon]
        overl = [q for q in self.recent if q is not tx and q.t0 < tx.t1 and q.t1 > tx.t0]
        interf = np.zeros(self.n)
        talkers = set()
        for q in overl:
            interf += 10.0 ** (q.rssi / 10.0)
            talkers.add(q.src)
        cand = np.nonzero(tx.rssi >= self.sens)[0]
        for r in cand:
            rx = self.nodes[r]
            if not rx.alive:
                continue
            intended = f.dst == BROADCAST or f.dst == r
            if r in talkers:          # half duplex: it was transmitting itself
                if intended:
                    self.halfduplex += 1
                continue
            if interf[r] > 0 and tx.rssi[r] - 10.0 * math.log10(interf[r]) < radio.CAPTURE_DB:
                if intended:
                    self.collisions += 1
                    if f.kind == DATA:
                        self.log(round(tx.t1, 3), EV_COLLISION, tx.src, int(r), f.msg, 0)
                continue
            self._receive(rx, f, float(tx.rssi[r]), tx)
        if f.kind == DATA and f.dst != BROADCAST and src.alive:
            src.ack_token += 1
            src.awaiting = (f, src.ack_token)
            timeout = 0.01 + self.toa(ACK_BYTES) + 0.06
            self.at(self.now + timeout, self._ack_timeout, src, src.ack_token)
        else:
            self.kick(src)

    # ------------------------------------------------------------ receive path

    def _receive(self, rx: NodeState, f: Frame, rssi: float, tx: Tx) -> None:
        snr = rssi - radio.NOISE_FLOOR_DBM
        if self.cfg.routing == "agent":
            self._observe(rx, f, rssi, snr)
        if f.kind == BEACON:
            return
        if f.kind == ACK:
            if rx.awaiting is not None and f.dst == rx.id and f.ack_for == rx.awaiting[0].key:
                frame, _ = rx.awaiting
                nb = rx.nbrs.get(f.src)
                if nb:
                    nb.ack = 0.7 * nb.ack + 0.3
                rx.awaiting = None
                rx.ack_token += 1
                self.kick(rx)
            return
        if self.cfg.routing == "flood":
            self._flood_rx(rx, f, snr)
        else:
            self._agent_rx(rx, f)

    def _deliver(self, f: Frame) -> None:
        got = self.gw_frags.setdefault(f.msg, set())
        got.add(f.frag)
        if len(got) == f.nfrag and f.msg not in self.delivered:
            m = self.sc.messages[f.msg]
            lat = self.now - m.t
            self.delivered[f.msg] = {"t": round(self.now, 3), "latency": round(lat, 3),
                                     "path": list(f.path) + [self.gw]}
            self.log(round(self.now, 3), EV_DELIVERED, f.orig, self.gw, f.msg, round(lat, 3))

    # Meshtastic-style managed flooding --------------------------------

    def _flood_rx(self, rx: NodeState, f: Frame, snr: float) -> None:
        if f.key in rx.seen:
            p = rx.pending.pop(f.key, None)
            if p is not None:
                p.cancelled = True        # someone else already rebroadcast it
            return
        rx.seen.add(f.key)
        if rx.gw:
            self._deliver(f)
            return
        if f.hop_limit <= 0:
            return
        # Contention window grows with SNR so distant (weak) receivers rebroadcast first.
        frac = min(max((snr + 20.0) / 30.0, 0.0), 1.0)
        cw = 3 + round(frac * 5)
        delay = self.rng.integers(0, 2 ** cw) * self.slot
        copy = Frame(DATA, rx.id, BROADCAST, f.orig, f.msg, f.frag, f.nfrag, f.key, f.hop_limit - 1,
                     f.size, f.path + (rx.id,), prev=f.src)
        rx.pending[f.key] = copy
        self.at(self.now + delay, self._flood_fire, rx, copy)

    def _flood_fire(self, rx: NodeState, f: Frame) -> None:
        if rx.pending.get(f.key) is f:
            del rx.pending[f.key]
        if not f.cancelled and rx.alive:
            self.enqueue(rx, f)

    # On-node agent routing --------------------------------------------

    def _observe(self, rx: NodeState, f: Frame, rssi: float, snr: float) -> None:
        nb = rx.nbrs.get(f.src)
        if nb is None:
            nb = rx.nbrs[f.src] = Nbr(snr=snr, rssi=rssi, last=self.now)
        else:
            nb.snr = 0.75 * nb.snr + 0.25 * snr
            nb.rssi = 0.75 * nb.rssi + 0.25 * rssi
            nb.last = self.now
        if f.meta is not None:
            nb.hops, nb.battery, nb.busy, nb.parent = f.meta
            if f.kind == BEACON:
                nb.ack = min(1.0, nb.ack + 0.1)   # give links that failed earlier another chance
        if not rx.gw:
            old = rx.hops
            rx.hops = self._hops(rx)
            if rx.hops < old and self.now - rx.last_beacon > 8.0:
                self.at(self.now + self.rng.uniform(0.3, 3.0), self._beacon, rx, False)

    def _fresh(self, nb: Nbr) -> bool:
        return self.now - nb.last < 3.2 * self.cfg.beacon_period

    def _hops(self, node: NodeState) -> int:
        """Distance-vector hop count with split horizon (ignore neighbours routing through us)."""
        best, parent = 99, -1
        for nid, nb in node.nbrs.items():
            if self._fresh(nb) and nb.snr >= self.snr_lim + 3.0 and nb.ack > 0.25 and nb.parent != node.id:
                if nb.hops < best:
                    best, parent = nb.hops, nid
        node.parent = parent
        return best + 1 if best + 1 <= MAX_HOPS else 99

    def score(self, node: NodeState, nid: int, nb: Nbr) -> float:
        w_link, w_prog, w_bat, w_idle = self.cfg.weights
        link = min(max((nb.snr - self.snr_lim) / 20.0, 0.0), 1.0) * nb.ack
        if nb.hops == 0:
            prog = 1.2
        else:
            prog = min(max((node.hops - nb.hops + 1) / 2.0, 0.0), 1.0)
        return w_link * link + w_prog * prog + w_bat * (nb.battery / 100.0) + w_idle * (1.0 - nb.busy)

    def ranked(self, node: NodeState, exclude=(), below: int = 99) -> list[tuple[int, float]]:
        """Usable neighbours advertising fewer than ``below`` hops, best score first."""
        out = []
        for nid, nb in node.nbrs.items():
            if nid in exclude or not self._fresh(nb) or nb.hops >= below or nb.ack < 0.2:
                continue
            if nb.snr < self.snr_lim + 1.0:
                continue
            out.append((nid, self.score(node, nid, nb)))
        out.sort(key=lambda p: -p[1])
        return out

    def _route(self, node: NodeState, f: Frame) -> None:
        if not node.alive:
            return
        exclude = set(f.excluded)
        exclude.update(f.path[-3:])          # prev + the two hops carried in the header
        exclude.add(node.id)
        limit = min(f.grad, node.hops) if node.hops < 99 else f.grad
        cands = self.ranked(node, exclude, below=limit)
        sideways = False
        if not cands and f.side < 2:
            cands = self.ranked(node, exclude, below=limit + 1)
            sideways = bool(cands)
        if not cands:
            f.holds += 1
            if f.holds == 4 and f.resets == 0 and node.hops < 99:
                f.resets, f.grad = 1, node.hops + 1   # re-anchor once on our refreshed view
            if f.holds > 10:
                self.drops += 1
                self.log(round(self.now, 3), EV_DROP, node.id, -1, f.msg, 2)
            else:                               # store and forward until a route appears
                f.excluded.clear()
                self.at(self.now + 6.0 + self.rng.uniform(0, 3), self._route, node, f)
            return
        f.dst = cands[0][0]
        f.side += sideways
        f.tries_nbr = 0
        self.enqueue(node, f)

    def _agent_rx(self, rx: NodeState, f: Frame) -> None:
        if f.dst != rx.id:
            return
        rx.ack_due = self.now + 0.01 + self.toa(ACK_BYTES)
        self.at(self.now + 0.01, self._send_ack, rx, f)
        if rx.gw:
            self._deliver(f)
            return
        first_hop = rx.pending.get(f.key)
        if first_hop is not None and first_hop.prev == f.src:
            return                      # retransmission because our ACK was lost
        if f.hop_limit <= 1:
            self.drops += 1
            self.log(round(self.now, 3), EV_DROP, rx.id, -1, f.msg, 3)
            return
        copy = Frame(DATA, rx.id, -1, f.orig, f.msg, f.frag, f.nfrag, f.key, f.hop_limit - 1,
                     f.size, f.path + (rx.id,), prev=f.src, side=f.side, resets=f.resets)
        sender_view = self.nodes[f.src].nbrs.get(rx.id)
        copy.grad = sender_view.hops if sender_view else f.grad
        if first_hop is not None:       # the frame looped back: route around where it went last time
            copy.excluded.update(first_hop.excluded)
            copy.excluded.add(first_hop.dst)
        rx.pending[f.key] = copy
        self._route(rx, copy)

    def _send_ack(self, rx: NodeState, f: Frame) -> None:
        if not rx.alive or rx.transmitting:
            return
        ack = Frame(ACK, rx.id, f.src, rx.id, f.msg, f.frag, f.nfrag, -1, 0, ACK_BYTES, (rx.id,), ack_for=f.key)
        self.start_tx(rx, ack)

    def _ack_timeout(self, node: NodeState, token: int) -> None:
        if node.awaiting is None or node.ack_token != token or not node.alive:
            return
        f, _ = node.awaiting
        node.awaiting = None
        nb = node.nbrs.get(f.dst)
        if nb:
            nb.ack = 0.7 * nb.ack
        f.attempts += 1
        f.tries_nbr += 1
        if f.attempts >= self.cfg.max_attempts:
            self.drops += 1
            self.log(round(self.now, 3), EV_DROP, node.id, f.dst, f.msg, 4)
            self.kick(node)
        elif f.tries_nbr >= self.cfg.ack_retries_per_nbr:
            f.excluded.add(f.dst)
            node.hops = self._hops(node)
            self._route(node, f)
            self.kick(node)
        else:
            node.queue.appendleft(f)
            self.kick(node, delay=self.rng.uniform(0.05, min(4.0, 0.3 * 2 ** f.attempts)))

    def _beacon_tick(self, node: NodeState) -> None:
        if not node.alive:
            return
        # Every header already carries our metric; only beacon when we've been quiet.
        if node.gw or self.now - node.last_tx > 0.5 * self.cfg.beacon_period:
            self._beacon(node, True)
        period = self.cfg.beacon_period * (0.4 if node.gw else 1.0)
        self.at(self.now + period * self.rng.uniform(0.85, 1.15), self._beacon_tick, node)

    def _beacon(self, node: NodeState, periodic: bool) -> None:
        if not node.alive or (not periodic and self.now - node.last_beacon < 8.0):
            return
        window = max(self.now - node.heard_t, 1.0)
        node.busy = 0.6 * node.busy + 0.4 * min(node.heard / window, 1.0)
        node.heard, node.heard_t = 0.0, self.now
        if not node.gw:
            node.hops = self._hops(node)
        node.last_beacon = self.now
        f = Frame(BEACON, node.id, BROADCAST, node.id, -1, 0, 1, -1, 0, BEACON_BYTES, (node.id,))
        self.enqueue(node, f, front=True)

    # ------------------------------------------------------------ outputs

    def _snapshot(self) -> None:
        for nd in self.nodes:
            self.drain(nd)
        if not self.cfg.record:
            return
        alive = np.array([nd.alive for nd in self.nodes])
        rssi = self.ch.rssi
        nbrs = []
        for i in range(self.n):
            if not alive[i]:
                nbrs.append([])
                continue
            js = np.nonzero((rssi[i] >= self.sens) & alive)[0]
            nbrs.append([[int(j), round(float(rssi[i, j]), 1)] for j in js])
        step = {
            "t": round(self.now, 2),
            "alive": [int(a) for a in alive],
            "battery": [round(nd.battery, 1) for nd in self.nodes],
            "nbrs": nbrs,
        }
        if self.cfg.routing == "agent":
            step["hops"] = [nd.hops for nd in self.nodes]
            nh, scores = [], []
            for nd in self.nodes:
                r = self.ranked(nd, {nd.id}, below=nd.hops + 1) if nd.alive and not nd.gw else []
                nh.append(r[0][0] if r else -1)
                scores.append([[j, round(s, 3)] for j, s in r[:5]])
            step["nh"] = nh
            step["scores"] = scores
        self.steps.append(step)

    def _results(self) -> dict:
        sc = self.sc
        n_msgs = len(sc.messages)
        lats = np.array([d["latency"] for d in self.delivered.values()]) if self.delivered else np.array([])
        busy, alive_time = 0.0, 0.0
        for nd in self.nodes:
            if nd.gw:
                continue
            end = nd.alive_t if nd.alive_t is not None else sc.duration
            alive_time += end
            busy += _union_length(nd.intervals, end)
        metrics = {
            "messages": n_msgs,
            "delivered": len(self.delivered),
            "delivery_ratio": len(self.delivered) / n_msgs if n_msgs else 0.0,
            "latency_avg_s": float(lats.mean()) if lats.size else float("nan"),
            "latency_p95_s": float(np.percentile(lats, 95)) if lats.size else float("nan"),
            "airtime_s": self.airtime,
            "airtime_data_s": self.airtime_by_kind[DATA],
            "airtime_control_s": self.airtime_by_kind[ACK] + self.airtime_by_kind[BEACON],
            "transmissions": self.n_tx,
            "collisions": self.collisions,
            "halfduplex_misses": self.halfduplex,
            "drops": self.drops,
            "battery_deaths": self.battery_deaths,
            "channel_utilization": busy / alive_time if alive_time else 0.0,
        }
        return {"name": self.cfg.name, "payload": self.cfg.payload, "routing": self.cfg.routing,
                "metrics": metrics, "delivered": self.delivered, "events": self.events, "steps": self.steps}


def _union_length(intervals: list[tuple[float, float]], end: float) -> float:
    total, cur0, cur1 = 0.0, None, None
    for a, b in sorted(intervals):
        if a >= end:
            break
        b = min(b, end)
        if cur1 is None or a > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        total += cur1 - cur0
    return total


VARIANTS = [("raw", "flood"), ("raw", "agent"), ("compressed", "flood"), ("compressed", "agent")]


def run_all(sc: Scenario, record: bool = True, **cfg_kw) -> dict[str, dict]:
    out = {}
    for payload, routing in VARIANTS:
        cfg = Config(payload=payload, routing=routing, record=record, **cfg_kw)
        out[cfg.name] = Simulator(sc, cfg).run()
    return out

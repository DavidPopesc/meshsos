"""Earthquake scenario: node layout, building collapses, node failures and SOS bursts.

Everything is drawn from one seeded generator so a (scenario, nodes, seed)
triple always produces the same disaster, and all four protocol variants
are scored against exactly the same events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codec, geo, radio
from .channel import Channel
from .codec import Injury, MsgType, Need, Hazard


@dataclass
class Node:
    id: int
    kind: str              # t3 | tdeck | gateway
    x: float
    y: float
    battery0: float        # starting charge, percent
    building: int = -1     # building the node is inside, or -1 outdoors


@dataclass
class Message:
    id: int                # index in the scenario
    node: int
    t: float
    text: str
    truth: dict
    packet: bytes = b""
    fields: dict = field(default_factory=dict)

    @property
    def raw_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass
class Scenario:
    name: str
    seed: int
    sf: int
    duration: float
    quake_t: float
    nodes: list[Node]
    buildings: list[geo.Building]
    events: list[tuple[float, str, object]]       # (t, "collapse"|"fail", payload)
    shocks: list[dict]
    messages: list[Message]
    channel: Channel

    @property
    def gateway(self) -> int:
        return next(n.id for n in self.nodes if n.kind == "gateway")


# ---------------------------------------------------------------- SOS corpus

_N_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]

_INJURIES = [
    ("one bleeding badly", Injury.BLEEDING, 3),
    ("my brother is bleeding from his head", Injury.BLEEDING, 2),
    ("broken leg", Injury.FRACTURE, 2),
    ("i think her arm is broken", Injury.FRACTURE, 2),
    ("grandma is unconscious", Injury.UNCONSCIOUS, 3),
    ("burns on his arms", Injury.BURN, 2),
    ("he cant breathe well", Injury.BREATHING, 3),
    ("my leg is crushed", Injury.CRUSH, 3),
    ("dad has chest pain", Injury.CARDIAC, 3),
    ("my wife is pregnant", Injury.PREGNANT, 2),
]
_NEEDS = [
    ("need water", Need.WATER), ("need rescue", Need.RESCUE), ("send an ambulance", Need.MEDIC),
    ("need food", Need.FOOD), ("need blankets", Need.SHELTER), ("need insulin", Need.MEDS),
    ("someone in a wheelchair needs to get out", Need.EVAC), ("need a generator for oxygen", Need.POWER | Need.MEDS),
]
_UNKNOWN_PLACES = ["3rd and Bay", "our building", "the yellow house", "here", "the apartment by the corner store"]
_TYPOS = {"trapped": "trappd", "bleeding": "bleding", "unconscious": "unconcious", "rescue": "resuce", "water": "watr"}


def _place(rng, node_xy) -> tuple[str, tuple[float, float], int]:
    """Location phrase, true position, loc_src."""
    r = rng.random()
    if r < 0.18:
        name = rng.choice(["Palace of Fine Arts", "Marina Green", "the Safeway", "Marina Middle School", "Moscone Rec"])
        key = name.lower().replace("the ", "")
        return name, geo.LANDMARKS[key], 1
    if r < 0.80:
        a = rng.choice(list(geo.NS_X))
        b = rng.choice([s for s in geo.EW_Y if s != "Greenwich"])
        # Survivors usually describe the corner nearest them.
        if rng.random() < 0.7:
            a = min(geo.NS_X, key=lambda k: abs(geo.NS_X[k] - node_xy[0]))
            b = min((s for s in geo.EW_Y if s != "Greenwich"), key=lambda k: abs(geo.EW_Y[k] - node_xy[1]))
        b_txt = "Marina Blvd" if b == "Marina" else b
        phrase = rng.choice([f"{a} and {b_txt}", f"{b_txt} & {a}", f"{a} st and {b_txt}", f"corner of {a} and {b_txt}"])
        return phrase, (geo.NS_X[a], geo.EW_Y[b]), 1
    return str(rng.choice(_UNKNOWN_PLACES)), node_xy, 0


def _people(rng) -> tuple[str, int, int]:
    """People phrase, total, kids."""
    r = rng.random()
    if r < 0.22:
        n = int(rng.integers(2, 7))
        word = _N_WORDS[n] if rng.random() < 0.5 else str(n)
        return f"{word} of us", n, 0
    if r < 0.40:
        k = int(rng.integers(1, 4))
        word = _N_WORDS[k] if rng.random() < 0.5 else str(k)
        return f"{word} kid{'s' if k > 1 else ''}", k + 1, k
    if r < 0.52:
        n = int(rng.integers(3, 6))
        return f"family of {n}", n, 0
    if r < 0.62:
        return "me and my wife", 2, 0
    if r < 0.70:
        return "alone", 1, 0
    if r < 0.80:
        n = int(rng.integers(2, 9))
        return f"{n} people", n, 0
    return "", 1, 0


def make_sos(rng: np.random.Generator, node_xy, trapped: bool = False) -> tuple[str, dict]:
    """A free-text SOS a survivor might type, with ground-truth fields."""
    weights = {"trapped": 0.22, "collapse": 0.08, "medical": 0.2, "fire": 0.06, "gas": 0.08,
               "flood": 0.05, "supply": 0.14, "checkin": 0.1, "missing": 0.04, "hazard": 0.03}
    if trapped:
        weights.update(trapped=0.6, collapse=0.2)
    kinds, w = zip(*weights.items())
    kind = str(rng.choice(kinds, p=np.array(w) / sum(w)))
    place, (tx, ty), loc_src = _place(rng, node_xy)
    ppl, total, kids = _people(rng)
    truth = {"type": 0, "people": total, "kids": kids, "injuries": 0, "needs": 0, "hazards": 0,
             "loc_src": loc_src, "cell": geo.cell_of(tx, ty)}

    def injury():
        txt, flag, _ = _INJURIES[int(rng.integers(len(_INJURIES)))]
        truth["injuries"] |= int(flag)
        truth["needs"] |= int(Need.MEDIC)
        return txt

    def need():
        txt, flag = _NEEDS[int(rng.integers(len(_NEEDS)))]
        truth["needs"] |= int(flag)
        return txt

    parts: list[str]
    if kind == "trapped":
        truth["type"] = MsgType.TRAPPED
        truth["needs"] |= Need.RESCUE
        lead = rng.choice(["trapped under debris at", "we are trapped in the rubble at", "stuck under the stairs at", "pinned under a wall at"])
        if "pinned" in lead:
            truth["injuries"] |= int(Injury.CRUSH)
            truth["needs"] |= int(Need.MEDIC)
        parts = [f"{lead} {place}"] + ([ppl] if ppl else [])
        if rng.random() < 0.6:
            parts.append(injury())
        parts.append(need())
    elif kind == "collapse":
        truth["type"] = MsgType.COLLAPSE
        truth["needs"] |= Need.RESCUE
        parts = [f"building collapsed at {place}", (ppl + " inside") if ppl else "people inside", "please send rescue"]
        if not ppl:
            truth["people"] = 1
        if rng.random() < 0.4:
            parts.insert(2, injury())
    elif kind == "medical":
        truth["type"] = MsgType.MEDICAL
        parts = [injury(), f"at {place}"] + ([ppl] if ppl else [])
        if rng.random() < 0.5:
            parts.append(need())
    elif kind == "fire":
        truth["type"] = MsgType.FIRE
        truth["hazards"] |= Hazard.FIRE
        parts = [f"fire in the building at {place}"] + ([ppl] if ppl else []) + ["send firefighters"]
        truth["needs"] |= Need.RESCUE
    elif kind == "gas":
        truth["type"] = MsgType.GAS_LEAK
        truth["hazards"] |= Hazard.GAS
        parts = [f"strong gas smell at {place}"] + ([ppl] if ppl else []) + ["we went outside"]
    elif kind == "flood":
        truth["type"] = MsgType.FLOOD
        truth["hazards"] |= Hazard.WATER
        parts = [f"water main broke at {place}, street flooding"] + ([ppl] if ppl else [])
    elif kind == "supply":
        truth["type"] = MsgType.SUPPLY
        parts = [f"at {place}"] + ([ppl] if ppl else []) + ["out of water"]
        truth["needs"] |= Need.WATER
        if rng.random() < 0.5:
            parts.append(need())
    elif kind == "checkin":
        truth["type"] = MsgType.CHECKIN
        parts = [rng.choice(["we are ok", "all safe", "checking in, we are safe"]), f"at {place}"] + ([ppl] if ppl else [])
    elif kind == "missing":
        truth["type"] = MsgType.MISSING
        parts = [f"looking for my mom, last seen at {place}"]
        truth["people"], truth["kids"] = 1, 0
    else:
        truth["type"] = MsgType.HAZARD
        truth["hazards"] |= Hazard.POWERLINE
        parts = [f"power lines down at {place}, sparking"] + ([ppl] if ppl else [])

    text = ", ".join(parts)
    if rng.random() < 0.12:                      # a typo the keyword rules may miss
        for good, bad in _TYPOS.items():
            if good in text:
                text = text.replace(good, bad, 1)
                break
    if rng.random() < 0.3:
        text = text.upper() if rng.random() < 0.3 else text.capitalize()
    if rng.random() < 0.25:
        text += rng.choice([" please hurry", " HELP", "!!", " pls"])
    truth["type"] = int(truth["type"])
    return text, truth


def field_accuracy(f: codec.SosFields, truth: dict) -> dict[str, bool]:
    return {
        "type": f.type == truth["type"],
        "people": f.people == truth["people"],
        "kids": f.kids == truth["kids"],
        "injuries": f.injuries == truth["injuries"],
        "needs": (f.needs & truth["needs"]) == truth["needs"],
        "hazards": f.hazards == truth["hazards"],
        "location": abs(f.cell_x - truth["cell"][0]) <= 1 and abs(f.cell_y - truth["cell"][1]) <= 1,
    }


# ---------------------------------------------------------------- the quake

def _place_nodes(rng, n_nodes: int, buildings: list[geo.Building]) -> list[Node]:
    nodes: list[Node] = []
    gx, gy = (geo.NS_X["Fillmore"] + geo.NS_X["Webster"]) / 2, geo.EW_Y["Marina"] + 45
    nodes.append(Node(0, "gateway", gx, gy, 100.0))             # EOC staging tent on the Marina Green
    n_tdeck = max(2, n_nodes // 16)
    x_lo, x_hi = geo.NS_X["Lyon"] - 10, geo.NS_X["Laguna"] + 10
    y_lo, y_hi = geo.EW_Y["Greenwich"], geo.EW_Y["Marina"] + 10
    res = [b for b in buildings if b.kind != "landmark"]
    for i in range(1, n_nodes):
        kind = "tdeck" if i <= n_tdeck else "t3"
        if kind == "t3" and rng.random() < 0.45:
            b = res[int(rng.integers(len(res)))]
            x = b.x + rng.uniform(-b.w / 3, b.w / 3)
            y = b.y + rng.uniform(-b.d / 3, b.d / 3)
            bid = b.id
        else:
            while True:
                x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
                if geo.in_zone(x, y, ("water",)) is None:
                    break
            bid = -1
        battery = float(rng.uniform(3, 12)) if rng.random() < 0.12 else float(rng.uniform(25, 100))
        if kind == "tdeck":
            battery = float(rng.uniform(70, 100))
        nodes.append(Node(i, kind, float(x), float(y), battery, bid))
    return nodes


def _connected(ch: Channel, sf: int, gw: int) -> np.ndarray:
    nbrs = ch.neighbors(sf, margin_db=6.0)
    seen = np.zeros(ch.n, dtype=bool)
    stack = [gw]
    seen[gw] = True
    while stack:
        a = stack.pop()
        for b in nbrs[a]:
            if not seen[b]:
                seen[b] = True
                stack.append(b)
    return seen


def make_quake(n_nodes: int = 50, seed: int = 1, sf: int = 9, duration: float = 900.0) -> Scenario:
    if not 7 <= sf <= 10:
        raise radio.DwellTimeExceeded(f"SF{sf} cannot fit any frame in the 400 ms US915 dwell limit at 125 kHz")
    rng = np.random.default_rng(seed)
    buildings = geo.build_city(rng)
    boxes = np.array([b.box for b in buildings])
    nodes = _place_nodes(rng, n_nodes, buildings)

    # Re-roll nodes stranded out of radio range at t=0: real kits get carried to where they work.
    for _ in range(30):
        ch = Channel([(n.x, n.y) for n in nodes], [n.kind for n in nodes], boxes, np.random.default_rng(seed + 7))
        ok = _connected(ch, sf, 0)
        if ok.all():
            break
        for n in nodes:
            if not ok[n.id]:
                anchor = nodes[int(rng.choice(np.nonzero(ok)[0]))]
                n.x = float(np.clip(anchor.x + rng.normal(0, 90), 0, geo.WIDTH_M))
                n.y = float(np.clip(anchor.y + rng.normal(0, 70), 0, geo.SHORELINE_Y - 5))
                n.building = -1
    ch = Channel([(n.x, n.y) for n in nodes], [n.kind for n in nodes], boxes, np.random.default_rng(seed + 7))

    quake_t = 120.0
    shocks = [{"t": quake_t, "kind": "mainshock", "mag": 6.9}]
    t_a = quake_t + float(rng.uniform(170, 260))
    shocks.append({"t": t_a, "kind": "aftershock", "mag": round(float(rng.uniform(5.2, 5.8)), 1)})
    t_b = t_a + float(rng.uniform(200, 300))
    if t_b < duration - 120:
        shocks.append({"t": t_b, "kind": "aftershock", "mag": round(float(rng.uniform(4.8, 5.4)), 1)})

    events: list[tuple[float, str, object]] = []
    collapsed: set[int] = set()
    failed: dict[int, float] = {}
    trapped_since: dict[int, float] = {}
    for s in shocks:
        main = s["kind"] == "mainshock"
        for b in buildings:
            if b.id in collapsed or b.kind == "landmark":
                continue
            p = (0.20 if b.on_fill else 0.07) if main else (0.05 if b.on_fill else 0.015)
            if rng.random() < p:
                tc = s["t"] + float(rng.exponential(6.0))
                collapsed.add(b.id)
                events.append((tc, "collapse", b.id))
                for n in nodes:
                    if n.building == b.id and n.kind != "gateway" and n.id not in failed:
                        if rng.random() < 0.25:
                            failed[n.id] = tc
                            events.append((tc + 0.5, "fail", n.id))
                        else:
                            trapped_since.setdefault(n.id, tc)
        for n in nodes:
            if n.kind != "gateway" and n.id not in failed and rng.random() < (0.06 if main else 0.03):
                tf = s["t"] + float(rng.uniform(0, 20))
                failed[n.id] = tf
                events.append((tf, "fail", n.id))
    for n in nodes:                                        # background attrition: water damage, fires
        if n.kind != "gateway" and n.id not in failed:
            tf = quake_t + float(rng.exponential(5000.0))
            if tf < duration:
                failed[n.id] = tf
                events.append((tf, "fail", n.id))
    events.sort(key=lambda e: e[0])

    # SOS traffic: a burst after each shock plus a trickle of follow-ups.
    times: list[tuple[float, int]] = []
    for n in nodes:
        if n.kind == "gateway":
            continue
        for s in shocks:
            p = 0.75 if s["kind"] == "mainshock" else 0.4
            if rng.random() < p:
                times.append((s["t"] + 4.0 + float(rng.gamma(2.0, 22.0)), n.id))
        t = quake_t
        while True:
            t += float(rng.exponential(420.0))
            if t > duration - 60:
                break
            times.append((t, n.id))
    times = sorted(tt for tt in times if tt[0] < duration - 60 and failed.get(tt[1], 1e9) > tt[0])

    messages: list[Message] = []
    used_ids: set[int] = set()
    for t, nid in times:
        node = nodes[nid]
        trapped = nid in trapped_since and trapped_since[nid] <= t
        if node.kind == "tdeck":
            text, truth = make_sos(rng, (node.x, node.y))
        else:
            text, truth = make_sos(rng, (node.x, node.y), trapped=trapped)
        mid = int(rng.integers(1, 0xFFFF))
        while mid in used_ids:
            mid = int(rng.integers(1, 0xFFFF))
        used_ids.add(mid)
        minute = int(max(0.0, t - quake_t) // 60)
        f = codec.encode_fields(text, (node.x, node.y), minute=minute, msg_id=mid)
        m = Message(len(messages), nid, round(t, 3), text, truth, codec.pack(f), f.to_dict())
        m.fields["place"] = f.place
        messages.append(m)

    return Scenario("quake", seed, sf, duration, quake_t, nodes, buildings, events, shocks, messages, ch)

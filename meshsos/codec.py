"""On-node SOS compressor: free text -> 10-byte bit-packed packet, and back.

The encoder is deliberately rule/keyword based (regexes and small lookup
tables) so the same logic fits in a few KB of flash on an ESP32. The C++
twin lives in ``firmware/meshsos_codec.h`` and produces identical bytes.

Packet layout, MSB first (75 bits used, padded to 10 bytes):

    version   2   format version (currently 1)
    type      4   MsgType
    severity  2   0 info .. 3 life-threatening
    people    5   people at the location (31 = 31 or more)
    kids      3   of which children (7 = 7 or more)
    injuries  8   Injury bit flags
    needs     8   Need bit flags
    hazards   4   Hazard bit flags
    floor     3   0 unknown, 1 ground/basement, 2..7 storey
    loc_src   1   1 = place named in the text, 0 = node GPS fix
    cell_x    6   64 x 32 grid over the map (~30 x 42 m cells)
    cell_y    5
    minute    8   minutes since the quake, mod 256
    msg_id   16   short message ID
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import IntEnum, IntFlag
from typing import Callable

from . import geo

VERSION = 1
PACKET_BYTES = 10
LAYOUT = [
    ("version", 2), ("type", 4), ("severity", 2), ("people", 5), ("kids", 3),
    ("injuries", 8), ("needs", 8), ("hazards", 4), ("floor", 3), ("loc_src", 1),
    ("cell_x", 6), ("cell_y", 5), ("minute", 8), ("msg_id", 16),
]
PAYLOAD_BITS = sum(n for _, n in LAYOUT)


class MsgType(IntEnum):
    CHECKIN = 0
    TRAPPED = 1
    MEDICAL = 2
    FIRE = 3
    GAS_LEAK = 4
    COLLAPSE = 5
    FLOOD = 6
    SUPPLY = 7
    MISSING = 8
    HAZARD = 9
    EVAC = 10
    OTHER = 15


class Injury(IntFlag):
    BLEEDING = 1
    FRACTURE = 2
    UNCONSCIOUS = 4
    BURN = 8
    BREATHING = 16
    CRUSH = 32
    CARDIAC = 64
    PREGNANT = 128


class Need(IntFlag):
    RESCUE = 1
    MEDIC = 2
    WATER = 4
    FOOD = 8
    SHELTER = 16
    MEDS = 32
    EVAC = 64
    POWER = 128


class Hazard(IntFlag):
    FIRE = 1
    GAS = 2
    WATER = 4
    POWERLINE = 8


@dataclass
class SosFields:
    type: int = MsgType.OTHER
    severity: int = 0
    people: int = 1
    kids: int = 0
    injuries: int = 0
    needs: int = 0
    hazards: int = 0
    floor: int = 0
    loc_src: int = 0
    cell_x: int = 0
    cell_y: int = 0
    minute: int = 0
    msg_id: int = 0
    version: int = VERSION
    place: str = field(default="", compare=False)   # encoder-side only, not on air

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("place")
        return d


# ---------------------------------------------------------------- bit packing

def pack(f: SosFields) -> bytes:
    acc = 0
    for name, bits in LAYOUT:
        v = int(getattr(f, name))
        if not 0 <= v < (1 << bits):
            raise ValueError(f"{name}={v} does not fit in {bits} bits")
        acc = (acc << bits) | v
    acc <<= PACKET_BYTES * 8 - PAYLOAD_BITS
    return acc.to_bytes(PACKET_BYTES, "big")


def unpack(data: bytes) -> SosFields:
    if len(data) != PACKET_BYTES:
        raise ValueError(f"expected {PACKET_BYTES} bytes, got {len(data)}")
    acc = int.from_bytes(data, "big") >> (PACKET_BYTES * 8 - PAYLOAD_BITS)
    vals = {}
    for name, bits in reversed(LAYOUT):
        vals[name] = acc & ((1 << bits) - 1)
        acc >>= bits
    if vals["version"] != VERSION:
        raise ValueError(f"unknown packet version {vals['version']}")
    return SosFields(**vals)


# ---------------------------------------------------------------- keyword rules

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple": 2, "couple": 2,
    "a few": 3, "few": 3, "several": 4, "dozen": 12, "a dozen": 12,
}
_NUM = r"\b(\d{1,2}|a couple|a few|a dozen|couple|few|several|dozen|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_KID = r"(?:kids?|children|child|babies|baby|toddlers?|infants?|boys?|girls?)"
_ADULT = r"(?:people|persons|ppl|adults?|of us|residents|neighbors|neighbours|tenants|students|workers|seniors|elderly)"

# (pattern, type, base severity), checked in priority order.
_TYPE_RULES = [
    (r"\b(fire|flames|burning|smoke)\b", MsgType.FIRE, 3),
    (r"\b(gas leak|smell(?:s)? (?:of )?gas|gas smell|leaking gas)\b", MsgType.GAS_LEAK, 2),
    (r"\b(trapped|pinned|stuck under|buried|under (?:the )?(?:debris|rubble)|cant get out|can't get out)\b", MsgType.TRAPPED, 2),
    (r"\b(collapsed?|caved in|pancaked|fell down|came down)\b", MsgType.COLLAPSE, 3),
    (r"\b(unconscious|not breathing|heart attack|bleeding|broken|fracture|injured|hurt|wounded|labor|labour|seizure|insulin|diabetic|burns?|burned|chest pain|pregnant|cant breathe|can't breathe|crush\w*|unconcious)\b", MsgType.MEDICAL, 2),
    (r"\b(flood(?:ing|ed)?|water main|burst pipe)\b", MsgType.FLOOD, 1),
    (r"\b(power lines?|live wires?|downed lines?|sparking)\b", MsgType.HAZARD, 1),
    (r"\b(missing|can'?t find|looking for|lost contact)\b", MsgType.MISSING, 1),
    (r"\b(evacuat\w*|need to leave|cant leave|can't leave)\b", MsgType.EVAC, 1),
    (r"\b(need|out of|running low|no )\s*(?:\w+ )?(water|food|blankets|supplies|diapers|formula|batteries)\b", MsgType.SUPPLY, 1),
    (r"\b(we are ok|we're ok|im ok|i'm ok|all ok|all safe|we are safe|we're safe|safe|fine|checking in|check in)\b", MsgType.CHECKIN, 0),
]

_INJURY_RULES = [
    (r"\bbleed\w*|\bblood\b", Injury.BLEEDING),
    (r"\bbroken\b|\bfractur\w*|\bbone\b", Injury.FRACTURE),
    (r"\bunconscious\b|\bpassed out\b|\bnot responding\b|\bunresponsive\b", Injury.UNCONSCIOUS),
    (r"\bburn(?:s|ed)?\b", Injury.BURN),
    (r"\bnot breathing\b|\bcan'?t breathe\b|\bcant breath\w*|\bbreathing\b|\basthma\b|\bchoking\b", Injury.BREATHING),
    (r"\bcrush\w*|\bpinned\b", Injury.CRUSH),
    (r"\bheart attack\b|\bchest pain\b|\bcardiac\b", Injury.CARDIAC),
    (r"\bpregnant\b|\bin labou?r\b|\bcontractions\b", Injury.PREGNANT),
]

_NEED_RULES = [
    (r"\brescue\b|\bdig\w*\b|\bget (?:us|them|him|her) out\b|\bextract\w*|\bfire ?fighters?\b|\bsearch and rescue\b", Need.RESCUE),
    (r"\bmedic\w*\b|\bdoctor\b|\bambulance\b|\bparamedic\w*|\bfirst aid\b|\bem[st]\b|\bnurse\b", Need.MEDIC),
    (r"\bwater\b(?! main)", Need.WATER),
    (r"\bfood\b|\bformula\b|\beat\b", Need.FOOD),
    (r"\bshelter\b|\bblankets?\b|\btent\b|\bcold\b|\bnowhere to (?:go|sleep)\b", Need.SHELTER),
    (r"\binsulin\b|\bmeds\b|\bmedicine\b|\bmedication\b|\binhaler\b|\bprescription\b|\boxygen\b", Need.MEDS),
    (r"\bevacuat\w*|\bneed to leave\b|\bcan'?t walk\b|\bwheelchair\b|\btransport\b", Need.EVAC),
    (r"\bpower\b(?! lines?)|\bgenerator\b|\bcharg\w*|\bbatter(?:y|ies)\b", Need.POWER),
]

_HAZARD_RULES = [
    (r"\bfire\b|\bflames\b|\bsmoke\b|\bburning\b", Hazard.FIRE),
    (r"\bgas\b", Hazard.GAS),
    (r"\bflood\w*|\bwater main\b|\bburst pipe\b|\bwater (?:everywhere|rising)\b", Hazard.WATER),
    (r"\bpower lines?\b|\blive wires?\b|\bdowned lines?\b|\bsparking\b", Hazard.POWERLINE),
]

_CRITICAL = r"\b(badly|severe\w*|critical|dying|not breathing|unconscious|heart attack|lots of blood|a lot of blood|hurry|urgent|crush\w*|in labou?r)\b"
_MINOR = r"\b(minor|small cut|scrapes?|bruised?|we are ok|we're ok|im ok|i'm ok|not urgent|no rush)\b"


def _num(token: str) -> int:
    token = token.strip()
    return int(token) if token.isdigit() else _NUM_WORDS.get(token, 1)


def _count_people(t: str) -> tuple[int | None, int]:
    kids = 0
    m = re.search(_NUM + r"\s+(?:little |small |young )?" + _KID, t)
    if m:
        kids = _num(m.group(1))
    elif re.search(r"\b(my|our|a|the|with)\s+(?:little |small |young )?(?:baby|toddler|infant|kid|son|daughter|child)\b", t):
        kids = 1
    total = None
    for pat in (
        r"\bfamily of\s+" + _NUM,
        _NUM + r"\s+(?:of us|people|persons|ppl|adults?|residents|neighbors|neighbours|tenants|students|workers|seniors|elderly)",
        r"\bwe(?: are|'re)\s+" + _NUM + r"\b",
        _NUM + r"\s+(?:trapped|stuck|inside|injured|hurt)\b",
    ):
        m = re.search(pat, t)
        if m:
            total = _num(m.group(1))
            if pat.startswith(_NUM) and "adult" in m.group(0):
                total += kids
            break
    if total is None:
        if re.search(r"\b(my (?:wife|husband|partner|mom|dad|mother|father|grandma|grandpa|roommate|neighbor|neighbour|friend)) and (?:i|me)\b|\b(?:me|i) and my (?:wife|husband|partner|mom|dad|mother|father|grandma|grandpa|roommate|neighbor|neighbour|friend)\b", t):
            total = 2
        elif re.search(r"\b(alone|just me|by myself|only me)\b", t):
            total = 1
    return total, kids


def encode_fields(text: str, node_xy: tuple[float, float] | None = None,
                  minute: int = 0, msg_id: int = 0) -> SosFields:
    t = " " + text.lower().replace("’", "'") + " "
    f = SosFields(minute=minute % 256, msg_id=msg_id & 0xFFFF)

    base = 1
    for pat, mtype, sev in _TYPE_RULES:
        if re.search(pat, t):
            f.type, base = int(mtype), sev
            break

    for pat, flag in _INJURY_RULES:
        if re.search(pat, t):
            f.injuries |= int(flag)
    for pat, flag in _NEED_RULES:
        if re.search(pat, t):
            f.needs |= int(flag)
    for pat, flag in _HAZARD_RULES:
        if re.search(pat, t):
            f.hazards |= int(flag)
    if f.type in (MsgType.TRAPPED, MsgType.COLLAPSE):
        f.needs |= Need.RESCUE
    if f.injuries:
        f.needs |= Need.MEDIC

    sev = base
    if f.injuries:
        sev = max(sev, 2)
    if f.injuries & (Injury.UNCONSCIOUS | Injury.BREATHING | Injury.CARDIAC | Injury.CRUSH):
        sev = 3
    if re.search(_CRITICAL, t):
        sev = 3
    elif re.search(_MINOR, t):
        sev = max(0, sev - 1)
    f.severity = min(3, sev)

    total, kids = _count_people(t)
    if total is None:
        total = kids + 1 if kids else 1
    total = max(total, kids)
    f.people = min(total, 31)
    f.kids = min(kids, 7)

    m = re.search(r"\b(\d)(?:st|nd|rd|th)?\s*(?:floor|fl)\b", t)
    if m:
        f.floor = max(1, min(int(m.group(1)), 7))
    elif re.search(r"\b(basement|ground floor|garage)\b", t):
        f.floor = 1

    loc = geo.find_location(text)
    if loc:
        x, y, f.place = loc
        f.loc_src = 1
    elif node_xy is not None:
        x, y = node_xy
        f.place = "GPS fix"
    else:
        x = y = 0.0
    f.cell_x, f.cell_y = geo.cell_of(x, y)
    return f


def encode(text: str, node_xy=None, minute: int = 0, msg_id: int = 0) -> bytes:
    return pack(encode_fields(text, node_xy, minute, msg_id))


# ---------------------------------------------------------------- gateway decoder

def _flag_names(value: int, enum) -> list[str]:
    return [m.name.lower().replace("_", " ") for m in enum if value & m.value]


def describe(f: SosFields) -> str:
    """Rebuild a readable dispatch line from the decoded fields."""
    kind = MsgType(f.type).name.replace("_", " ") if f.type in MsgType._value2member_map_ else "OTHER"
    parts = [f"[{kind} sev {f.severity}/3]"]
    who = f"{f.people}{'+' if f.people == 31 else ''} {'person' if f.people == 1 else 'people'}"
    if f.kids:
        who += f" ({f.kids} {'child' if f.kids == 1 else 'children'})"
    parts.append(who)
    x, y = geo.cell_center(f.cell_x, f.cell_y)
    where = geo.describe_location(x, y)
    parts.append(f"{where} [cell {f.cell_x},{f.cell_y}{'' if f.loc_src else ', GPS'}]")
    if f.floor:
        parts.append("ground floor" if f.floor == 1 else f"floor {f.floor}")
    if f.injuries:
        parts.append("injuries: " + ", ".join(_flag_names(f.injuries, Injury)))
    if f.needs:
        parts.append("needs: " + ", ".join(_flag_names(f.needs, Need)))
    if f.hazards:
        parts.append("hazards: " + ", ".join(_flag_names(f.hazards, Hazard)))
    parts.append(f"T+{f.minute} min")
    parts.append(f"#{f.msg_id:04X}")
    return " · ".join(parts)


LlmFn = Callable[[str], str]


def llm_prompt(f: SosFields) -> str:
    fields = f.to_dict()
    fields["type"] = MsgType(f.type).name if f.type in MsgType._value2member_map_ else "OTHER"
    fields["injuries"] = _flag_names(f.injuries, Injury)
    fields["needs"] = _flag_names(f.needs, Need)
    fields["hazards"] = _flag_names(f.hazards, Hazard)
    fields["location"] = geo.describe_location(*geo.cell_center(f.cell_x, f.cell_y))
    return (
        "You are the gateway decoder for MeshSOS, a disaster-relief LoRa mesh in San Francisco's "
        "Marina district after an earthquake. A survivor's message was compressed on a microcontroller "
        "into the structured fields below. Write ONE short sentence a dispatcher can act on. "
        "Use only facts in the fields; do not invent details.\n\n" + json.dumps(fields)
    )


def decode(data: bytes, llm: LlmFn | None = None) -> str:
    """Packet bytes -> readable text. ``llm`` optionally rewrites it as natural prose."""
    f = unpack(data)
    text = describe(f)
    if llm is not None:
        try:
            return llm(llm_prompt(f)).strip() or text
        except Exception:   # the radio path must never fail on an LLM outage
            return text
    return text


def anthropic_llm(model: str = "claude-opus-5-5", api_key: str | None = None, timeout: float = 30.0) -> LlmFn:
    """LLM hook using the Claude Messages API over plain urllib.

    Raw HTTP keeps the project at numpy + matplotlib only. Reads
    ANTHROPIC_API_KEY from the environment. Intended for the gateway, which
    has power and backhaul; the nodes never need it. Server-side refusal
    fallbacks are enabled; a refusal falls back to the template text.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "max_tokens": 2048,
            "output_config": {"effort": "low"},
            "fallbacks": "default",
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "server-side-fallback-2026-07-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.load(resp)
        if out.get("stop_reason") == "refusal":
            raise RuntimeError("model declined")
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")

    return call

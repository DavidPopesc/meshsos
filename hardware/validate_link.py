"""Validate a Meshtastic LoRa link between two boards, or range-test one side.

Two boards on this computer (bench test):
    python hardware/validate_link.py --ports /dev/cu.usbserial-A /dev/cu.usbserial-B

One board per laptop (venue range test; run it on both laptops):
    python hardware/validate_link.py --range-test --ports COM5

The bench test checks firmware, region and channel on both boards, then sends
text both ways, a 10-byte MeshSOS packet, and a near-maximum binary payload,
verifying every byte and timing each delivery on one shared clock.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from pathlib import Path

from pubsub import pub

import meshtastic.serial_interface
from meshtastic.protobuf import portnums_pb2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meshsos import codec  # noqa: E402

PRIVATE_APP = portnums_pb2.PortNum.PRIVATE_APP
MAX_PAYLOAD = 216          # largest direct message that survives PKI encryption overhead (measured; broadcasts fit 230)
RX_TIMEOUT = 45.0
_LISTENERS = []            # pypubsub keeps only weak references to listeners


class Board:
    def __init__(self, port: str):
        self.port = port
        self.iface = meshtastic.serial_interface.SerialInterface(devPath=port)
        info = self.iface.getMyNodeInfo()
        self.num = info["num"]
        self.user = info.get("user", {})
        self.name = self.user.get("longName", port)
        self.inbox: list[dict] = []
        self.cv = threading.Condition()

    def describe(self) -> dict:
        lora = self.iface.localNode.localConfig.lora
        ch = self.iface.localNode.channels[0] if self.iface.localNode.channels else None
        meta = getattr(self.iface, "metadata", None)
        return {
            "port": self.port,
            "name": self.name,
            "id": self.user.get("id"),
            "hw": self.user.get("hwModel"),
            "firmware": getattr(meta, "firmware_version", "?") if meta else "?",
            "region": lora.region,
            "region_name": lora.DESCRIPTOR.fields_by_name["region"].enum_type.values_by_number[lora.region].name,
            "preset": lora.DESCRIPTOR.fields_by_name["modem_preset"].enum_type.values_by_number[lora.modem_preset].name,
            "tx_power": lora.tx_power,
            "channel": (ch.settings.name or "(default)") if ch else "?",
        }

    def wait_for(self, pred, timeout: float = RX_TIMEOUT):
        deadline = time.time() + timeout
        with self.cv:
            while True:
                for p in self.inbox:
                    if pred(p):
                        return p
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.cv.wait(left)

    def close(self):
        self.iface.close()


def install_receiver(boards: list[Board]):
    by_iface = {id(b.iface): b for b in boards}

    def on_receive(packet, interface):
        b = by_iface.get(id(interface))
        if b is None or "decoded" not in packet:
            return
        packet["_rx_time"] = time.time()
        with b.cv:
            b.inbox.append(packet)
            b.cv.notify_all()

    _LISTENERS.append(on_receive)
    pub.subscribe(on_receive, "meshtastic.receive")


def summarize(name: str, lat: list[float], snr: list[float], rssi: list[float], sent: int) -> tuple[str, bool]:
    ok = len(lat)
    line = f"{name:34s} {ok}/{sent} delivered"
    if lat:
        line += f" · latency avg {statistics.mean(lat):.2f}s max {max(lat):.2f}s"
    if snr:
        line += f" · SNR {statistics.mean(snr):+.1f} dB · RSSI {statistics.mean(rssi):.0f} dBm"
    return line, ok == sent


def bench(ports: list[str], count: int) -> int:
    print(f"Connecting to {ports[0]} and {ports[1]} …")
    a, b = Board(ports[0]), Board(ports[1])
    install_receiver([a, b])
    failures = []
    try:
        print("\n== Configuration")
        da, db = a.describe(), b.describe()
        for d in (da, db):
            print(f"  {d['name']:14s} {d['id']}  hw={d['hw']}  fw={d['firmware']}  region={d['region_name']}  "
                  f"preset={d['preset']}  tx={d['tx_power'] or 'default'}  channel={d['channel']}")
        if da["region_name"] != "US" or db["region_name"] != "US":
            failures.append("region is not US on both boards (radio stays off while UNSET)")
        if da["channel"] != db["channel"] or da["preset"] != db["preset"]:
            failures.append("channel or modem preset differs between the boards")
        if failures:
            for f in failures:
                print("  FAIL", f)
            return 1

        print(f"\n== Text messages, {count} each way (wantAck, direct)")
        results = []
        for src, dst in ((a, b), (b, a)):
            lat, snr, rssi = [], [], []
            for i in range(count):
                tag = f"meshsos-ping {os.urandom(3).hex()} {i}"
                t0 = time.time()
                src.iface.sendText(tag, destinationId=dst.num, wantAck=True)
                p = dst.wait_for(lambda p, tag=tag: p["decoded"].get("text") == tag)
                if p:
                    lat.append(p["_rx_time"] - t0)
                    if "rxSnr" in p:
                        snr.append(p["rxSnr"])
                        rssi.append(p.get("rxRssi", 0))
                    print(f"  {src.name} → {dst.name}  #{i}  {lat[-1]:.2f}s  SNR {p.get('rxSnr', '?')}  RSSI {p.get('rxRssi', '?')}")
                else:
                    print(f"  {src.name} → {dst.name}  #{i}  LOST (no delivery within {RX_TIMEOUT:.0f}s)")
                time.sleep(1.0)
            results.append(summarize(f"text {src.name} → {dst.name}", lat, snr, rssi, count))

        print("\n== MeshSOS 10-byte packet over PRIVATE_APP")
        sos = "trapped under debris at Scott and Bay, 2 kids, one bleeding badly, need water"
        pkt = codec.encode(sos, minute=7, msg_id=int.from_bytes(os.urandom(2), "big"))
        t0 = time.time()
        a.iface.sendData(pkt, destinationId=b.num, portNum=PRIVATE_APP, wantAck=True)
        p = b.wait_for(lambda p: p["decoded"].get("portnum") in ("PRIVATE_APP", PRIVATE_APP)
                       and p["decoded"].get("payload") == pkt)
        if p:
            print(f"  sent text ({len(sos.encode())} B): {sos}")
            print(f"  on air  ({len(pkt)} B):  {pkt.hex()}")
            print(f"  decoded at {b.name} in {p['_rx_time'] - t0:.2f}s: {codec.decode(p['decoded']['payload'])}")
            results.append((f"{'MeshSOS packet A → B':34s} 1/1 delivered, bytes identical", True))
        else:
            results.append((f"{'MeshSOS packet A → B':34s} 0/1 delivered", False))

        print(f"\n== {MAX_PAYLOAD}-byte binary payload, B → A")
        blob = os.urandom(MAX_PAYLOAD)
        t0 = time.time()
        b.iface.sendData(blob, destinationId=a.num, portNum=PRIVATE_APP, wantAck=True)
        p = a.wait_for(lambda p: p["decoded"].get("payload") == blob)
        if p:
            print(f"  {MAX_PAYLOAD} random bytes received intact in {p['_rx_time'] - t0:.2f}s")
            results.append((f"{f'{MAX_PAYLOAD}-byte payload B → A':34s} 1/1 delivered, bytes identical", True))
        else:
            results.append((f"{f'{MAX_PAYLOAD}-byte payload B → A':34s} 0/1 delivered", False))

        print("\n== Summary")
        all_ok = True
        for line, ok in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {line}")
            all_ok &= ok
        print(f"\n{'LINK OK' if all_ok else 'LINK HAS PROBLEMS'}")
        return 0 if all_ok else 1
    finally:
        a.close()
        b.close()


def range_test(port: str, interval: float) -> int:
    """Broadcast a numbered beacon and print whatever the other side hears; ctrl+c to stop."""
    me = Board(port)
    install_receiver([me])
    d = me.describe()
    print(f"{d['name']} ({d['id']}) fw={d['firmware']} region={d['region_name']} preset={d['preset']} channel={d['channel']}")
    print("Broadcasting a beacon every %.0fs. Lines below are beacons heard from the other board.\n" % interval)
    seq, seen = 0, 0
    try:
        while True:
            me.iface.sendText(f"meshsos-rt {me.name} {seq}")
            seq += 1
            end = time.time() + interval
            while time.time() < end:
                with me.cv:
                    new, me.inbox = me.inbox, []
                for p in new:
                    text = p["decoded"].get("text", "")
                    if text.startswith("meshsos-rt") and p.get("from") != me.num:
                        seen += 1
                        snr = p.get("rxSnr")
                        quality = "solid" if snr is not None and snr > -10 else ("weak" if snr is not None and snr > -17 else "marginal")
                        hops = p.get("hopStart", 0) - p.get("hopLimit", 0) if "hopStart" in p else "?"
                        print(f"{time.strftime('%H:%M:%S')}  heard '{text}'  SNR {snr} dB  RSSI {p.get('rxRssi')} dBm  "
                              f"hops {hops}  [{quality}]  (sent {seq}, heard {seen})")
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        me.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--count", type=int, default=5, help="text messages each way in the bench test")
    ap.add_argument("--range-test", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    if a.range_test:
        return range_test(a.ports[0], a.interval)
    if len(a.ports) != 2:
        ap.error("the bench test needs exactly two --ports")
    return bench(a.ports, a.count)


if __name__ == "__main__":
    sys.exit(main())

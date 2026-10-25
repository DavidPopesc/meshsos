# MeshSOS on real radios

Two **LilyGO T3 LoRa32 V1.6.1** boards (ESP32-PICO-D4 + SX1276, 915 MHz) running stock **Meshtastic 2.7.26**,
bench-validated on 2026-10-24. Each board connects to a laptop by Micro-USB and acts as a radio modem. Meshtastic
handles relaying, and your code talks to it through the `meshtastic` Python package.

## Validated result

`python hardware/validate_link.py --ports <A> <B>`

| Check | Result |
|---|---|
| Text A → B and B → A (direct, acked) | 10/10 delivered, ~1.6–1.8 s each, SNR +7 to +12 dB |
| MeshSOS 10-byte packet (PRIVATE_APP port) | delivered byte-identical, decoded at the far end |
| Largest direct message | **216 bytes**. 220+ is dropped because the public-key encryption adds overhead |
| Largest broadcast | 230 bytes |

## Gotchas found on the bench

* **Don't test the boards side by side at full power.** At about 30 cm and 20 dBm, frames over ~130 bytes failed
  CRC (`error=-7` in the firmware log). At 2 dBm everything up to 230 bytes went through. Either keep the boards a few
  metres apart, or lower the power for desk tests with `meshtastic --set lora.tx_power 2` and set it back to `0`
  (full power) for the real thing.
* **Use 230400 baud with esptool.** 460800 and 921600 corrupted transfers on these CH9102 USB-serial chips (at least
  through a USB hub).
* **Region must be US.** With region UNSET the radio stays off. One board lost its region setting during back-to-back
  config writes, so verify with `--info` after configuring.
* **Meshtastic lists the T3 V1.6.1 as "not recommended for new purchases"** and is phasing out support. v2.7.26 still
  ships a build for it and marks it `activelySupported`, but plan on newer boards (T3S3, Heltec V3/V4, T-Deck Plus)
  for anything beyond this hackathon.

## Reproduce

```bash
pip install esptool meshtastic
```

1. **Back up the factory firmware** (optional; LilyGO ships a Paxcounter demo):
   `esptool --port <PORT> --baud 230400 read-flash 0 0x400000 backup.bin`
2. **Flash** `firmware-tlora-v2-1-1_6-2.7.26.54e0d8d` from the `firmware-esp32-2.7.26.54e0d8d.zip` release asset
   (offsets from the board's `.mt.json`):
   ```
   esptool --port <PORT> --baud 230400 erase-flash
   esptool --port <PORT> --baud 230400 write-flash 0x0 firmware-tlora-v2-1-1_6-2.7.26.54e0d8d.factory.bin \
       0x260000 mt-esp32-ota.bin 0x300000 littlefs-tlora-v2-1-1_6-2.7.26.54e0d8d.bin
   ```
3. **Configure** (antenna attached first, because setting the region turns the radio on):
   ```
   meshtastic --port <PORT> --set-owner "MeshSOS A" --set-owner-short MSA
   meshtastic --port <PORT> --set lora.region US
   meshtastic --port <PORT> --seturl "$(cat hardware/secrets/channel_url.txt)"   # private MeshSOS channel
   ```
   `hardware/secrets/channel_url.txt` is git-ignored. It holds the private channel link (name and key) and is how you
   add a third board to the same mesh. Don't share it publicly.
4. **Validate:** `python hardware/validate_link.py --ports <A> <B>`

## At the venue

Run this on each laptop with its board plugged in. It prints every beacon heard from the other side with SNR, RSSI and
a solid/weak/marginal rating:

```bash
python hardware/validate_link.py --range-test --ports <PORT>
```

SNR above −10 dB is solid; −10 to −17 dB works but is fragile. If the link is weak, place a third board on a power bank
partway between them, and Meshtastic relays through it automatically (the output's `hops` column shows 1).

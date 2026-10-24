// meshsos_codec.h - MeshSOS 10-byte SOS packet for ESP32 LoRa nodes.
//
// Bit-exact twin of meshsos/codec.py (tests/test_firmware_compat.py compiles
// this header on the host and checks it against the Python reference).
// Header-only, no heap, no STL: works under Arduino-ESP32 and ESP-IDF.
//
// Target hardware
//   LilyGO T3 LoRa32 V1.6.1  - ESP32 + SX1276 (915 MHz), 0.96" SSD1306 OLED
//   LilyGO T-Deck Plus       - ESP32-S3 + SX1262, keyboard + screen (field terminal)
//
// Packet layout, MSB first, 75 bits padded to 10 bytes:
//   version 2 | type 4 | severity 2 | people 5 | kids 3 | injuries 8 | needs 8 |
//   hazards 4 | floor 3 | loc_src 1 | cell_x 6 | cell_y 5 | minute 8 | msg_id 16
//
// Usage with RadioLib on a T3 V1.6.1 (check the pins against your board revision):
//
//   #include <RadioLib.h>
//   #include "meshsos_codec.h"
//   SX1276 radio = new Module(18 /*NSS*/, 26 /*DIO0*/, 23 /*RST*/, 33 /*DIO1*/);
//   void setup() {
//     SPI.begin(5 /*SCK*/, 19 /*MISO*/, 27 /*MOSI*/, 18 /*NSS*/);
//     // 915 MHz, 125 kHz, SF9, CR 4/5, private sync word, 17 dBm, 8-symbol preamble
//     radio.begin(915.0, 125.0, 9, 5, RADIOLIB_SX127X_SYNC_WORD, 17, 8);
//     radio.setCRC(true);
//   }
//   void sendSos(const char* text, float x_m, float y_m, uint8_t minute, uint16_t id) {
//     meshsos::SosPacket p = meshsos::encodeText(text, x_m, y_m, minute, id);
//     uint8_t buf[meshsos::kPacketBytes];
//     meshsos::pack(p, buf);
//     // prepend your mesh header here; keep the frame under the 400 ms dwell limit
//     if (meshsos::timeOnAirMs(sizeof(buf) + 12, 9) <= meshsos::kMaxDwellMs) radio.transmit(buf, sizeof(buf));
//   }

#pragma once

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace meshsos {

constexpr uint8_t kVersion = 1;
constexpr size_t kPacketBytes = 10;
constexpr uint8_t kPayloadBits = 75;
constexpr float kMaxDwellMs = 400.0f;

enum MsgType : uint8_t {
  CHECKIN = 0, TRAPPED = 1, MEDICAL = 2, FIRE = 3, GAS_LEAK = 4, COLLAPSE = 5,
  FLOOD = 6, SUPPLY = 7, MISSING = 8, HAZARD = 9, EVAC = 10, OTHER = 15,
};
enum Injury : uint8_t {
  INJ_BLEEDING = 1, INJ_FRACTURE = 2, INJ_UNCONSCIOUS = 4, INJ_BURN = 8,
  INJ_BREATHING = 16, INJ_CRUSH = 32, INJ_CARDIAC = 64, INJ_PREGNANT = 128,
};
enum Need : uint8_t {
  NEED_RESCUE = 1, NEED_MEDIC = 2, NEED_WATER = 4, NEED_FOOD = 8,
  NEED_SHELTER = 16, NEED_MEDS = 32, NEED_EVAC = 64, NEED_POWER = 128,
};
enum Hazard : uint8_t { HAZ_FIRE = 1, HAZ_GAS = 2, HAZ_WATER = 4, HAZ_POWERLINE = 8 };

struct SosPacket {
  uint8_t version = kVersion;
  uint8_t type = OTHER;
  uint8_t severity = 0;
  uint8_t people = 1;
  uint8_t kids = 0;
  uint8_t injuries = 0;
  uint8_t needs = 0;
  uint8_t hazards = 0;
  uint8_t floor = 0;
  uint8_t loc_src = 0;
  uint8_t cell_x = 0;
  uint8_t cell_y = 0;
  uint8_t minute = 0;
  uint16_t msg_id = 0;
};

// ------------------------------------------------------------------ bit packing

struct BitWriter {
  uint8_t* out;
  uint16_t bit = 0;
  void put(uint32_t value, uint8_t bits) {
    for (int i = bits - 1; i >= 0; --i, ++bit) {
      if ((value >> i) & 1u) out[bit >> 3] |= (uint8_t)(0x80u >> (bit & 7));
    }
  }
};

struct BitReader {
  const uint8_t* in;
  uint16_t bit = 0;
  uint32_t get(uint8_t bits) {
    uint32_t v = 0;
    for (uint8_t i = 0; i < bits; ++i, ++bit) v = (v << 1) | ((in[bit >> 3] >> (7 - (bit & 7))) & 1u);
    return v;
  }
};

// Returns false if a field does not fit its bit width.
inline bool pack(const SosPacket& p, uint8_t out[kPacketBytes]) {
  if (p.version > 3 || p.type > 15 || p.severity > 3 || p.people > 31 || p.kids > 7 || p.hazards > 15 ||
      p.floor > 7 || p.loc_src > 1 || p.cell_x > 63 || p.cell_y > 31) {
    return false;
  }
  memset(out, 0, kPacketBytes);
  BitWriter w{out};
  w.put(p.version, 2);
  w.put(p.type, 4);
  w.put(p.severity, 2);
  w.put(p.people, 5);
  w.put(p.kids, 3);
  w.put(p.injuries, 8);
  w.put(p.needs, 8);
  w.put(p.hazards, 4);
  w.put(p.floor, 3);
  w.put(p.loc_src, 1);
  w.put(p.cell_x, 6);
  w.put(p.cell_y, 5);
  w.put(p.minute, 8);
  w.put(p.msg_id, 16);
  return true;
}

inline bool unpack(const uint8_t in[kPacketBytes], SosPacket& p) {
  BitReader r{in};
  p.version = r.get(2);
  p.type = r.get(4);
  p.severity = r.get(2);
  p.people = r.get(5);
  p.kids = r.get(3);
  p.injuries = r.get(8);
  p.needs = r.get(8);
  p.hazards = r.get(4);
  p.floor = r.get(3);
  p.loc_src = r.get(1);
  p.cell_x = r.get(6);
  p.cell_y = r.get(5);
  p.minute = r.get(8);
  p.msg_id = r.get(16);
  return p.version == kVersion;
}

// ------------------------------------------------------------------ map grid

// Local metres from the south-west corner of the MeshSOS map (37.7965 N, -122.4515 E).
constexpr float kMapWidthM = 1935.06f;
constexpr float kMapHeightM = 1335.84f;
constexpr uint8_t kGridNx = 64, kGridNy = 32;

inline void cellOf(float x, float y, uint8_t& gx, uint8_t& gy) {
  int ix = (int)(x / kMapWidthM * kGridNx);
  int iy = (int)(y / kMapHeightM * kGridNy);
  gx = (uint8_t)(ix < 0 ? 0 : ix > kGridNx - 1 ? kGridNx - 1 : ix);
  gy = (uint8_t)(iy < 0 ? 0 : iy > kGridNy - 1 ? kGridNy - 1 : iy);
}

struct Place { const char* name; float pos; };
static const Place kNsStreets[] = {
  {"lyon", 184.7f}, {"baker", 492.6f}, {"broderick", 642.1f}, {"divisadero", 782.8f}, {"scott", 923.5f},
  {"pierce", 1073.1f}, {"steiner", 1213.8f}, {"fillmore", 1363.3f}, {"webster", 1512.9f},
  {"buchanan", 1662.4f}, {"laguna", 1803.1f},
};
static const Place kEwStreets[] = {
  {"greenwich", 100.2f}, {"lombard", 256.0f}, {"chestnut", 456.4f}, {"francisco", 612.3f}, {"bay", 768.1f},
  {"north point", 879.4f}, {"beach", 990.7f}, {"marina", 1124.3f},
};
struct Landmark { const char* name; float x, y; };
static const Landmark kLandmarks[] = {
  {"palace of fine arts", 290.3f, 690.2f}, {"palace", 290.3f, 690.2f}, {"rotunda", 290.3f, 690.2f},
  {"lagoon", 404.6f, 801.5f}, {"marina green", 1187.4f, 1168.9f}, {"yacht harbor", 659.7f, 1257.9f},
  {"harbor", 659.7f, 1257.9f}, {"safeway", 1671.2f, 946.2f}, {"marina middle school", 1319.4f, 723.6f},
  {"moscone rec", 1732.8f, 556.6f}, {"moscone", 1732.8f, 556.6f}, {"presidio", 88.0f, 389.6f},
  {"fort mason", 1891.1f, 1057.5f}, {"chestnut st shops", 1275.4f, 456.4f},
};

// ------------------------------------------------------------------ keyword encoder
//
// A strstr()-based port of the Python rules. It covers the common phrasings;
// codec.py stays the reference for edge cases (typos, number words in odd places).

namespace detail {

inline bool has(const char* t, const char* word) {
  // whole-word-ish match: the hit must not be glued to letters on its left
  size_t n = strlen(word);
  for (const char* p = strstr(t, word); p; p = strstr(p + 1, word)) {
    bool left_ok = p == t || !isalpha((unsigned char)p[-1]);
    bool right_ok = !isalpha((unsigned char)p[n]) || word[n - 1] == ' ';
    if (left_ok && right_ok) return true;
  }
  return false;
}

inline bool any(const char* t, const char* const* words) {
  for (; *words; ++words) {
    if (has(t, *words)) return true;
  }
  return false;
}

inline int numberBefore(const char* t, const char* p) {
  // parse the word immediately before position p ("2 kids", "two kids")
  static const char* const kWords[] = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                                       "eight", "nine", "ten", "eleven", "twelve"};
  const char* end = p;
  while (end > t && end[-1] == ' ') --end;
  const char* start = end;
  while (start > t && isalnum((unsigned char)start[-1])) --start;
  if (start == end) return -1;
  char buf[12] = {0};
  size_t len = (size_t)(end - start) < sizeof(buf) - 1 ? (size_t)(end - start) : sizeof(buf) - 1;
  memcpy(buf, start, len);
  if (isdigit((unsigned char)buf[0])) return atoi(buf);
  for (int i = 0; i < 13; ++i) {
    if (strcmp(buf, kWords[i]) == 0) return i;
  }
  if (strcmp(buf, "couple") == 0) return 2;
  if (strcmp(buf, "few") == 0) return 3;
  return -1;
}

inline int countBefore(const char* t, const char* const* nouns) {
  for (; *nouns; ++nouns) {
    for (const char* p = strstr(t, *nouns); p; p = strstr(p + 1, *nouns)) {
      if (p > t && p[-1] == ' ') {
        int n = numberBefore(t, p);
        if (n >= 0) return n;
      }
    }
  }
  return -1;
}

}  // namespace detail

inline SosPacket encodeText(const char* text, float node_x, float node_y, uint8_t minute, uint16_t msg_id) {
  using namespace detail;
  char t[256];
  size_t n = 0;
  t[n++] = ' ';
  for (const char* s = text; *s && n < sizeof(t) - 2; ++s) t[n++] = (char)tolower((unsigned char)*s);
  t[n++] = ' ';
  t[n] = 0;

  SosPacket p;
  p.minute = minute;
  p.msg_id = msg_id;

  static const char* const kFire[] = {"fire", "flames", "burning", "smoke", nullptr};
  static const char* const kGas[] = {"gas leak", "smell gas", "smells of gas", "gas smell", "leaking gas", nullptr};
  static const char* const kTrapped[] = {"trapped", "pinned", "stuck under", "buried", "under debris",
                                         "under the debris", "under rubble", "under the rubble", "cant get out", nullptr};
  static const char* const kCollapse[] = {"collapsed", "collapse", "caved in", "pancaked", "came down", nullptr};
  static const char* const kMedical[] = {"unconscious", "not breathing", "heart attack", "bleeding", "broken",
                                         "fracture", "injured", "hurt", "wounded", "seizure", "insulin", "burns",
                                         "chest pain", "pregnant", "cant breathe", "crushed", nullptr};
  static const char* const kFlood[] = {"flood", "flooding", "flooded", "water main", "burst pipe", nullptr};
  static const char* const kPower[] = {"power line", "power lines", "live wire", "downed lines", "sparking", nullptr};
  static const char* const kMissing[] = {"missing", "cant find", "looking for", "lost contact", nullptr};
  static const char* const kEvac[] = {"evacuate", "evacuating", "need to leave", "cant leave", nullptr};
  static const char* const kSupply[] = {"out of water", "out of food", "need water", "need food", "need blankets",
                                        "need supplies", "running low", nullptr};
  static const char* const kOk[] = {"we are ok", "im ok", "all ok", "all safe", "we are safe", "safe", "fine",
                                    "checking in", nullptr};
  struct Rule { const char* const* words; uint8_t type, sev; };
  const Rule rules[] = {{kFire, FIRE, 3}, {kGas, GAS_LEAK, 2}, {kTrapped, TRAPPED, 2}, {kCollapse, COLLAPSE, 3},
                        {kMedical, MEDICAL, 2}, {kFlood, FLOOD, 1}, {kPower, HAZARD, 1}, {kMissing, MISSING, 1},
                        {kEvac, EVAC, 1}, {kSupply, SUPPLY, 1}, {kOk, CHECKIN, 0}};
  uint8_t sev = 1;
  for (const Rule& r : rules) {
    if (any(t, r.words)) {
      p.type = r.type;
      sev = r.sev;
      break;
    }
  }

  struct Flag { const char* word; uint8_t bit; };
  static const Flag kInj[] = {{"bleed", INJ_BLEEDING}, {"blood", INJ_BLEEDING}, {"broken", INJ_FRACTURE},
                              {"fractur", INJ_FRACTURE}, {"unconscious", INJ_UNCONSCIOUS}, {"passed out", INJ_UNCONSCIOUS},
                              {"burns", INJ_BURN}, {"burned", INJ_BURN}, {"not breathing", INJ_BREATHING},
                              {"cant breathe", INJ_BREATHING}, {"breathing", INJ_BREATHING}, {"asthma", INJ_BREATHING},
                              {"crush", INJ_CRUSH}, {"pinned", INJ_CRUSH}, {"heart attack", INJ_CARDIAC},
                              {"chest pain", INJ_CARDIAC}, {"pregnant", INJ_PREGNANT}, {"in labor", INJ_PREGNANT}};
  static const Flag kNeed[] = {{"rescue", NEED_RESCUE}, {"dig", NEED_RESCUE}, {"firefighter", NEED_RESCUE},
                               {"medic", NEED_MEDIC}, {"doctor", NEED_MEDIC}, {"ambulance", NEED_MEDIC},
                               {"food", NEED_FOOD}, {"formula", NEED_FOOD}, {"shelter", NEED_SHELTER},
                               {"blanket", NEED_SHELTER}, {"insulin", NEED_MEDS}, {"meds", NEED_MEDS},
                               {"medicine", NEED_MEDS}, {"oxygen", NEED_MEDS}, {"inhaler", NEED_MEDS},
                               {"evacuat", NEED_EVAC}, {"wheelchair", NEED_EVAC}, {"cant walk", NEED_EVAC},
                               {"generator", NEED_POWER}, {"charg", NEED_POWER}, {"batter", NEED_POWER}};
  static const Flag kHaz[] = {{"fire", HAZ_FIRE}, {"flames", HAZ_FIRE}, {"smoke", HAZ_FIRE}, {"gas", HAZ_GAS},
                              {"flood", HAZ_WATER}, {"water main", HAZ_WATER}, {"power line", HAZ_POWERLINE},
                              {"live wire", HAZ_POWERLINE}, {"sparking", HAZ_POWERLINE}};
  // prefix matches (no right-hand word boundary), as the Python \bword\w* rules
  auto prefix = [&](const char* w) {
    for (const char* q = strstr(t, w); q; q = strstr(q + 1, w)) {
      if (q == t || !isalpha((unsigned char)q[-1])) return true;
    }
    return false;
  };
  for (const Flag& f : kInj) if (prefix(f.word)) p.injuries |= f.bit;
  for (const Flag& f : kNeed) if (prefix(f.word)) p.needs |= f.bit;
  for (const Flag& f : kHaz) if (prefix(f.word)) p.hazards |= f.bit;
  if (has(t, "water") && !has(t, "water main")) p.needs |= NEED_WATER;
  if (p.type == TRAPPED || p.type == COLLAPSE) p.needs |= NEED_RESCUE;
  if (p.injuries) p.needs |= NEED_MEDIC;

  if (p.injuries) sev = sev > 2 ? sev : 2;
  if (p.injuries & (INJ_UNCONSCIOUS | INJ_BREATHING | INJ_CARDIAC | INJ_CRUSH)) sev = 3;
  static const char* const kCritical[] = {"badly", "severe", "critical", "dying", "hurry", "urgent", nullptr};
  static const char* const kMinor[] = {"minor", "scrape", "bruise", "we are ok", "im ok", "not urgent", nullptr};
  if (any(t, kCritical)) sev = 3;
  else if (any(t, kMinor) && sev > 0) sev -= 1;
  p.severity = sev > 3 ? 3 : sev;

  static const char* const kKidNouns[] = {"kids", "kid", "children", "child", "babies", "baby", "toddlers", nullptr};
  static const char* const kPeopleNouns[] = {"of us", "people", "persons", "adults", "residents", "neighbors", nullptr};
  int kids = countBefore(t, kKidNouns);
  if (kids < 0) kids = (has(t, "my baby") || has(t, "our baby") || has(t, "my son") || has(t, "my daughter")) ? 1 : 0;
  int total = -1;
  const char* fam = strstr(t, "family of ");
  if (fam) total = atoi(fam + 10) > 0 ? atoi(fam + 10) : -1;
  if (total < 0) total = countBefore(t, kPeopleNouns);
  if (total < 0 && (has(t, "me and my") || has(t, "and i"))) total = 2;
  if (total < 0 && (has(t, "alone") || has(t, "just me"))) total = 1;
  if (total < 0) total = kids ? kids + 1 : 1;
  if (total < kids) total = kids;
  p.people = (uint8_t)(total > 31 ? 31 : total);
  p.kids = (uint8_t)(kids > 7 ? 7 : kids);

  // "3rd floor" / "3 floor": step back over an ordinal suffix to the digit
  const char* fl = strstr(t, " floor");
  const char* d = fl ? fl - 1 : nullptr;
  while (d && d > t && isalpha((unsigned char)*d)) --d;
  if (d && isdigit((unsigned char)*d)) {
    int f = *d - '0';
    p.floor = (uint8_t)(f < 1 ? 1 : f > 7 ? 7 : f);
  } else if (has(t, "basement") || has(t, "ground floor") || has(t, "garage")) {
    p.floor = 1;
  }

  float x = node_x, y = node_y;
  bool found = false;
  for (const Landmark& l : kLandmarks) {
    if (strstr(t, l.name)) { x = l.x; y = l.y; found = true; break; }
  }
  if (!found) {
    const Place* ns = nullptr;
    const Place* ew = nullptr;
    for (const Place& s : kNsStreets) if (has(t, s.name)) { ns = &s; break; }
    for (const Place& s : kEwStreets) if (has(t, s.name)) { ew = &s; break; }
    if (ns && ew && (strstr(t, " and ") || strchr(t, '&') || strchr(t, '/'))) {
      x = ns->pos; y = ew->pos; found = true;
    }
  }
  p.loc_src = found ? 1 : 0;
  cellOf(x, y, p.cell_x, p.cell_y);
  return p;
}

// ------------------------------------------------------------------ airtime

// Semtech time-on-air for 125 kHz, CR 4/5, 8-symbol preamble, explicit header, CRC on.
inline float timeOnAirMs(uint8_t payload_len, uint8_t sf) {
  const float t_sym = (float)(1u << sf) / 125.0f;  // ms
  const int de = t_sym > 16.0f ? 1 : 0;
  const float t_pre = (8 + 4.25f) * t_sym;
  int num = 8 * payload_len - 4 * sf + 28 + 16;
  int den = 4 * (sf - 2 * de);
  int n = (num + den - 1) / den;
  if (num <= 0) n = 0;
  return t_pre + (8 + n * 5) * t_sym;
}

// Human-readable line for the OLED / serial console.
inline int describe(const SosPacket& p, char* out, size_t len) {
  static const char* const kTypes[16] = {"CHECKIN", "TRAPPED", "MEDICAL", "FIRE", "GAS LEAK", "COLLAPSE",
                                         "FLOOD", "SUPPLY", "MISSING", "HAZARD", "EVAC", "?", "?", "?", "?", "OTHER"};
  return snprintf(out, len, "%s s%u %up/%uk @%u,%u T+%u #%04X", kTypes[p.type & 15], p.severity, p.people, p.kids,
                  p.cell_x, p.cell_y, p.minute, p.msg_id);
}

}  // namespace meshsos

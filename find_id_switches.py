import json
import math

data = json.load(open("tracks.json"))
players = data["players"]
fps = data["fps"]

by_track = {}
for p in players:
    by_track.setdefault(p["track_id"], []).append(p)
for samples in by_track.values():
    samples.sort(key=lambda s: s["frame"])

tracks = []
for tid, samples in by_track.items():
    tracks.append({
        "id": tid,
        "start_t": samples[0]["t"], "end_t": samples[-1]["t"],
        "start_xy": (samples[0]["x_ft"], samples[0]["y_ft"]),
        "end_xy": (samples[-1]["x_ft"], samples[-1]["y_ft"]),
    })

MAX_GAP_SECONDS = 1.0
MAX_DIST_FT = 6.0

print(f"Looking for track pairs where one ends within {MAX_GAP_SECONDS}s and {MAX_DIST_FT}ft of another starting.")
print("These are likely the SAME physical player getting a new track ID (an ID switch), not two real people.\n")

candidates = []
for a in tracks:
    for b in tracks:
        if a["id"] == b["id"]:
            continue
        gap = b["start_t"] - a["end_t"]
        if 0 <= gap <= MAX_GAP_SECONDS:
            dist = math.hypot(a["end_xy"][0] - b["start_xy"][0], a["end_xy"][1] - b["start_xy"][1])
            if dist <= MAX_DIST_FT:
                candidates.append((a["id"], b["id"], gap, dist))

candidates.sort(key=lambda c: c[2])
if not candidates:
    print("No likely ID-switch pairs found within the thresholds -- these tracks may be genuinely distinct people.")
else:
    print(f"{'track A':>8} -> {'track B':>8}   gap_s   dist_ft")
    for a_id, b_id, gap, dist in candidates:
        print(f"{a_id:>8} -> {b_id:>8}   {gap:>5.2f}   {dist:>7.1f}")
    print(f"\n{len(candidates)} likely ID-switch pair(s) found out of {len(tracks)} total tracks.")

"""SkillCorner open data (github.com/SkillCorner/opendata, MIT licence): broadcast tracking of 20
A-League 2024/25 matches at 10 fps. Used for one thing only: how often a TV broadcast does not show
the pitch (replays, close-ups) while the ball is dead and while it is live.

Per frame we read
- camera on the pitch: SkillCorner projects the image corners onto the pitch; no projection means the
  broadcast shows something else. The same idea as a valid homography in the BAR-ATH broadcast file.
- ball in play: SkillCorner's phases of play are defined only while the ball is in play ("when the
  ball is out of play there is no phase of play"), so a frame inside a period and outside every
  phase is a dead ball.

    python -m fc_anomaly.skillcorner        # downloads nothing; prints the numbers per match
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fc_anomaly.data import FPS, ROOT, save_atomic

SC_DIR = ROOT / "data" / "external" / "skillcorner"
SC_FPS = 10


def match_ids() -> list:
    return [str(m["id"]) for m in json.loads((SC_DIR / "matches.json").read_text())]


def camera_and_dead(mid: str) -> dict:
    """Per period, at 25 fps: camera_on and dead (bool arrays). Cached as npz."""
    cache = SC_DIR / "cache" / f"{mid}.npz"
    if cache.exists():
        z = np.load(cache)
        return {int(k[3:]): (z[k], z["dead" + k[3:]]) for k in z.files if k.startswith("cam")}
    period, cam = [], []
    with open(SC_DIR / mid / f"{mid}_tracking_extrapolated.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            period.append(d["period"] if d["period"] is not None else 0)
            cam.append(d["image_corners_projection"]["x_top_left"] is not None)
    period, cam = np.array(period), np.array(cam)
    live = np.zeros(len(period), dtype=bool)
    ph = pd.read_csv(SC_DIR / mid / f"{mid}_phases_of_play.csv")
    for a, b in zip(ph.frame_start, ph.frame_end):
        live[a:b + 1] = True
    out = {}
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx) == 0:
            continue
        # 10 fps -> 25 fps: every 25 fps frame takes the nearest 10 fps frame
        up = np.minimum((np.arange(len(idx) * FPS // SC_FPS) * SC_FPS) // FPS, len(idx) - 1)
        out[p] = (cam[idx][up], ~live[idx][up])
    save_atomic(cache, lambda fh: np.savez_compressed(fh, **{f"cam{p}": v[0] for p, v in out.items()},
                                                      **{f"dead{p}": v[1] for p, v in out.items()}))
    return out


def stretches(cam: np.ndarray, dead: np.ndarray) -> list:
    """(is_dead, camera_on pattern) for every stretch of constant ball state."""
    d = np.flatnonzero(np.diff(dead.astype(np.int8))) + 1
    edges = np.r_[0, d, len(dead)]
    return [(bool(dead[a]), cam[a:b]) for a, b in zip(edges[:-1], edges[1:])]


def library(mids) -> dict:
    """{'dead': [patterns], 'live': [patterns]} from these matches."""
    lib = {"dead": [], "live": []}
    for mid in mids:
        for cam, dead in camera_and_dead(mid).values():
            for is_dead, pat in stretches(cam, dead):
                lib["dead" if is_dead else "live"].append(pat)
    return lib


if __name__ == "__main__":
    rows = []
    for mid in match_ids():
        for p, (cam, dead) in camera_and_dead(mid).items():
            rows.append({"match": mid, "period": p, "dead share": dead.mean(),
                         "camera off | dead": (~cam[dead]).mean(), "camera off | live": (~cam[~dead]).mean()})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    print("\nall:", df[["dead share", "camera off | dead", "camera off | live"]].mean().round(3).to_dict())

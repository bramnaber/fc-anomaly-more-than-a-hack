"""The TV broadcast as a dead-ball signal: is the pitch on screen or not?

During a stoppage the director cuts to close-ups and replays; during play the main camera shows the
pitch. On 204 SoccerNet games (external) the pitch is off screen
on 64% of dead seconds against 13% of live seconds.

BAR-ATH: the broadcast homography (DATA.md) is `valid` exactly when a pitch transform exists, i.e.
when the pitch is on screen; replays, close-ups and crowd shots have none. The broadcast recording
is not frame-aligned to the tactical one, so `offsets()` finds, per half, the shift that best lines
up the ball in the two tracking files. That is an automatic alignment of two inputs, no labels.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from io import StringIO
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fc_anomaly.data import CACHE_DIR, FPS, ROOT, save_atomic

BROADCAST_DIR = ROOT / "data" / "broadcast"
TACTICAL_DIR = ROOT / "data" / "tactical"


def _ball(source: str) -> np.ndarray:
    """(T, 2) ball position in normalised pitch units, row i = video frame i of that recording."""
    cache = CACHE_DIR / f"barath_{source}_ball.npy"
    if cache.exists():
        return np.load(cache)
    d = TACTICAL_DIR if source == "tactical" else BROADCAST_DIR
    n = len(list(ET.parse(d / f"BAR-ATH_{source}_atd.xml").getroot().iter("Player")))
    with open(d / f"BAR-ATH_{source}_atd.txt") as fh:
        text = fh.read().replace(":", ",").replace(";", ",")
    arr = pd.read_csv(StringIO(text), header=None, dtype=np.float64, na_values=["NaN"]).to_numpy()
    ball = arr[:, 1 + 3 * n:1 + 3 * n + 2].astype(np.float32)
    save_atomic(cache, lambda fh: np.save(fh, ball))
    return ball


def broadcast_valid() -> np.ndarray:
    """(T_broadcast,) bool: a homography exists for this broadcast frame (the pitch is on screen)."""
    h = pd.read_csv(BROADCAST_DIR / "BAR-ATH_broadcast_homography.csv", usecols=["frame", "valid"])
    out = np.zeros(int(h.frame.max()) + 1, dtype=bool)
    out[h.frame.to_numpy()] = h.valid.to_numpy() == 1
    return out


def _miss(tac: np.ndarray, bc: np.ndarray, lag: int, frames: np.ndarray) -> tuple[float, int]:
    """Median ball distance between broadcast frame f and tactical frame f + lag, over `frames`."""
    f = frames[(frames + lag >= 0) & (frames + lag < len(tac))]
    d = np.linalg.norm(tac[f + lag] - bc[f], axis=1)
    d = d[np.isfinite(d)]
    return (float(np.median(d)) if len(d) > 200 else np.inf), len(d)


def _best_lag(tac, bc, frames, step=25) -> int:
    coarse = min(range(-len(bc), len(tac), step), key=lambda L: _miss(tac, bc, L, frames[::5])[0])
    return min(range(coarse - step, coarse + step + 1), key=lambda L: _miss(tac, bc, L, frames)[0])


def offsets(chunk: int = 9000) -> dict:
    """tactical_frame = broadcast_frame + offset[half], and the first broadcast frame of half 2.

    Every chunk of broadcast frames votes for the shift that lines the ball up best; the two most
    common shifts are the two halves (the broadcast shortens halftime, so they differ). The switch
    point is where, in 500-frame steps, the second shift starts to fit better than the first."""
    cache = CACHE_DIR / "barath_broadcast_offsets.npy"
    if cache.exists():
        o1, o2, switch = np.load(cache).tolist()
        return {"offset": {1: o1, 2: o2}, "half2_from": switch}
    tac, bc = _ball("tactical"), _ball("broadcast")
    seen = np.flatnonzero(np.isfinite(bc[:, 0]))
    votes = []
    for a in range(0, len(bc), chunk):
        fr = seen[(seen >= a) & (seen < a + chunk)]
        if len(fr) >= 1000:
            votes.append((a, _best_lag(tac, bc, fr)))
    lags = np.array([lag for _, lag in votes])
    near = lambda x: lags[np.abs(lags - x) <= 50]                                  # noqa: E731
    o1, o2 = (int(np.bincount(near(x) - near(x).min()).argmax() + near(x).min()) for x in (lags[0], lags[-1]))
    better2 = []
    for a in range(0, len(bc), 500):
        fr = seen[(seen >= a) & (seen < a + 500)]
        if len(fr) > 200:
            better2.append((a, _miss(tac, bc, o2, fr)[0] < _miss(tac, bc, o1, fr)[0]))
    switch = len(bc)
    for i, (a, b2) in enumerate(better2):                       # first step from which o2 keeps winning
        if b2 and all(x for _, x in better2[i:i + 6]):
            switch = a
            break
    save_atomic(cache, lambda fh: np.save(fh, np.array([o1, o2, switch])))
    return {"offset": {1: o1, 2: o2}, "half2_from": switch}


def ball_on_tactical() -> np.ndarray:
    """(199319, 2) the broadcast tracking's ball per TACTICAL frame (normalised pitch units), NaN where
    the broadcast has no ball or no frame maps here. Same alignment as pitch_on_screen."""
    al = offsets()
    bc = _ball("broadcast")
    out = np.full((len(_ball("tactical")), 2), np.nan, dtype=np.float32)
    f = np.arange(len(bc))
    t = f + np.where(f < al["half2_from"], al["offset"][1], al["offset"][2])
    ok = (t >= 0) & (t < len(out))
    out[t[ok]] = bc[ok]
    return out


def pitch_on_screen() -> np.ndarray:
    """(199319,) float per TACTICAL frame: 1 the broadcast shows the pitch, 0 it does not, NaN no
    broadcast frame maps here."""
    al = offsets()
    valid = broadcast_valid()
    out = np.full(len(_ball("tactical")), np.nan, dtype=np.float32)
    f = np.arange(len(valid))
    half = np.where(f < al["half2_from"], 1, 2)
    t = f + np.where(half == 1, al["offset"][1], al["offset"][2])
    ok = (t >= 0) & (t < len(out))
    out[t[ok]] = valid[ok]
    return out


# ---- training data: a broadcast made up from SoccerNet ----------------------------------------
#
# IDSSE and Metrica have no broadcast. To train on the signal we give every training match a
# made-up one: each stoppage and each stretch of play gets the pitch-on-screen sequence of a real
# SoccerNet stretch of the same kind (dead or live) and about the same length, stretched to fit.
# The model so learns from real directors how the pitch comes and goes, including the timing (the
# first ~2 s of a stoppage the main camera is still on). Only SoccerNet's own labels are used.

SN_DIR = ROOT / "data" / "external" / "soccernet" / "games"
PITCH_VIEW = {"Main camera center", "Main camera left", "Main camera right", "Main behind the goal",
              "Spider camera"}
STOPS = {"Ball out of play", "Foul", "Offside", "Goal"}
RESTARTS = {"Throw-in", "Corner", "Clearance", "Direct free-kick", "Indirect free-kick", "Kick-off",
            "Penalty"}                            # "Clearance" is SoccerNet's goal kick
MAX_STOPPAGE_S = 180
MIN_CAMERA_CHANGES = 250                          # per game: fewer means only replays are labelled (~200 games are full)


def _half_s(a: dict) -> tuple[int, float]:
    half, clock = a["gameTime"].split(" - ")
    return int(half), int(a["position"]) / 1000.0


def game(folder: Path):
    """Per half: (pitch view per second, dead per second), or None if the cameras are incomplete."""
    cams = json.load(open(folder / "Labels-cameras.json"))["annotations"]
    acts = json.load(open(folder / "Labels-v2.json"))["annotations"]
    if len(cams) < MIN_CAMERA_CHANGES:
        return None
    out = {}
    for half in (1, 2):
        c = sorted((_half_s(a)[1], a["label"] in PITCH_VIEW and a.get("replay") == "real-time")
                   for a in cams if _half_s(a)[0] == half)
        e = sorted((_half_s(a)[1], a["label"]) for a in acts if _half_s(a)[0] == half)
        if len(c) < 10 or len(e) < 10:
            continue
        start, end = int(c[0][0]), int(c[-1][0])
        pitch = np.zeros(end - start, dtype=bool)
        for (t, _), (nxt, p) in zip(c, c[1:]):          # a label describes the shot that ENDS at it
            pitch[int(t) - start:int(nxt) - start] = p
        dead = np.zeros(end - start, dtype=bool)
        restarts = np.array([t for t, lab in e if lab in RESTARTS])
        for t, lab in e:
            if lab in STOPS:
                nxt = restarts[restarts > t]
                if len(nxt) and nxt[0] - t <= MAX_STOPPAGE_S:
                    dead[max(int(t) - start, 0):int(nxt[0]) - start] = True
        out[half] = (pitch, dead)
    return out


K_CLOSEST = 10            # pick at random among the K SoccerNet stretches closest in length
WINDOWS_S = (2, 5, 10)    # signals: share of frames with the pitch on screen, before and after


def soccernet_stretches() -> dict:
    """{"dead": [...], "live": [...]}: per stretch, pitch on screen per second (bool array)."""
    cache = CACHE_DIR / "soccernet_stretches_games.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: list(z[k]) for k in ("dead", "live", "dead_game", "live_game")}
    out = {"dead": [], "live": [], "dead_game": [], "live_game": []}
    gi = -1
    for folder in sorted(p.parent for p in SN_DIR.rglob("Labels-cameras.json")):
        g = game(folder)
        if g:
            gi += 1
        for pitch, dead in (g or {}).values():
            d = np.diff(np.r_[0, dead.astype(np.int8), 0])
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                out["dead"].append(pitch[a:b])
                out["dead_game"].append(gi)
            live_edges = np.diff(np.r_[0, (~dead).astype(np.int8), 0])
            for a, b in zip(np.where(live_edges == 1)[0], np.where(live_edges == -1)[0]):
                if a > 0 and b < len(dead):            # whole stretches only, not cut by the half's edge
                    out["live"].append(pitch[a:b])
                    out["live_game"].append(gi)
    obj = lambda x: np.array(x + [None], dtype=object)[:-1]           # noqa: E731
    save_atomic(cache, lambda fh: np.savez_compressed(fh, dead=obj(out["dead"]), live=obj(out["live"]),
                                                      dead_game=np.array(out["dead_game"]),
                                                      live_game=np.array(out["live_game"])))
    return out


# A second source of real directors: SkillCorner open data (20 A-League broadcasts, MIT licence,
# fc_anomaly/skillcorner.py). Pitch off screen = no projection of the image corners; dead = outside
# SkillCorner's phases of play. Patterns at 25 fps (SoccerNet's are per second).

def skillcorner_stretches() -> dict:
    from fc_anomaly import skillcorner
    out = {"dead": [], "live": [], "dead_game": [], "live_game": []}
    for gi, mid in enumerate(skillcorner.match_ids()):
        for cam, dead in skillcorner.camera_and_dead(mid).values():
            for is_dead, pat in skillcorner.stretches(cam, dead):
                k = "dead" if is_dead else "live"
                out[k].append(pat)
                out[k + "_game"].append(gi)
    return out


SOURCES = {"soccernet": (soccernet_stretches, 1.0), "skillcorner": (skillcorner_stretches, float(FPS))}
# Both, each equally often: trained on one source and tested on
# the other's directors the gain holds (+0.02 either way); both together is as good as the best single
# source on either test, so the model meets an unseen director (BAR-ATH: LaLiga) with the widest pool.
DEFAULT_SOURCES = ("soccernet", "skillcorner")


@lru_cache(maxsize=8)
def library(sources: tuple = DEFAULT_SOURCES) -> dict:
    """Pitch-on-screen patterns of real dead and live stretches, with their length in seconds."""
    lib = {"dead": [], "live": [], "dead_s": [], "live_s": [], "dead_src": [], "live_src": []}
    for i, src in enumerate(sources):
        load, rate = SOURCES[src]
        st = load()
        for kind in ("dead", "live"):
            for pat in st[kind]:
                lib[kind].append(pat)
                lib[kind + "_s"].append(len(pat) / rate)
                lib[kind + "_src"].append(i)
    for k in ("dead_s", "live_s", "dead_src", "live_src"):
        lib[k] = np.array(lib[k])
    return lib


def simulate(dead: np.ndarray, segment: np.ndarray, seed: int) -> np.ndarray:
    """(T,) float32 made-up pitch on screen (1/0) for a match whose dead frames are `dead`."""
    rng = np.random.default_rng(seed)
    st = library()
    lengths = {k: st[k + "_s"] for k in ("dead", "live")}
    n_src = {k: len(np.unique(st[k + "_src"])) for k in ("dead", "live")}
    out = np.zeros(len(dead), dtype=np.float32)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        x = dead[idx]
        edges = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
        for a, b in zip(edges[:-1], edges[1:]):
            kind = "dead" if x[a] else "live"
            secs = (b - a) / FPS
            if n_src[kind] > 1:                        # several sources: each director pool equally often
                pool = np.flatnonzero(st[kind + "_src"] == rng.integers(n_src[kind]))
                pick = pool[rng.choice(np.argsort(np.abs(lengths[kind][pool] - secs))[:K_CLOSEST])]
            else:
                pick = rng.choice(np.argsort(np.abs(lengths[kind] - secs))[:K_CLOSEST])
            src = st[kind][pick]
            pos = np.minimum((np.arange(b - a) / (b - a) * len(src)).astype(int), len(src) - 1)
            out[idx[a:b]] = src[pos]
    return out




def signals(pitch: np.ndarray, segment: np.ndarray) -> pd.DataFrame:
    """Dead-ball signals from pitch on screen per frame (NaN where unknown): now, and the share of
    frames with the pitch on screen in the last and next 2, 5 and 10 s (within the half)."""
    cols = {"bc_pitch": pitch}
    for w in WINDOWS_S:
        cols[f"bc_past_{w}s"] = np.full(len(pitch), np.nan, dtype=np.float32)
        cols[f"bc_next_{w}s"] = np.full(len(pitch), np.nan, dtype=np.float32)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        ser = pd.Series(pitch[idx])
        for w in WINDOWS_S:
            n = int(w * FPS)
            cols[f"bc_past_{w}s"][idx] = ser.rolling(n, min_periods=1).mean().to_numpy()
            cols[f"bc_next_{w}s"][idx] = ser[::-1].rolling(n, min_periods=1).mean()[::-1].to_numpy()
    return pd.DataFrame(cols).astype(np.float32)


if __name__ == "__main__":
    al = offsets()
    print(f"offsets {al['offset']}, half 2 from broadcast frame {al['half2_from']}")
    from games import GAME, half_video_blocks
    p = pitch_on_screen()
    for half, a, b in half_video_blocks(GAME):
        x = p[a:b + 1]
        print(f"half {half} (tactical {a}..{b}): broadcast covers {np.isfinite(x).mean():.1%}, "
              f"pitch on screen {np.nanmean(x):.1%} of covered frames")

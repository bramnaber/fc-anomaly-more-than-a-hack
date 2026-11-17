"""Make clean IDSSE tracking look like the automatic BAR-ATH tracking, before training.

IDSSE comes from a corrected optical system: the ball on every frame, its path as sharp as it was
kicked. The BAR-ATH tracking loses the ball for long stretches and smooths its path. A model trained
on clean data would lean on what is simply absent at test time, so we take it away in training too.

Every number comes from EXTERNAL data, the three Metrica sample games (same provider as BAR-ATH;
`python -m fc_anomaly.degrade` prints them). Nothing is measured on BAR-ATH. The two ways of hiding
the ball the models train on (models.BALL_MODES):

- "ext_random":    the ball disappears in long stretches, independently of the game state: on
                   average 13.2 s hidden, 22.3 s visible (Metrica sample games: ~37% hidden).
- "ext_stoppages": the ball disappears mostly during stoppages, as in the Metrica sample games:
                   90% of dead frames and 2.7% of live frames hidden (live gaps 1.36 s on average).

In both: small Gaussian noise on every player position (5 cm, the order of magnitude of tracking
noise), and the ball is kept outside the pitch on 0.94% of the frames where it is seen (the provider
mostly loses the ball once it is out; IDSSE keeps tracking it). No player is hidden: the Metrica games
track all 22 players on every frame.

"keep" leaves a match as it is (the Metrica sample games as a test: the provider already hides the ball).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from fc_anomaly.data import FPS, Match

# Measured on the three Metrica sample games (metrica_input_stats).
BALL_HIDDEN_S = 13.2      # "ext_random": mean length of a stretch with the ball hidden ...
BALL_VISIBLE_S = 22.3     # ... and with the ball seen
DEAD_HIDDEN = 0.90        # "ext_stoppages": share of dead frames with the ball hidden ...
LIVE_HIDDEN = 0.027       # ... share of live frames ...
LIVE_GAP_S = 1.36         # ... and the mean length of a gap during play
OUTSIDE = 0.0094          # share of the seen ball outside the pitch
POS_NOISE_M = 0.05        # player position noise (m)
BALL_MODES = ("ext_random", "ext_stoppages", "keep")


def _markov_mask(T: int, hidden_mean: float, visible_mean: float, rng) -> np.ndarray:
    """Boolean array, True = hidden, alternating runs with geometric lengths."""
    out = np.zeros(T, dtype=bool)
    t = 0
    hidden = rng.random() < hidden_mean / (hidden_mean + visible_mean)
    while t < T:
        mean = hidden_mean if hidden else visible_mean
        run = max(1, int(rng.geometric(1.0 / mean)))
        out[t:t + run] = hidden
        t += run
        hidden = not hidden
    return out


def smooth_ball(ball: np.ndarray, segment: np.ndarray, seconds: float) -> np.ndarray:
    """The ball's path as a smoothing tracker would give it: every position becomes the mean of the
    positions within `seconds` around it (a centred moving average). Only within one stretch of
    visible ball in one half; at the ends of a stretch the window shrinks. seconds < 0.08: unchanged."""
    w = 2 * int(round(seconds * FPS / 2)) + 1                 # odd number of frames, centred
    if w < 3:
        return ball
    seen = ~np.isnan(ball[:, 0])
    change = np.r_[True, (seen[1:] != seen[:-1]) | (segment[1:] != segment[:-1])]
    run = np.cumsum(change)
    out = pd.DataFrame(ball).groupby(run).rolling(w, center=True, min_periods=1).mean()
    return out.to_numpy(dtype=ball.dtype)                     # groupby keeps the frame order


def degrade(m: Match, seed: int, ball: str, smooth_s: float = 0.0) -> Match:
    """ball: 'ext_random' / 'ext_stoppages' (training) or 'keep' (as it is).
    smooth_s: smooth the ball's path over this many seconds first (smooth_ball); 0 = as tracked."""
    if ball not in BALL_MODES:
        raise ValueError(f"unknown way of hiding the ball: {ball}")
    if smooth_s:
        m = dataclasses.replace(m, ball=smooth_ball(m.ball, m.segment, smooth_s))
    if ball == "keep":
        return m
    rng = np.random.default_rng(seed)
    T = m.n_frames
    players = m.players + rng.normal(0, POS_NOISE_M, m.players.shape).astype(np.float32)
    b = m.ball.copy()
    if ball == "ext_random":
        b[_markov_mask(T, BALL_HIDDEN_S * FPS, BALL_VISIBLE_S * FPS, rng)] = np.nan
    else:
        h = LIVE_GAP_S * FPS
        live_gaps = _markov_mask(T, h, h * (1 - LIVE_HIDDEN) / LIVE_HIDDEN, rng)
        b[live_gaps & ~m.dead] = np.nan
        d = np.diff(np.r_[0, m.dead.astype(int), 0])
        stops = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
        # chance per stoppage, so that DEAD_HIDDEN of ALL dead frames ends up hidden
        chance = min(1.0, DEAD_HIDDEN * m.dead.sum() / max(1, sum(e - a for a, e in stops)))
        for a, e in stops:
            if rng.random() < chance:
                b[a:e] = np.nan

    # The provider rarely keeps the ball once it leaves the pitch; IDSSE always does. Drop whole
    # outside-the-pitch runs at random until the outside share of the seen ball is OUTSIDE.
    out = _outside(b, m.pitch)
    n_out, n_in = out.sum(), (~np.isnan(b[:, 0]) & ~out).sum()
    if n_out:
        keep = min(1.0, OUTSIDE * n_in / (n_out * (1 - OUTSIDE)))
        d = np.diff(np.r_[0, out.astype(int), 0])
        for a, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if rng.random() > keep:
                b[a:e] = np.nan

    return Match(m.name + "+deg", players, m.team, b, m.segment, m.pitch, m.dead, m.owner)


def _outside(ball: np.ndarray, pitch) -> np.ndarray:
    from fc_anomaly.features import OUT_MARGIN
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN) |
                (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))


def metrica_input_stats() -> dict:
    """The numbers above, measured on the three Metrica sample games, pooled."""
    from fc_anomaly.metrica import load_metrica

    def runs(mask):
        d = np.diff(np.r_[0, mask.astype(int), 0])
        return list(np.where(d == -1)[0] - np.where(d == 1)[0])
    gaps, vis, live, n_out, n_seen, tracked = [], [], [], 0, 0, []
    for g in (1, 2, 3):
        m, _ = load_metrica(g)
        for s in np.unique(m.segment):
            sel = m.segment == s
            seen = ~np.isnan(m.ball[sel, 0])
            gaps += runs(~seen)
            vis += runs(seen)
            live += runs(~seen & ~m.dead[sel])
        seen = ~np.isnan(m.ball[:, 0])
        n_out += _outside(m.ball, m.pitch).sum()
        n_seen += seen.sum()
        tracked.append((~np.isnan(m.players[..., 0])).sum(1).mean())
    out = dict(ball_hidden_s=np.mean(gaps) / FPS, ball_visible_s=np.mean(vis) / FPS,
               players_tracked=float(np.mean(tracked)), outside=n_out / n_seen, live_gap_s=np.mean(live) / FPS)
    print({k: round(float(v), 4) for k, v in out.items()})
    return out


if __name__ == "__main__":
    metrica_input_stats()

"""The signals both models see, computed identically on any `Match` (BAR-ATH or IDSSE).

Model 1, dead ball (`dead_signals`).
Per-frame features for 'is the ball dead?', computed identically on any `Match`.

The idea, from football rather than from this match: when play stops, the players stop. They walk
back, wait for a throw-in, line up a wall, jog to the centre circle. When play restarts, they
accelerate. So the features describe the *collective* motion of the 20-odd tracked players, now,
over the last few seconds and over the next few seconds. They need no ball and no player identity,
which matters because the BAR-ATH ball is missing on 41% of graded frames and the track slots
switch identity.

Three measures of that motion are kept: the speed of the faster players (75th and 90th percentile)
and the share of players who are walking. The median speed, the share of sprinters and the number of
tracked players were dropped: leaving them out costs nothing.

The ball adds two things. Where it is seen: whether it is outside the pitch. And whether it is
seen at all, and how long the current gap lasts (`ball_seen_signals`). The tracking of BAR-ATH's
provider loses the ball mostly during stoppages; the training data teaches that relation only
through the way of hiding measured on the provider's own sample games (degrade.py), next to a way
that hides the ball regardless of the game state.

And how hard the players change pace (`acceleration_signals`): during play they keep speeding up,
braking and turning, in a stoppage they walk on at an even pace. The speed signals above do not show
that (a player jogging back is as fast as one following play). Measured against the match's own usual
level, so it means the same whatever the provider or the league.

Model 2, which team (`team_signals`).
Per-frame features for 'which team has the ball?', computed identically on any `Match`.

Three kinds of evidence, because the BAR-ATH ball is missing on 41% of graded frames:

1. Ball geometry, where the ball is seen: how close the nearest home and away player are, how
   many of each are around it, how far up the pitch it is. Carried forward and backward into gaps
   ('the last time we saw the ball, who was nearest; the next time, who is nearest; how long ago'),
   because a gap usually starts and ends with the same team.
2. Team shape, always available: the team with the ball pushes up, spreads wide and long and runs
   towards the opponent goal; the team without it compacts and drops. These are measured in each
   team's own attacking direction, which comes from where the team stands on average in that half
   (no labels needed).
3. The ball's own movement, where it is seen: which way it travels (`team_signals`), the touches
   that change its velocity (`touch_signals`) and when each team was last / is next right at it
   (`contact_signals`).
4. Four more groups (`carry_signals`, `block_signals`, `press_signals`, `long_signals`): the ball
   moving along with a player, the ball relative to each team's block, players closing in on the
   ball, and the distance and touch signals over longer windows. No group clears the noise on its
   own; together, in the team model only, they add about +0.009.
5. The ball's path as straight pieces (`ball_path_signals`): between two touches nobody steers the
   ball, so every kink in its path is a touch. Finds touches through a smoothed ball, where the
   rule of `touches` misses them.

Model 3, the second look (`context_signals`).
Not signals from the tracking but the first two models' own probabilities: now, a little earlier and
later, and averaged over a few seconds. One frame's 'home has the ball: 0.45' does not say whether
home really lost the ball or an opponent only touched it; the seconds around it do.

Every feature exists as a home value, an away value and/or a home-minus-away difference. Training
also sees every match with home and away swapped, so the model learns 'the team that ...' and not
'the home team'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fc_anomaly.data import FPS, Match

# Physical limits used to reject tracking glitches (identity switches show up as teleports).
MAX_PLAYER_SPEED = 12.0   # m/s; sprint world record is ~12.4 m/s
MAX_BALL_SPEED = 40.0     # m/s; hard shots peak around 35 m/s
OUT_MARGIN = 0.5          # m beyond a line before we call the ball out (tracking noise)
MIN_PLAYERS = 8           # fewer tracked players than this: collective stats are not trusted

# Window lengths in seconds. Short ones see the whistle and the restart kick; long ones see a
# substitution, an injury or a VAR check.
CENTRED = (1, 3, 8, 15)
SIDED = (3, 8)


def speed(pos: np.ndarray, segment: np.ndarray, half_span: int, cap: float) -> np.ndarray:
    """Central-difference speed over +-half_span frames, NaN across segments or above cap."""
    T = len(pos)
    v = np.full(pos.shape[:-1], np.nan, dtype=np.float32)
    a, b = pos[2 * half_span:], pos[:-2 * half_span]
    d = np.linalg.norm(a - b, axis=-1) / (2 * half_span / FPS)
    same = segment[2 * half_span:] == segment[:-2 * half_span]
    if d.ndim == 2:
        same = same[:, None]
    d = np.where(same, d, np.nan)
    d = np.where(d > cap, np.nan, d)
    v[half_span:T - half_span] = d
    return v


def rolling(s: pd.Series, seg: pd.Series, window: int, where: str) -> pd.Series:
    """Rolling mean within segments. where: 'centre', 'past' (ending at t), 'future' (from t)."""
    g = s.groupby(seg.values)
    minp = max(1, window // 4)
    if where == "centre":
        return g.transform(lambda x: x.rolling(window, center=True, min_periods=minp).mean())
    if where == "past":
        return g.transform(lambda x: x.rolling(window, min_periods=minp).mean())
    return g.transform(lambda x: x[::-1].rolling(window, min_periods=minp).mean()[::-1])



# ============================== model 1: dead ball ==============================

def dead_signals(m: Match) -> pd.DataFrame:
    """Signals for 'is the ball dead?' (model 1)."""
    T = m.n_frames
    length, width = m.pitch
    seg = pd.Series(m.segment)

    # ---- players: collective speed statistics per frame ------------------------------------
    v = speed(m.players, m.segment, half_span=3, cap=MAX_PLAYER_SPEED)        # (T, P)
    n = np.sum(~np.isnan(v), axis=1)
    ok = n >= MIN_PLAYERS
    with np.errstate(all="ignore"):
        p75 = np.where(ok, np.nanpercentile(v, 75, axis=1), np.nan)
        p90 = np.where(ok, np.nanpercentile(v, 90, axis=1), np.nan)
        slow = np.where(ok, np.nansum(v < 1.0, axis=1) / np.maximum(n, 1), np.nan)
        # how spread out the players are: set pieces bunch them, open play stretches them
        xs, ys = m.players[..., 0], m.players[..., 1]
        npos = np.sum(~np.isnan(xs), axis=1)
        okp = npos >= MIN_PLAYERS
        spread_x = np.where(okp, np.nanstd(xs, axis=1), np.nan)
        spread_y = np.where(okp, np.nanstd(ys, axis=1), np.nan)

    base = {"p75": p75, "p90": p90, "slow": slow}
    f = {}
    for k, arr in base.items():
        s = pd.Series(arr)
        for w in CENTRED:
            f[f"{k}_c{w}"] = rolling(s, seg, w * FPS, "centre")
        for w in SIDED:
            past = rolling(s, seg, w * FPS, "past")
            fut = rolling(s, seg, w * FPS, "future")
            f[f"{k}_p{w}"] = past
            f[f"{k}_f{w}"] = fut
            f[f"{k}_d{w}"] = fut - past
    f["spread_x_c3"] = rolling(pd.Series(spread_x), seg, 3 * FPS, "centre")
    f["spread_y_c3"] = rolling(pd.Series(spread_y), seg, 3 * FPS, "centre")

    # ---- ball, only where it is seen ---------------------------------------------------------
    bx, by = m.ball[:, 0], m.ball[:, 1]
    seen = ~np.isnan(bx)
    outside = seen & ((bx < -OUT_MARGIN) | (bx > length + OUT_MARGIN) |
                      (by < -OUT_MARGIN) | (by > width + OUT_MARGIN))
    out_s = pd.Series(np.where(seen, outside.astype(float), np.nan))
    for w in (1, 3):
        f[f"out_p{w}"] = rolling(out_s, seg, w * FPS, "past")
        f[f"out_f{w}"] = rolling(out_s, seg, w * FPS, "future")
    # Ball speed is deliberately left out. A parked ball is exactly still in the IDSSE data but
    # always jitters in the ATD (10th percentile 0.0 vs 1.3 m/s), so a model would learn a cue
    # that does not exist in BAR-ATH.

    return pd.DataFrame(f).astype(np.float32)


def ball_seen_signals(m: Match) -> pd.DataFrame:
    """Is the ball tracked? Now, around now, and how long is the gap we are in (seconds)."""
    seen = ~np.isnan(m.ball[:, 0])
    seg = pd.Series(m.segment)
    s = pd.Series(seen.astype(np.float32))
    f = {"seen": s}
    for w in (1, 3):
        f[f"seen_p{w}"] = rolling(s, seg, w * FPS, "past")
        f[f"seen_f{w}"] = rolling(s, seg, w * FPS, "future")
    # length of the run (gap or visible stretch) this frame is in, and where we are inside it
    change = np.r_[True, (seen[1:] != seen[:-1]) | (m.segment[1:] != m.segment[:-1])]
    run_id = np.cumsum(change) - 1
    run_len = np.bincount(run_id)[run_id]
    pos = np.arange(len(seen)) - np.flatnonzero(change)[run_id]
    cap = 30 * FPS
    f["gap_s"] = np.where(seen, 0, np.minimum(run_len, cap)) / FPS            # 0 when the ball is seen
    f["seen_run_s"] = np.where(seen, np.minimum(run_len, cap), 0) / FPS       # 0 inside a gap
    f["since_change_s"] = np.minimum(pos, cap) / FPS                          # time since the run began
    f["until_change_s"] = np.minimum(run_len - 1 - pos, cap) / FPS            # time until it ends
    return pd.DataFrame(f).astype(np.float32)


ACC_SMOOTH_S = 1.0   # velocities are averaged over 1 s: tracking noise goes, a player's own change of pace stays
ACC_SPAN = 5         # frames either side: the change of that velocity over 0.4 s


def acceleration_signals(m: Match, play: np.ndarray | None = None) -> pd.DataFrame:
    """Signals for 'is the ball dead?' from how hard the players change pace.

    During play the players keep speeding up, braking and turning; in a stoppage they walk on at an even pace.
    The speed signals of `dead_signals` do not show that: a player jogging back to his position is as fast
    as one following play.

    Built to mean the same in every tracking:
    - velocities come from the raw positions with the speed cap (a track that jumps to another player gives no
      velocity instead of a false one) and are averaged over ACC_SMOOTH_S, so frame-to-frame noise is gone;
    - over the players the median and the 75th percentile are taken, so a few bad tracks do not count;
    - every value is divided by this match's own median (over `play`, the frames of the match itself; default
      all frames), so it reads 'calmer / wilder than usual in this match' and does not depend on how smooth a
      provider's positions are or how intense a league plays. Uses the match's tracking only, no labels.
    The windows are those of `dead_signals`.
    """
    seg = pd.Series(m.segment)
    v = velocity(m.players, m.segment, 3, MAX_PLAYER_SPEED)                       # (T, P, 2); NaN on a jump
    T, n = v.shape[:2]
    w = int(round(ACC_SMOOTH_S * FPS))
    v = pd.DataFrame(v.reshape(T, -1)).groupby(m.segment, sort=False).transform(
        lambda x: x.rolling(w, center=True, min_periods=w // 2).mean()).to_numpy(np.float32).reshape(T, n, 2)
    acc = np.full((T, n), np.nan, dtype=np.float32)
    acc[ACC_SPAN:-ACC_SPAN] = np.linalg.norm(v[2 * ACC_SPAN:] - v[:-2 * ACC_SPAN], axis=-1) / (2 * ACC_SPAN / FPS)
    acc[ACC_SPAN:-ACC_SPAN][m.segment[2 * ACC_SPAN:] != m.segment[:-2 * ACC_SPAN]] = np.nan
    ok = np.sum(~np.isnan(acc), axis=1) >= MIN_PLAYERS
    play = np.ones(T, dtype=bool) if play is None else play
    f = {}
    with np.errstate(all="ignore"):
        for name, q in (("acc50", 50), ("acc75", 75)):
            a = np.where(ok, np.nanpercentile(acc, q, axis=1), np.nan)
            s = pd.Series(a / np.nanmedian(a[play]))
            for c in CENTRED:
                f[f"{name}_c{c}"] = rolling(s, seg, c * FPS, "centre")
            for c in SIDED:
                past, fut = rolling(s, seg, c * FPS, "past"), rolling(s, seg, c * FPS, "future")
                f[f"{name}_p{c}"], f[f"{name}_f{c}"], f[f"{name}_d{c}"] = past, fut, fut - past
    return pd.DataFrame(f).astype(np.float32).add_prefix("mo_")


# ============================== model 2: which team =============================

CLIP = 15.0   # m; beyond this the nearest-player distance carries no extra information


def attack_sign(m: Match) -> np.ndarray:
    """Per frame, +1 if the home team attacks towards +x in that segment, else -1.

    A team defends the goal on the side where it stands on average over a half.
    """
    sign = np.ones(m.n_frames, dtype=np.float32)
    for s in np.unique(m.segment):
        sel = m.segment == s
        with np.errstate(all="ignore"):
            hx = np.nanmean(m.players[sel][:, m.team == 0, 0])
            ax = np.nanmean(m.players[sel][:, m.team == 1, 0])
        sign[sel] = 1.0 if (np.isnan(hx) or np.isnan(ax) or hx <= ax) else -1.0
    return sign


def _ffill_with_age(x: np.ndarray, segment: np.ndarray):
    """Last non-NaN value and frames since it, per segment (NaN / large if none yet)."""
    s = pd.Series(x)
    g = segment
    idx = pd.Series(np.where(~np.isnan(x), np.arange(len(x)), np.nan))
    last_val = s.groupby(g).ffill().to_numpy()
    last_idx = idx.groupby(g).ffill().to_numpy()
    age = np.arange(len(x)) - last_idx
    return last_val, np.where(np.isnan(age), 10_000, age)


def team_signals(m: Match) -> pd.DataFrame:
    """Signals for 'which team has the ball?' (model 2)."""
    T = m.n_frames
    length, width = m.pitch
    seg = pd.Series(m.segment)
    sign = attack_sign(m)
    home, away = m.team == 0, m.team == 1
    f = {}

    # ---- ball geometry ---------------------------------------------------------------------
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)              # (T, P)
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(home, d, np.nan), axis=1)
        da = np.nanmin(np.where(away, d, np.nan), axis=1)
    dh, da = np.minimum(dh, CLIP), np.minimum(da, CLIP)
    diff = dh - da                                                          # < 0: home closer
    f["d_home"], f["d_away"], f["d_diff"] = dh, da, diff
    for r in (2.0, 5.0):
        nh = np.sum(home & (d < r), axis=1).astype(np.float32)
        na = np.sum(away & (d < r), axis=1).astype(np.float32)
        seen = ~np.isnan(m.ball[:, 0])
        f[f"n_home_{r:.0f}m"] = np.where(seen, nh, np.nan)
        f[f"n_away_{r:.0f}m"] = np.where(seen, na, np.nan)
    # ball position towards the away goal (home attacking frame), 0..1
    bx = m.ball[:, 0] / length
    f["ball_x_home"] = np.where(sign > 0, bx, 1 - bx)

    ds = pd.Series(diff)
    for w, where in ((0.4, "centre"), (1, "centre"), (3, "centre"), (1, "past"), (1, "future"),
                     (3, "past"), (3, "future")):
        f[f"d_diff_{where[0]}{w}"] = rolling(ds, seg, max(1, int(w * FPS)), where)

    # into the gaps: last and next value seen, and how far away in time
    last, age = _ffill_with_age(diff, m.segment)
    rev = slice(None, None, -1)
    nxt, ahead = _ffill_with_age(diff[rev], m.segment[rev])
    f["d_diff_last"], f["d_diff_next"] = last, nxt[rev]
    f["age_last_s"] = np.minimum(age, 60 * FPS) / FPS
    f["age_next_s"] = np.minimum(ahead[rev], 60 * FPS) / FPS

    # ---- team shape, in each team's own attacking direction --------------------------------
    x = m.players[..., 0]
    y = m.players[..., 1]
    v = speed(m.players, m.segment, half_span=3, cap=MAX_PLAYER_SPEED)
    # velocity along x (towards +x), from the same central difference
    vx = np.full_like(x, np.nan)
    vx[3:-3] = (x[6:] - x[:-6]) / (6 / FPS)
    vx = np.where(np.isnan(v), np.nan, vx)
    for name, mask, s_team in (("home", home, sign), ("away", away, -sign)):
        xs, ys, vxs = x[:, mask], y[:, mask], vx[:, mask]
        n = np.sum(~np.isnan(xs), axis=1)
        ok = n >= MIN_PLAYERS // 2
        with np.errstate(all="ignore"):
            cx = np.nanmean(xs, axis=1)
            fwd = np.where(s_team > 0, cx / length, 1 - cx / length)               # 0 own goal
            ln = np.nanstd(xs, axis=1)
            wd = np.nanstd(ys, axis=1)
            run = np.nanmean(vxs, axis=1) * s_team                                   # m/s forward
            top = np.where(s_team > 0, np.nanmax(xs, axis=1), length - np.nanmin(xs, axis=1)) / length
        for k, arr in (("fwd", fwd), ("len", ln), ("wid", wd), ("run", run), ("top", top)):
            f[f"{name}_{k}"] = np.where(ok, arr, np.nan)

    # ---- ball direction, where the ball is seen and actually travelling ----------------------
    # A ball moving towards the away goal is usually being played by the home team. Only used
    # when it moves faster than a jog, so the tracker's jitter on a slow ball carries no weight.
    bvx = np.full(T, np.nan, dtype=np.float32)
    bvx[2:-2] = (m.ball[4:, 0] - m.ball[:-4, 0]) / (4 / FPS)
    bv = speed(m.ball, m.segment, half_span=2, cap=40.0)
    bvx = np.where(bv > 3.0, bvx * sign, np.nan)
    f["ball_vx_home"] = bvx

    out = pd.DataFrame(f)
    bs = pd.Series(bvx)
    for w, where in ((1, "past"), (1, "future"), (3, "centre")):
        out[f"ball_vx_{where[0]}{w}"] = rolling(bs, seg, w * FPS, where)
    for k in ("fwd", "len", "wid", "run", "top"):
        out[f"diff_{k}"] = out[f"home_{k}"] - out[f"away_{k}"]
        s = out[f"diff_{k}"]
        for w in (1, 3):
            out[f"diff_{k}_c{w}"] = rolling(s, seg, w * FPS, "centre")
            # change around this frame: a turnover shows as one team suddenly opening up / pushing
            past = rolling(s, seg, w * FPS, "past")
            fut = rolling(s, seg, w * FPS, "future")
            out[f"diff_{k}_p{w}"], out[f"diff_{k}_f{w}"] = past, fut
            out[f"diff_{k}_d{w}"] = fut - past
    out["d_diff_d1"] = out["d_diff_f1"] - out["d_diff_p1"]
    out["d_diff_d3"] = out["d_diff_f3"] - out["d_diff_p3"]
    return out.astype(np.float32)


CONTACT_R = 1.0     # m; a player this close to the ball can play it
CONTACT_CAP_S = 10  # s; longer ago than this is simply 'long ago'


def contact_signals(m: Match) -> pd.DataFrame:
    """Per team: how long ago did one of its players last have the ball at his feet, and how long
    until one does again? In the training labels a team's possession starts at exactly such a
    moment, so these tell the model who had the ball last and who gets it next."""
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)
    f = {}
    for name, k in (("home", 0), ("away", 1)):
        with np.errstate(all="ignore"):
            close = np.nanmin(np.where(m.team == k, d, np.nan), axis=1) < CONTACT_R
        mark = np.where(close, 1.0, np.nan)
        _, age = _ffill_with_age(mark, m.segment)
        _, ahead = _ffill_with_age(mark[::-1], m.segment[::-1])
        f[f"since_{name}"] = np.minimum(age, CONTACT_CAP_S * FPS) / FPS
        f[f"until_{name}"] = np.minimum(ahead[::-1], CONTACT_CAP_S * FPS) / FPS
    f["since_diff"] = f["since_home"] - f["since_away"]
    f["until_diff"] = f["until_home"] - f["until_away"]
    return pd.DataFrame(f).astype(np.float32)


def swapped(m: Match) -> Match:
    """The same match with home and away exchanged (labels too)."""
    owner = None if m.owner is None else np.where(m.owner >= 0, 1 - m.owner, -1).astype(np.int8)
    return Match(m.name + "+swap", m.players, (1 - m.team).astype(np.int8), m.ball, m.segment,
                 m.pitch, m.dead, owner)


# ============================== model 2: touches ==================================================
# A touch (pass, reception, shot, deflection) shows as a sudden change of the ball's velocity with a
# player right next to it. Thresholds from physics and football, not from a score: even a soft pass
# changes the ball's speed by several m/s within a quarter of a second, and a player touching the
# ball is within about 1.5 m of the tracked ball position.
TOUCH_DV = 5.0          # m/s change of ball velocity across TOUCH_SPAN frames on each side
TOUCH_SPAN = 3          # frames (0.12 s each side)
TOUCH_R = 1.5           # m from the nearest player
TOUCH_GAP_S = 0.3       # two touches are at least this far apart


def touches(m: Match):
    """Frames of detected touches and the side that touched (+1 home, -1 away)."""
    T = m.n_frames
    v = np.full((T, 2), np.nan, dtype=np.float32)
    v[2:-2] = (m.ball[4:] - m.ball[:-4]) / (4 / FPS)
    same = np.r_[np.zeros(2, bool), m.segment[4:] == m.segment[:-4], np.zeros(2, bool)]
    v[~same] = np.nan
    k = TOUCH_SPAN
    dv = np.full(T, np.nan, dtype=np.float32)
    dv[k:-k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1)
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)
    with np.errstate(all="ignore"):
        nearest = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        dmin = np.nanmin(d, axis=1)
        cand = (dv > TOUCH_DV) & (dmin <= TOUCH_R)
    frames, sides = [], []
    gap = int(TOUCH_GAP_S * FPS)
    dd = np.diff(np.r_[0, cand.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
        t = a + int(np.nanargmax(dv[a:b]))                     # strongest change in the run
        if frames and t - frames[-1] < gap:
            continue
        frames.append(t)
        sides.append(1 if m.team[nearest[t]] == 0 else -1)
    return np.array(frames, dtype=np.int64), np.array(sides, dtype=np.int8)


def touch_signals(m: Match, found=None) -> pd.DataFrame:
    """Per frame: who touched last / touches next, how long ago / until, and touch counts nearby.
    found: (frames, sides) of touches found another way (ball_path_signals); default: `touches`."""
    T = m.n_frames
    frames, sides = touches(m) if found is None else found
    side = np.full(T, np.nan, dtype=np.float32)
    side[frames] = sides
    seg = pd.Series(m.segment)
    last, age = _ffill_with_age(side, m.segment)
    rev = slice(None, None, -1)
    nxt, ahead = _ffill_with_age(side[rev], m.segment[rev])
    f = {"touch_last": last, "touch_age_s": np.minimum(age, 30 * FPS) / FPS,
         "touch_next": nxt[rev], "touch_ahead_s": np.minimum(ahead[rev], 30 * FPS) / FPS}
    # the last two touches by the same side: a completed pass or a controlled touch
    prev = np.r_[np.nan, sides[:-1]].astype(np.float32)
    pair = np.full(T, np.nan, dtype=np.float32)
    pair[frames] = np.where(prev == sides, sides, 0)
    f["touch_pair"], _ = _ffill_with_age(pair, m.segment)
    for name, s in (("home", side == 1), ("away", side == -1)):
        x = pd.Series(s.astype(np.float32))
        for w, where in ((3, "past"), (3, "future")):
            f[f"touches_{name}_{where[0]}{w}"] = rolling(x, seg, w * FPS, where) * w * FPS
    return pd.DataFrame(f).astype(np.float32)


# ============================== model 2: four more groups =========================================
# Every number below comes from football, not from a score.
FEET_R, FEET_DV = 2.5, 2.5     # "at his feet and moving along": within 2.5 m, speeds within 2.5 m/s
PRESS_R = 20.0                 # m; players this close to the ball count for pressing


def velocity(pos: np.ndarray, segment: np.ndarray, k: int, cap: float) -> np.ndarray:
    """Velocity vectors (m/s) by central difference over +-k frames; NaN across halves or above cap."""
    v = np.full(pos.shape, np.nan, dtype=np.float32)
    d = (pos[2 * k:] - pos[:-2 * k]) / (2 * k / FPS)
    same = (segment[2 * k:] == segment[:-2 * k]).reshape((-1,) + (1,) * (pos.ndim - 1))
    with np.errstate(invalid="ignore"):
        ok = same & (np.linalg.norm(d, axis=-1, keepdims=True) <= cap)
    v[k:len(pos) - k] = np.where(ok, d, np.nan)
    return v


class Scene:
    """What the four groups share, computed once per match."""

    def __init__(self, m: Match):
        self.m, self.T = m, m.n_frames
        self.seg = pd.Series(m.segment)
        self.sign = attack_sign(m)                                           # +1: home attacks towards +x
        self.seen = ~np.isnan(m.ball[:, 0])
        self.d = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)     # (T, P) player to ball
        self.vp = velocity(m.players, m.segment, 3, MAX_PLAYER_SPEED)        # (T, P, 2)
        self.vb = velocity(m.ball, m.segment, 2, MAX_BALL_SPEED)             # (T, 2)
        self.rows = np.arange(self.T)
        self.near, self.dmin = {}, {}
        for name, mask, _ in self.teams():
            dt = np.where(mask & ~np.isnan(self.d), self.d, np.inf)
            self.near[name] = np.argmin(dt, axis=1)                          # the team's nearest player
            dm = dt[self.rows, self.near[name]]
            self.dmin[name] = np.where(np.isfinite(dm), dm, np.nan)          # and his distance to the ball

    def teams(self):
        """(name, which players, +1 / -1 per frame: the direction this team attacks)."""
        return (("home", self.m.team == 0, self.sign), ("away", self.m.team == 1, -self.sign))

    def windows(self, f: dict, key: str, arr, spans) -> None:
        s = pd.Series(arr)
        for w, where in spans:
            f[f"{key}_{where[0]}{w}"] = rolling(s, self.seg, w * FPS, where)


def carry_signals(sc: Scene) -> dict:
    """The ball moving along with a player (a dribble): the ball's velocity minus the velocity of each
    team's nearest player, and whether the ball is at his feet and moving along."""
    f = {}
    for name, _, _ in sc.teams():
        rel = np.linalg.norm(sc.vb - sc.vp[sc.rows, sc.near[name]], axis=1)   # NaN if either is unknown
        rel = np.where(np.isnan(sc.dmin[name]), np.nan, rel)
        f[f"rel_{name}"] = rel
        with np.errstate(invalid="ignore"):
            feet = ((sc.dmin[name] < FEET_R) & (rel < FEET_DV)).astype(np.float32)
        f[f"feet_{name}"] = np.where(sc.seen, feet, np.nan)
    f["rel_diff"] = f["rel_home"] - f["rel_away"]
    f["feet_diff"] = f["feet_home"] - f["feet_away"]
    sc.windows(f, "rel_diff", f["rel_diff"], ((1, "centre"), (1, "past"), (1, "future")))
    sc.windows(f, "feet_diff", f["feet_diff"], ((1, "centre"), (1, "past"), (1, "future"), (3, "past"), (3, "future")))
    return f


def block_signals(sc: Scene) -> dict:
    """The ball relative to each team's block, in that team's own attacking direction: how far it is
    ahead of the team's centre and of its last line, how many of the team are goal-side of it, and how
    many are around it. The side without the ball stands between the ball and its own goal."""
    f = {}
    bx, by = sc.m.ball[:, 0], sc.m.ball[:, 1]
    x, y = sc.m.players[..., 0], sc.m.players[..., 1]
    for name, mask, s in sc.teams():
        xs, ys = x[:, mask], y[:, mask]
        n = np.maximum(np.sum(~np.isnan(xs), axis=1), 1)
        with np.errstate(all="ignore"):
            cx, cy = np.nanmean(xs, axis=1), np.nanmean(ys, axis=1)
            f[f"ahead_{name}"] = (bx - cx) * s                      # m the ball is further up than the team's centre
            f[f"cdist_{name}"] = np.hypot(bx - cx, by - cy)
            up = (xs - bx[:, None]) * s[:, None]                    # > 0: this player is ahead of the ball
            f[f"goalside_{name}"] = np.where(sc.seen, np.sum(up < 0, axis=1) / n, np.nan)
            line = np.sort(xs * s[:, None], axis=1)[:, 1]           # the last outfield line (rearmost = keeper)
            f[f"to_line_{name}"] = bx * s - line                    # m the ball is in front of that line
            for r in (10, 20):
                f[f"n{r}_{name}"] = np.where(sc.seen, np.sum(sc.d[:, mask] < r, axis=1), np.nan)
    for k in ("ahead", "cdist", "goalside", "to_line", "n10", "n20"):
        f[f"{k}_diff"] = f[f"{k}_home"] - f[f"{k}_away"]
    return f


def press_signals(sc: Scene) -> dict:
    """Players closing in on the ball: the speed towards it of all of a team's players nearby and of
    its nearest player, and how fast that nearest distance shrinks."""
    f = {}
    to_ball = sc.m.ball[:, None, :] - sc.m.players
    with np.errstate(all="ignore"):
        radial = np.sum(sc.vp * to_ball, axis=-1) / np.maximum(sc.d, 0.1)     # m/s towards the ball
        for name, mask, _ in sc.teams():
            f[f"all_{name}"] = np.nanmean(np.where(mask & (sc.d < PRESS_R), radial, np.nan), axis=1)
            f[f"first_{name}"] = np.where(np.isnan(sc.dmin[name]), np.nan, radial[sc.rows, sc.near[name]])
            dm = sc.dmin[name]
            shrink = np.full(sc.T, np.nan, dtype=np.float32)
            shrink[3:-3] = -(dm[6:] - dm[:-6]) / (6 / FPS)
            shrink[3:-3][sc.m.segment[6:] != sc.m.segment[:-6]] = np.nan
            f[f"shrink_{name}"] = shrink
    for k in ("all", "first", "shrink"):
        f[f"{k}_diff"] = f[f"{k}_home"] - f[f"{k}_away"]
        sc.windows(f, f"{k}_diff", f[f"{k}_diff"], ((1, "centre"),))
    return f


def long_signals(sc: Scene) -> dict:
    """Two existing signals over longer windows: which team is nearer to the ball over 6 and 12 s, and
    home minus away touches in the last / next 8 and 20 s."""
    f = {}
    diff = np.minimum(sc.dmin["home"], CLIP) - np.minimum(sc.dmin["away"], CLIP)
    sc.windows(f, "d_diff", diff, ((6, "centre"), (12, "centre"), (6, "past"), (6, "future")))
    frames, sides = touches(sc.m)
    side = np.zeros(sc.T, dtype=np.float32)
    side[frames] = sides
    s = pd.Series(side)
    for w in (8, 20):
        f[f"touch_diff_p{w}"] = rolling(s, sc.seg, w * FPS, "past") * w * FPS
        f[f"touch_diff_f{w}"] = rolling(s, sc.seg, w * FPS, "future") * w * FPS
    return f


EXTRA_GROUPS = {"carry": carry_signals, "block": block_signals, "press": press_signals, "long": long_signals}


def extra_team_signals(m: Match) -> pd.DataFrame:
    """The four groups' columns, named <group>_<signal>."""
    sc = Scene(m)
    out = {}
    for g, fn in EXTRA_GROUPS.items():
        for k, v in fn(sc).items():
            out[f"{g}_{k}"] = np.asarray(v, dtype=np.float32)
    return pd.DataFrame(out)


# ============================== model 2: the ball's path as straight pieces =======================
# Between two touches nobody steers the ball: seen from above it travels in a straight line, only
# slowing down. So its path is a chain of straight pieces and every kink is a touch. `touches` looks
# for the kink within 3 frames and wants a player within TOUCH_R of the ball at that moment; a tracker
# that smooths the ball rounds the kink off, and the rounded corner passes the player at a distance.
# Here one straight line is fitted through the ball before every frame and one after it, each over
# `window` frames and each starting `guard` frames away, so the rounded part is skipped. The kink is
# where the two lines meet, and the player nearest to THAT point touched the ball. Three sizes: a
# short guard tells two quick touches apart on a sharp ball, a long one sees through 0.6 s of
# smoothing. Thresholds are those of `touches`.
PATH_SCALES = ((1, 5), (4, 6), (8, 8))      # (guard, window) in frames: 0.04 / 0.16 / 0.32 s skipped each side


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """x[t + k] at t (NaN outside)."""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def two_lines(ball: np.ndarray, segment: np.ndarray, guard: int, window: int):
    """Least-squares lines through the ball before and after every frame: (v_before, v_after) in m/s
    and (p_before, p_after), each line extended to the frame itself. NaN where a window is not
    completely tracked or crosses a half."""
    k = np.arange(guard, guard + window + 1, dtype=np.float64)
    slope = (k - k.mean()) / ((k - k.mean()) ** 2).sum()            # weights of the after-line's slope
    level = 1 / len(k) - k.mean() * slope                           # weights of its value at the frame
    out = [np.full(ball.shape, np.nan) for _ in range(4)]
    for s in np.unique(segment):
        sel = segment == s
        b = ball[sel].astype(np.float64)
        va, vb, pa, pb = (np.zeros(b.shape) for _ in range(4))
        for kj, sj, lj in zip(k.astype(int), slope, level):
            after, before = _shift(b, kj), _shift(b, -kj)
            va += sj * after
            vb -= sj * before
            pa += lj * after
            pb += lj * before
        for o, x in zip(out, (vb * FPS, va * FPS, pb, pa)):
            o[sel] = x
    return out


def kinks(m: Match, guard: int, window: int):
    """(frames, sides, strength per frame): the kinks of the ball's path with a player at the kink."""
    v0, v1, p0, p1 = two_lines(m.ball, m.segment, guard, window)
    dv = np.linalg.norm(v1 - v0, axis=1)
    gap = np.linalg.norm(p1 - p0, axis=1)                           # 0 where the two lines meet at this frame
    spot = (p0 + p1) / 2
    d = np.linalg.norm(m.players - spot[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    nearest, dmin = np.argmin(d, axis=1), d.min(axis=1)
    with np.errstate(invalid="ignore"):
        cand = (dv > TOUCH_DV) & (dmin <= TOUCH_R)
        # a kink x frames away leaves a gap of strength * x / FPS: prefer the frame where the lines meet
        sharp = dv - gap * FPS / max(guard, 1)
    frames, sides = [], []
    min_gap = int(TOUCH_GAP_S * FPS)
    dd = np.diff(np.r_[0, cand.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
        t = a + int(np.nanargmax(sharp[a:b]))
        if frames and t - frames[-1] < min_gap:
            continue
        frames.append(t)
        sides.append(1 if m.team[nearest[t]] == 0 else -1)
    return np.array(frames, dtype=np.int64), np.array(sides, dtype=np.int8), dv.astype(np.float32)


def ball_path_signals(m: Match) -> pd.DataFrame:
    """Per size: the touch signals (touch_signals) from the kinks, and the kink strength now and the
    strongest in the last / next second."""
    out = []
    seg = pd.Series(m.segment)
    for guard, window in PATH_SCALES:
        frames, sides, dv = kinks(m, guard, window)
        F = touch_signals(m, (frames, sides))
        s = pd.Series(dv)
        F["dv"] = dv
        F["dv_p1"] = s.groupby(seg.values).transform(lambda x: x.rolling(FPS, min_periods=1).max())
        F["dv_f1"] = s.groupby(seg.values).transform(lambda x: x[::-1].rolling(FPS, min_periods=1).max()[::-1])
        out.append(F.add_prefix(f"bb{guard}_"))
    return pd.concat(out, axis=1).astype(np.float32)


# ============================== the two tables =================================================

def dead_table(m: Match, play: np.ndarray | None = None) -> pd.DataFrame:
    """Every signal of the dead-ball model. play: the frames of the match itself (acceleration_signals);
    default all frames, which is right for a training match (it holds the two halves only)."""
    return pd.concat([dead_signals(m), ball_seen_signals(m), acceleration_signals(m, play)], axis=1)


def team_table(m: Match) -> pd.DataFrame:
    """Every signal of the team model."""
    return pd.concat([team_signals(m), touch_signals(m), contact_signals(m), extra_team_signals(m),
                      ball_path_signals(m)], axis=1)


# ============================== model 3: the second look ========================

CONTEXT_OFFSETS = (5, 12, 25, 50, 100)    # frames: 0.2, 0.5, 1, 2 and 4 s earlier and later
CONTEXT_WINDOWS = (1, 3, 6)               # seconds: mean over the past / the coming stretch


def context_signals(p_dead: np.ndarray, p_home: np.ndarray, segment: np.ndarray) -> pd.DataFrame:
    """The dead-ball and team models' probabilities around every frame (within one half)."""
    seg = pd.Series(segment)
    f = {}
    for name, p, offsets, windows in (("home", p_home, CONTEXT_OFFSETS, CONTEXT_WINDOWS),
                                      ("dead", p_dead, CONTEXT_OFFSETS[1:4], (3,))):
        s = pd.Series(p)
        f[f"{name}_now"] = s
        for o in offsets:
            f[f"{name}_{o}_earlier"] = s.groupby(segment).shift(o)
            f[f"{name}_{o}_later"] = s.groupby(segment).shift(-o)
        for w in windows:
            f[f"{name}_past{w}s"] = rolling(s, seg, w * FPS, "past")
            f[f"{name}_next{w}s"] = rolling(s, seg, w * FPS, "future")
    return pd.DataFrame(f).astype(np.float32)


def mirrored(context: pd.DataFrame) -> pd.DataFrame:
    """The same context with home and away exchanged: every 'home has the ball' probability p becomes 1 - p."""
    out = context.copy()
    cols = [c for c in out.columns if c.startswith("home_")]
    out[cols] = 1 - out[cols]
    return out

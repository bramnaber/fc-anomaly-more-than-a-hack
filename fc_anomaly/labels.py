"""Training labels for IDSSE under the challenge's definition of possession.

CHALLENGE.md: possession changes only when the other team gains *controlled* possession. It names
two signs of control: "a player keeps the ball for a few seconds" and passing. The IDSSE tracking
labels (`owner`) switch on every touch, so we keep a switch only when the new team shows control:

    the new team completes at least one pass  OR  keeps the ball for at least CONTROL_HOLD_S

A run of the other team that shows neither (a touch, a deflection, a header won without control)
is given back to the team that had the ball. The first run after a stoppage is the restart and
always counts. Dead frames stay dead.

Passes come from the IDSSE event data. Events are logged by hand and run up to about a second off
the tracking, so per half we shift them to where the ball is closest to the passer, and an event
counts for a run if it falls within EVENT_TOLERANCE_S of it. All of this uses IDSSE only.
"""
from __future__ import annotations

import itertools

import numpy as np

from fc_anomaly.data import CACHE_DIR, FPS, idsse_path, load_idsse, save_atomic

CONTROL_HOLD_S = 1.5      # "keeps the ball for a few seconds" (CHALLENGE.md): of 1 / 1.5 / 2 / 3 s, the hold
                          # that loses least on external tests if the organisers meant another one
EVENT_TOLERANCE_S = 1.0   # hand-logged event times are about a second off the tracking
MAX_SHIFT_S = 6.0         # search range for the per-half event offset


def complete_passes(match_id: str) -> np.ndarray:
    """(n, 2) int array: tracking frame and team (0 home, 1 away) of every completed pass.

    The event times are hand-logged and offset from the tracking by a few seconds per half; the offset
    is found on the clean tracking (where the ball is always seen) as the shift that puts the passer
    closest to the ball."""
    cache = CACHE_DIR / f"{match_id}.passes.v2.npy"
    if cache.exists():
        return np.load(cache)
    m = load_idsse(match_id)

    from kloppy import sportec
    ev = sportec.load_event(event_data=str(idsse_path(match_id, "event")),
                            meta_data=str(idsse_path(match_id, "meta")), coordinates="kloppy")
    home, away = ev.metadata.teams
    slot = {p.player_id: i for i, p in enumerate(list(home.players) + list(away.players))}
    first = {p: int(np.argmax(m.segment == p)) for p in np.unique(m.segment)}
    passes = [e for e in ev.events if e.event_type.name == "PASS" and e.player is not None]

    def frame(e, shift):
        return first[e.period.id] + int(round(e.timestamp.total_seconds() * FPS)) + shift

    shift = {}
    for half in first:
        best = (np.inf, 0)
        for s in range(-int(MAX_SHIFT_S * FPS), int(MAX_SHIFT_S * FPS) + 1, 5):
            d = [np.linalg.norm(m.players[t, slot[e.player.player_id]] - m.ball[t])
                 for e in passes if e.period.id == half and 0 <= (t := frame(e, s)) < m.n_frames]
            best = min(best, (float(np.nanmedian(d)), s))
        shift[half] = best[1]

    out = np.array([(frame(e, shift[e.period.id]), 0 if e.team.team_id == home.team_id else 1)
                    for e in passes if e.result is not None and e.result.name == "COMPLETE"],
                   dtype=np.int64).reshape(-1, 2)
    save_atomic(cache, lambda fh: np.save(fh, out))
    return out


def to_challenge_definition(owner: np.ndarray, dead: np.ndarray, segment: np.ndarray,
                            passes: np.ndarray) -> np.ndarray:
    """Per-frame labels 0=H, 1=A, 2=D under the challenge's possession rules."""
    lab = np.where(dead | (owner < 0), 2, owner).astype(np.int8)    # no owner given: not in play
    out = lab.copy()
    tol = int(EVENT_TOLERANCE_S * FPS)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        x = out[idx]
        pos, current = 0, None
        for k, group in itertools.groupby(lab[idx]):
            n = len(list(group))
            a, b = idx[pos], idx[pos] + n                              # frames of this run
            if k == 2:
                current = None                                         # stoppage
            elif current is None or k == current:
                current = k                                            # restart, or same team
            else:
                passed = np.any((passes[:, 1] == k) & (passes[:, 0] >= a - tol) & (passes[:, 0] < b + tol))
                if passed or n >= CONTROL_HOLD_S * FPS:
                    current = k                                        # controlled: a real change
                else:
                    x[pos:pos + n] = current                           # no control: no change
            pos += n
        out[idx] = x
    return out

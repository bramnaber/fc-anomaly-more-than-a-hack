"""From per-frame probabilities to labels, without flicker: one decision for H, A and D, then the
challenge's rule.

`three_states` finds the most likely sequence of labels given the two models' probabilities, where
every change of team costs a fixed penalty and every edge of a stoppage costs another. Two numbers,
both chosen on IDSSE (train.py).

`control_rule` then applies the challenge's definition: the team model says which team touched the
ball last, and a touch is only a change of possession if that team keeps the ball for
labels.CONTROL_HOLD_S. `decide` is the two together.

(The older two-step version, dead ball first and then the team inside every live stretch, lost to
this one in every test: see experiments/README.md.)
"""
from __future__ import annotations

import itertools

import numpy as np

from fc_anomaly.data import FPS
from fc_anomaly.labels import CONTROL_HOLD_S


def runs(mask: np.ndarray):
    """(start, stop) of every run of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.where(d == 1)[0], np.where(d == -1)[0])


def three_states(p_dead: np.ndarray, p_home: np.ndarray, segment: np.ndarray,
                 team_cost: float, dead_cost: float) -> np.ndarray:
    """0=H, 1=A, 2=D: the most likely sequence when a change of team costs `team_cost` and going
    into or out of a stoppage costs `dead_cost`."""
    pd_ = np.clip(p_dead, 1e-4, 1 - 1e-4)
    ph = np.clip(p_home, 1e-4, 1 - 1e-4)
    e = np.log(np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], axis=1))
    cost = np.array([[0, team_cost, dead_cost], [team_cost, 0, dead_cost], [dead_cost, dead_cost, 0]])
    out = np.zeros(len(e), dtype=np.int8)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        out[idx] = viterbi(e[idx], cost)
    return out


def viterbi(e: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Best path through log-probabilities e (T, K) when going from state i to j costs cost[i, j]."""
    T, K = e.shape
    back = np.zeros((T, K), dtype=np.int8)
    score = e[0].copy()
    for t in range(1, T):
        cand = score[:, None] - cost               # cand[i, j]: come from i, now in j
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def control_rule(labels: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """The challenge's rule on the decided labels: a team that has the ball for less than
    CONTROL_HOLD_S and then loses it again showed no control, so that stretch goes back to the team
    that had the ball. The first possession after a stoppage always counts. The same rule as in
    labels.to_challenge_definition, without its pass test: passes are not known here."""
    out = labels.copy()
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        x = out[idx]
        pos, current = 0, None
        for k, group in itertools.groupby(labels[idx]):
            n = len(list(group))
            if k == 2:
                current = None                                         # stoppage
            elif current is None or k == current or n >= CONTROL_HOLD_S * FPS:
                current = k                                            # restart, same team, or control
            else:
                x[pos:pos + n] = current                               # no control: no change
            pos += n
        out[idx] = x
    return out


def decide(p_dead: np.ndarray, p_home: np.ndarray, segment: np.ndarray,
           team_cost: float, dead_cost: float) -> np.ndarray:
    """0=H, 1=A, 2=D: the most likely sequence (three_states), then the challenge's rule (control_rule)."""
    return control_rule(three_states(p_dead, p_home, segment, team_cost, dead_cost), segment)


def to_letters(lab: np.ndarray) -> np.ndarray:
    return np.array(["H", "A", "D"], dtype=object)[lab]

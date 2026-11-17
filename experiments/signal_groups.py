"""The test bench, and which groups of team signals can go.

    python -m experiments.signal_groups [seed offset] [variant ...]

The pipeline trained on one seed and scored the way fc_anomaly/train.py does: leave one IDSSE match out
with both second looks and the penalties from train.py's grid, plus the three Metrica games (as tracked
and with the ball smoothed 0.8 s), predicted by the models trained on all seven matches. The first row
(all signals) is train.py's own report for that seed. The other rows leave groups of team signals out.

A group can go if leaving it out costs less than 0.005 on leave-one-match-out and Metrica is not down
by more than 0.005, on two seeds (the seed-to-seed noise is about 0.005).
Results: experiments/results/signal_groups.csv and signal_groups_per_test.csv.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fc_anomaly import features, models
from fc_anomaly.data import IDSSE_MATCHES as MATCHES, ROOT
from fc_anomaly.degrade import degrade
from fc_anomaly.evaluate import leave_one_out, spell_f1
from fc_anomaly.labels import to_challenge_definition
from fc_anomaly.stabilise import decide
from fc_anomaly.train import DEAD_COSTS, TEAM_COSTS

warnings.filterwarnings("ignore")
BASE_SEED = models.SEED
JOBS = 6
METRICA_SMOOTH = (0.0, 0.8)
OUT = ROOT / "experiments" / "results" / "signal_groups.csv"
OUT_TESTS = ROOT / "experiments" / "results" / "signal_groups_per_test.csv"

FOUR = ("carry", "block", "press", "long")
# name -> the groups left out of the team model
VARIANTS = {
    "all 183": (),
    "without the four groups": FOUR,
    "without the four groups and the rule touches": FOUR + ("touch",),
    "without the four groups, rule touches, contact, ball direction": FOUR + ("touch", "contact", "direction"),
    "without the rule touches": ("touch",),
    "without the ball path": ("path",),
    "without carry": ("carry",),
    "without block": ("block",),
    "without press": ("press",),
    "without long": ("long",),
    "without contact": ("contact",),
    "without ball direction": ("direction",),
}


def group(col: str) -> str:
    """The group a team-model column belongs to (features.team_table)."""
    for g in FOUR:
        if col.startswith(g + "_"):
            return g
    if col.startswith("bb"):
        return "path"
    if col.startswith("touch"):
        return "touch"
    if col.startswith(("since_", "until_")):
        return "contact"
    if col.startswith("ball_vx"):
        return "direction"
    if col.startswith(("home_", "away_", "diff_")):
        return "shape"
    return "geometry"


def metrica_table(game: int, seed: int, smooth_s: float) -> dict:
    """A Metrica sample game as a test table (as experiments/lab.table: the provider's own ball)."""
    from fc_anomaly.metrica import load_metrica
    m, passes = load_metrica(game)
    m = degrade(m, seed, ball="keep", smooth_s=smooth_s)
    gt = to_challenge_definition(m.owner, m.dead, m.segment, passes)
    return dict(m=m, gt=gt, Fd=models.with_broadcast(features.dead_table(m), gt, m.segment, seed),
                Fp=features.team_table(m))


def _score(gt, p_dead, p_home, segment, team_cost, dead_cost) -> float:
    return spell_f1(gt, decide(p_dead, p_home, segment, team_cost, dead_cost))["spellF1"]


def scores(tabs: dict, probs: dict, team_cost: float, dead_cost: float) -> dict:
    keys = list(probs)
    s = Parallel(n_jobs=JOBS)(delayed(_score)(tabs[k]["gt"], *probs[k], tabs[k]["m"].segment, team_cost, dead_cost)
                              for k in keys)
    return dict(zip(keys, s))


def looks(second, second_dead, first: tuple, segment) -> tuple:
    """(p_dead, p_home) after both second looks, each reading the first models' probabilities."""
    return models.second_look_dead(second_dead, *first, segment), models.second_look(second, *first, segment)


def main(offset: int, names: list) -> None:
    models.SEED = BASE_SEED + offset            # make_model, the degrade seeds and the smoothing all read it
    seed = models.SEED
    keys = models.both(MATCHES)
    t0 = time.time()
    jobs = [(mid, seed + i, ball, models.training_smoothing(i, ball))
            for i, mid in enumerate(MATCHES) for ball in models.BALL_MODES]
    T = dict(zip(keys, Parallel(n_jobs=JOBS)(delayed(models.tables)(*j) for j in jobs)))
    mkeys = [("metrica", g, s) for g in (1, 2, 3) for s in METRICA_SMOOTH]
    M = dict(zip(mkeys, Parallel(n_jobs=JOBS)(delayed(metrica_table)(g, seed + g - 1, s) for _, g, s in mkeys)))
    print(f"tables: {(time.time() - t0) / 60:.1f} min", flush=True)

    p_dead = {}                                  # the dead-ball model: the same in every row
    for test, rest in leave_one_out(MATCHES):
        dead = models.fit_dead(T, models.both(rest))
        for k in models.both([test]):
            p_dead[k] = dead.predict_proba(T[k]["Fd"])[:, 1]
    dead = models.fit_dead(T, keys)
    for k in M:
        p_dead[k] = dead.predict_proba(M[k]["Fd"])[:, 1]
    print(f"dead-ball model: {(time.time() - t0) / 60:.1f} min", flush=True)

    all_cols = list(T[keys[0]]["Fp"].columns)
    ref = None
    for name in names:
        t1 = time.time()
        cols = [c for c in all_cols if group(c) not in VARIANTS[name]]
        first = {}
        for test, rest in leave_one_out(MATCHES):
            team = models.fit_team(T, models.both(rest), cols=cols)
            for k in models.both([test]):
                first[k] = (p_dead[k], team.predict_proba(T[k]["Fp"][cols])[:, 1])
        oof = {}
        for test, rest in leave_one_out(MATCHES):
            second, second_dead = models.fit_second(T, first, models.both(rest)), models.fit_second_dead(T, first, models.both(rest))
            for k in models.both([test]):
                oof[k] = looks(second, second_dead, first[k], T[k]["m"].segment)
        team = models.fit_team(T, keys, cols=cols)
        second, second_dead = models.fit_second(T, first, keys), models.fit_second_dead(T, first, keys)
        met_first = {k: (p_dead[k], team.predict_proba(M[k]["Fp"][cols])[:, 1]) for k in M}
        met = {k: looks(second, second_dead, met_first[k], M[k]["m"].segment) for k in M}

        grid = {(tc, dc): scores(T, oof, tc, dc) for tc, dc in itertools.product(TEAM_COSTS, DEAD_COSTS)}
        tc, dc = max(grid, key=lambda g: np.mean(list(grid[g].values())))
        lomo, lomo_first, m_sc = grid[(tc, dc)], scores(T, first, tc, dc), scores(M, met, tc, dc)
        row = {"variant": name, "seed": offset, "signals": len(cols), "team_cost": tc, "dead_cost": dc,
               "LOMO": np.mean(list(lomo.values())), "LOMO first layer": np.mean(list(lomo_first.values())),
               **{f"LOMO {b}": np.mean([v for k, v in lomo.items() if k[1] == b]) for b in models.BALL_MODES},
               **{f"Metrica {s}": np.mean([v for k, v in m_sc.items() if k[2] == s]) for s in METRICA_SMOOTH}}
        if name == list(VARIANTS)[0]:
            ref = row
        if ref is not None:
            for c in ("LOMO", "Metrica 0.0", "Metrica 0.8"):
                row[f"d {c}"] = row[c] - ref[c]
        tests = [{"variant": name, "seed": offset, "test": "/".join(map(str, k)), "spellF1": v}
                 for k, v in {**lomo, **m_sc}.items()]
        for path, new in ((OUT, [row]), (OUT_TESTS, tests)):
            old = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if len(old):
                old = old[~((old.variant == name) & (old.seed == offset))]
            pd.concat([old, pd.DataFrame(new)]).to_csv(path, index=False, float_format="%.4f")
        print(f"{name} (seed +{offset}): {len(cols)} signals, penalties {tc}/{dc}, LOMO {row['LOMO']:.4f} "
              f"(first layer {row['LOMO first layer']:.4f}), Metrica sharp {row['Metrica 0.0']:.4f}, "
              f"0.8 s {row['Metrica 0.8']:.4f}   [{(time.time() - t1) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    off = int(args[0]) if args and args[0].lstrip("-").isdigit() else 0
    chosen = [a for a in args if a in VARIANTS] or list(VARIANTS)
    if list(VARIANTS)[0] not in chosen:
        chosen = [list(VARIANTS)[0]] + chosen
    main(off, chosen)

"""Train the four models on IDSSE, five times over, report leave-one-match-out, and save them.

    python -m fc_anomaly.train          # ~30-40 min; -> models/dead.joblib, team.joblib, second.joblib, second_dead.joblib

For every seed in models.SEED_OFFSETS (the training matches degraded with another random draw):
1. Leave one match out: the dead-ball and team model are trained on six matches and predict the
   seventh. Every match is there twice, with the ball hidden in two ways (models.py).
2. Leave one match out again for the two second looks: they are trained on step 1's predictions for
   six matches and improve the team probability and the dead-ball probability of the seventh.
3. Train the dead-ball and team model on all seven matches and the two second looks on all of step
   1's predictions.
Then:
4. Average the five pipelines' held-out probabilities (after their second looks) and choose on them the
   two penalties of the final decision (stabilise.py): what a change of team costs and what the edge of
   a stoppage costs. The decision is scored as it is used: with the challenge's rule applied after it.
5. Save the five sets of models with the chosen penalties.

The held-out scores are measured on the tables of the first seed, the same tables as a single model:
every seed's fold models predict those, so the report compares like with like.
Nothing from BAR-ATH is used: not its labels and not its frames.
"""
from __future__ import annotations

import itertools
from functools import partial

import joblib
import numpy as np
import pandas as pd

from fc_anomaly.data import IDSSE_MATCHES as MATCHES, ROOT
from fc_anomaly.evaluate import leave_one_out, spell_f1
from fc_anomaly.models import (DEAD_MODEL, SECOND_DEAD_MODEL, SECOND_MODEL, SEED, SEED_OFFSETS,
                               TEAM_MODEL, all_tables, both, fit_dead, fit_second, fit_second_dead,
                               fit_team, make_model, second_look, second_look_dead)
from fc_anomaly.stabilise import decide

TEAM_COSTS = [5, 8, 12, 20]
DEAD_COSTS = [8, 12, 18, 25, 50, 100]


def one_seed(seed: int, T0: dict) -> dict:
    """Steps 1-3 for one seed. Held-out probabilities are for the tables T0 (the first seed's)."""
    T = T0 if seed == SEED else all_tables(MATCHES, seed)
    make = partial(make_model, seed=seed)
    first, first0 = {}, {}                   # (p_dead, p_home), that match unseen: on T (to train the looks) and on T0
    for test, rest in leave_one_out(MATCHES):
        dead = fit_dead(T, both(rest), make=make)
        team = fit_team(T, both(rest), make=make)
        for key in both([test]):
            for out, tabs in ((first, T), (first0, T0)):
                out[key] = (dead.predict_proba(tabs[key]["Fd"])[:, 1], team.predict_proba(tabs[key]["Fp"])[:, 1])
        print(f"  seed {seed}: fold {test} done", flush=True)
    oof, team_only = {}, {}                  # on T0, after both looks / after the team's look only
    for test, rest in leave_one_out(MATCHES):
        second = fit_second(T, first, both(rest), make=make)
        second_dead = fit_second_dead(T, first, both(rest), make=make)
        for key in both([test]):
            seg = T0[key]["m"].segment
            team_only[key] = (first0[key][0], second_look(second, *first0[key], seg))
            oof[key] = (second_look_dead(second_dead, *first0[key], seg), team_only[key][1])
    final = dict(dead=fit_dead(T, both(MATCHES), make=make),
                 team=fit_team(T, both(MATCHES), make=make),
                 second=fit_second(T, first, both(MATCHES), make=make),
                 second_dead=fit_second_dead(T, first, both(MATCHES), make=make))
    return dict(oof=oof, team_only=team_only, first=first0, final=final)


def mean_probs(runs: list, part: str) -> dict:
    """{key: (p_dead, p_home)} averaged over the seeds."""
    return {k: tuple(np.mean([r[part][k][i] for r in runs], axis=0) for i in (0, 1)) for k in runs[0][part]}


def score(T: dict, probs: dict, team_cost: float, dead_cost: float) -> float:
    return float(np.mean([spell_f1(T[k]["gt"], decide(*probs[k], T[k]["m"].segment, team_cost, dead_cost))["spellF1"]
                          for k in probs]))


def main() -> None:
    T0 = all_tables(MATCHES, SEED)
    runs = []
    for offset in SEED_OFFSETS:
        runs.append(one_seed(SEED + offset, T0))
        print(f"seed {SEED + offset} done", flush=True)
    oof = mean_probs(runs, "oof")

    rows = []
    for team_cost, dead_cost in itertools.product(TEAM_COSTS, DEAD_COSTS):
        res = pd.DataFrame([{"ball": key[1], **spell_f1(T0[key]["gt"], decide(*oof[key], T0[key]["m"].segment,
                                                                            team_cost, dead_cost))}
                            for key in oof])
        rows.append({"team_cost": team_cost, "dead_cost": dead_cost, "spellF1": res.spellF1.mean(),
                     **{f"spellF1 {b}": v for b, v in res.groupby("ball").spellF1.mean().items()},
                     "macroF1": res.macroF1.mean(), "worst match": res.spellF1.min()})
    grid = pd.DataFrame(rows).sort_values("spellF1", ascending=False)
    print(f"\nH/A/D, leave one match out on IDSSE, {len(runs)} seeds averaged, mean over the 7 held-out matches "
          "(both ways of hiding):")
    print(grid.round(3).to_string(index=False))
    best = grid.iloc[0]
    penalties = {"team_cost": float(best.team_cost), "dead_cost": float(best.dead_cost)}
    for name, part in (("without the dead-ball look", "team_only"), ("without both second looks", "first")):
        print(f"with these penalties but {name}: spellF1 {score(T0, mean_probs(runs, part), **penalties):.3f}")
    for offset, r in zip(SEED_OFFSETS, runs):
        print(f"seed {SEED + offset} alone, with these penalties: spellF1 {score(T0, r['oof'], **penalties):.4f}")

    DEAD_MODEL.parent.mkdir(exist_ok=True)
    table = T0[both(MATCHES)[0]]
    finals = [r["final"] for r in runs]
    joblib.dump({"models": [f["dead"] for f in finals], "columns": list(table["Fd"].columns)}, DEAD_MODEL)
    joblib.dump({"models": [f["team"] for f in finals], "columns": list(table["Fp"].columns), **penalties}, TEAM_MODEL)
    joblib.dump({"models": [f["second"] for f in finals]}, SECOND_MODEL)
    joblib.dump({"models": [f["second_dead"] for f in finals]}, SECOND_DEAD_MODEL)
    print(f"\nsaved {DEAD_MODEL.relative_to(ROOT)}, {TEAM_MODEL.name}, {SECOND_MODEL.name} and {SECOND_DEAD_MODEL.name} "
          f"({len(finals)} seeds each) with {penalties}")


if __name__ == "__main__":
    main()

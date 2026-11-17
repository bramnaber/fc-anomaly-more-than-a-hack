"""The saved models on the three Metrica games, per seed and averaged.

    python -m experiments.seeds_metrica

Metrica as tracked and with the ball smoothed 0.8 s (experiments/signal_groups.metrica_table), with the
penalties train.py saved. Rows: each seed's pipeline alone, and the average of the first n seeds, the way
fc_anomaly/predict.py averages them (per seed its own first models and second looks; then the mean).
Results: experiments/results/seeds_metrica.csv.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.signal_groups import METRICA_SMOOTH, metrica_table
from fc_anomaly import models
from fc_anomaly.evaluate import spell_f1
from fc_anomaly.models import second_look, second_look_dead
from fc_anomaly.stabilise import decide
from fc_anomaly.data import ROOT

warnings.filterwarnings("ignore")
OUT = ROOT / "experiments" / "results" / "seeds_metrica.csv"


def main() -> None:
    db, tb = joblib.load(models.DEAD_MODEL), joblib.load(models.TEAM_MODEL)
    sh, sd = joblib.load(models.SECOND_MODEL)["models"], joblib.load(models.SECOND_DEAD_MODEL)["models"]
    tc, dc = tb["team_cost"], tb["dead_cost"]
    keys = [(g, s) for g in (1, 2, 3) for s in METRICA_SMOOTH]
    M = dict(zip(keys, Parallel(n_jobs=4)(delayed(metrica_table)(g, models.SEED + g - 1, s) for g, s in keys)))
    looked = {}                                   # key -> [(p_dead, p_home) per seed]
    for k, t in M.items():
        seg = t["m"].segment
        looked[k] = []
        for dm, tm, s2, d2 in zip(db["models"], tb["models"], sh, sd):
            first = (dm.predict_proba(t["Fd"][db["columns"]])[:, 1], tm.predict_proba(t["Fp"][tb["columns"]])[:, 1])
            looked[k].append((second_look_dead(d2, *first, seg), second_look(s2, *first, seg)))

    def row(name, pick):
        out = {"pipeline": name, "team_cost": tc, "dead_cost": dc}
        for s in METRICA_SMOOTH:
            sc = []
            for g in (1, 2, 3):
                L = [looked[(g, s)][i] for i in pick]
                p = (np.mean([a for a, _ in L], axis=0), np.mean([b for _, b in L], axis=0))
                sc.append(spell_f1(M[(g, s)]["gt"], decide(*p, M[(g, s)]["m"].segment, tc, dc))["spellF1"])
            out[f"Metrica {s}"] = float(np.mean(sc))
        print(out, flush=True)
        return out

    n = len(db["models"])
    rows = [row(f"seed +{o} alone", [i]) for i, o in enumerate(models.SEED_OFFSETS[:n])]
    rows += [row(f"first {j} seeds averaged", list(range(j))) for j in range(2, n + 1)]
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()

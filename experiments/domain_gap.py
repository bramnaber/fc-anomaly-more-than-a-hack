"""How does the BAR-ATH input still differ from the training data? Input only, no labels.

    python -m experiments.domain_gap            # BAR-ATH against the training tables
    python -m experiments.domain_gap control    # one IDSSE match against the other six

A classifier gets the signals both models read and has to tell a BAR-ATH frame from a training frame
(IDSSE, degraded as in training). AUC 0.5: it cannot tell them apart; 1.0: every frame gives itself
away. Per signal: the AUC of that signal alone, its share of missing values and its median in both.
Per group: the classifier's AUC on that group only. The control tells one IDSSE match from the other
six: a group that separates BAR-ATH much better than it separates one Bundesliga match from the others
points to the tracking, not to "another match, other teams".

Folds: BAR-ATH first half + IDSSE matches 1-3 against second half + matches 4-7, and the other way
round. BAR-ATH frames from each half's kick-off to its end, built as predict.py builds them; IDSSE as
train.py builds its tables; every 5th frame.
Results: experiments/results/domain_gap.csv (per signal), domain_gap_groups.csv, domain_gap_control.csv.
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from experiments.signal_groups import group as team_group
from fc_anomaly import broadcast, features, models
from fc_anomaly.data import IDSSE_MATCHES as MATCHES, ROOT, load_barath
from fc_anomaly.degrade import degrade
from fc_anomaly.data import load_idsse
from fc_anomaly.labels import complete_passes, to_challenge_definition

warnings.filterwarnings("ignore")
JOBS = 4
STEP = models.SUBSAMPLE
OUT = ROOT / "experiments" / "results" / "domain_gap.csv"
OUT_GROUPS = ROOT / "experiments" / "results" / "domain_gap_groups.csv"


def raw(m) -> pd.DataFrame:
    """A few plain facts per frame that no model reads directly."""
    tracked = ~np.isnan(m.players[..., 0])
    return pd.DataFrame({"raw_n_home": tracked[:, m.team == 0].sum(1), "raw_n_away": tracked[:, m.team == 1].sum(1),
                         "raw_ball_seen": (~np.isnan(m.ball[:, 0])).astype(float)}).astype(np.float32)


def idsse(i: int, mid: str, ball: str) -> pd.DataFrame:
    seed = models.SEED + i
    m = degrade(load_idsse(mid), seed, ball, smooth_s=models.training_smoothing(i, ball))
    gt = to_challenge_definition(m.owner, m.dead, m.segment, complete_passes(mid))
    t = pd.concat([models.with_broadcast(features.dead_table(m), gt, m.segment, seed).add_prefix("D:"),
                   features.team_table(m).reset_index(drop=True).add_prefix("T:"), raw(m).add_prefix("R:")], axis=1)
    t = t.iloc[::STEP].copy()
    t["fold"] = 0 if i < 3 else 1
    return t


def barath() -> pd.DataFrame:
    from games import half_video_blocks
    sys.path.insert(0, str(ROOT / "starter"))
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    m = load_barath()
    Fd = pd.concat([features.dead_table(m).reset_index(drop=True),
                    broadcast.signals(broadcast.pitch_on_screen(), m.segment)], axis=1)
    bc_ball = broadcast.ball_on_tactical() * np.array(m.pitch, dtype=np.float32)
    mt = replace(m, ball=np.where(np.isnan(m.ball[:, :1]), bc_ball, m.ball).astype(np.float32))
    t = pd.concat([Fd.add_prefix("D:"), features.team_table(mt).reset_index(drop=True).add_prefix("T:"),
                   raw(m).add_prefix("R:")], axis=1)
    keep, fold = np.zeros(m.n_frames, bool), np.zeros(m.n_frames, np.int8)
    for (half, a, b), kickoff in zip(half_video_blocks("BAR-ATH"), sorted(KICKOFF_VIDEO_FRAMES.values())):
        keep[kickoff:b + 1] = True
        fold[kickoff:b + 1] = 0 if half == 1 else 1
    t["fold"] = fold
    return t[keep].iloc[::STEP].copy()


def group(col: str) -> str:
    kind, name = col.split(":", 1)
    if kind == "R":
        return "raw"
    if kind == "T":
        return "team/" + team_group(name)
    if name.startswith(("p75", "p90", "slow")):
        return "dead/player speed"
    if name.startswith("spread"):
        return "dead/spread"
    if name.startswith("out_"):
        return "dead/ball outside"
    if name.startswith("bc_"):
        return "dead/broadcast"
    return "dead/ball seen"


def classifier_auc(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray) -> float:
    p = np.zeros(len(y))
    for f in (0, 1):
        tr = fold != f
        w = np.where(y[tr] == 1, (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), 1.0)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        p[~tr] = clf.fit(X[tr], y[tr], sample_weight=w).predict_proba(X[~tr])[:, 1]
    return roc_auc_score(y, p)


def control(n: int = 4) -> None:
    """Group AUCs for 'this IDSSE match against the other six', for the first n matches."""
    t0 = time.time()
    parts = Parallel(n_jobs=JOBS)(delayed(idsse)(i, mid, ball) for i, mid in enumerate(MATCHES)
                                  for ball in models.BALL_MODES)
    seg = [p.index for p in parts]
    cols = [c for c in parts[0].columns if c != "fold"]
    groups = sorted({group(c) for c in cols})
    rows = []
    for j in range(n):
        own = [p for i, p in enumerate(parts) if i // 2 == j]
        rest = [p for i, p in enumerate(parts) if i // 2 != j]
        B = pd.concat(own, ignore_index=True)
        half = np.r_[[(p.index.to_numpy() >= p.index.to_numpy().max() / 2).astype(int) for p in own]].ravel()
        I = pd.concat(rest, ignore_index=True)
        ifold = np.concatenate([np.full(len(p), 0 if k < 6 else 1) for k, p in enumerate(rest)])
        X = pd.concat([I[cols], B[cols]], ignore_index=True)
        y = np.r_[np.zeros(len(I)), np.ones(len(B))]
        fold = np.r_[ifold, half]
        row = {"match": MATCHES[j], "ALL": classifier_auc(X, y, fold)}
        for g in groups:
            gc = [c for c in cols if group(c) == g]
            row[g] = classifier_auc(X[gc], y, fold)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(ROOT / "experiments" / "results" / "domain_gap_control.csv", index=False, float_format="%.4f")
    print(C.drop(columns="match").mean().round(3).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    t0 = time.time()
    parts = Parallel(n_jobs=JOBS)(delayed(idsse)(i, mid, ball) for i, mid in enumerate(MATCHES)
                                  for ball in models.BALL_MODES)
    I = pd.concat(parts, ignore_index=True)
    B = barath().reset_index(drop=True)
    print(f"tables: {(time.time() - t0) / 60:.1f} min; IDSSE {len(I)} frames, BAR-ATH {len(B)}", flush=True)
    cols = [c for c in B.columns if c != "fold"]
    X = pd.concat([I[cols], B[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(I)), np.ones(len(B))]
    fold = np.r_[I.fold.to_numpy(), B.fold.to_numpy()]

    rows = []
    for c in cols:
        a, b = I[c].to_numpy(), B[c].to_numpy()
        ok = ~np.isnan(X[c].to_numpy())
        auc = roc_auc_score(y[ok], X[c].to_numpy()[ok]) if ok.sum() and len(np.unique(y[ok])) == 2 else np.nan
        rows.append({"signal": c, "group": group(c), "auc": auc, "sep": abs(auc - 0.5) if auc == auc else np.nan,
                     "missing IDSSE": np.isnan(a).mean(), "missing BAR-ATH": np.isnan(b).mean(),
                     "median IDSSE": np.nanmedian(a), "median BAR-ATH": np.nanmedian(b),
                     "p90 IDSSE": np.nanpercentile(a, 90), "p90 BAR-ATH": np.nanpercentile(b, 90)})
    S = pd.DataFrame(rows).sort_values("sep", ascending=False)
    S.to_csv(OUT, index=False, float_format="%.4f")
    print(S.head(30).round(3).to_string(index=False), flush=True)

    grows = [{"group": "ALL", "signals": len(cols), "classifier AUC": classifier_auc(X, y, fold)}]
    print(f"\nclassifier on all signals: AUC {grows[0]['classifier AUC']:.3f}", flush=True)
    for g in sorted(S.group.unique()):
        gc = [c for c in cols if group(c) == g]
        grows.append({"group": g, "signals": len(gc), "classifier AUC": classifier_auc(X[gc], y, fold)})
        print(f"  {g:28s} {len(gc):4d} signals  AUC {grows[-1]['classifier AUC']:.3f}", flush=True)
    pd.DataFrame(grows).to_csv(OUT_GROUPS, index=False, float_format="%.4f")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    control() if sys.argv[1:] == ["control"] else main()

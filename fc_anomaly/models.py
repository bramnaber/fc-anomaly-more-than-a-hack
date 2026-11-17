"""The four models, the training tables they learn from, and where they are saved.

All are scikit-learn's HistGradientBoostingClassifier with library defaults apart from a
fixed seed and a few more, smaller steps: deliberately untuned.

    dead-ball model   every frame          -> p(the ball is dead), from its own signals
    team model        every live frame     -> p(the home team touched the ball last); frames within
                                              1 s of a change count double in training
    second look       every live frame     -> a better p(the home team touched the ball last), from the
                                              first two models' probabilities in the seconds around the frame
    dead-ball look    every frame          -> a better p(the ball is dead), from the same probabilities

Training tables, per IDSSE match: the match degraded to look like the BAR-ATH tracking
(degrade.py), with the ball smoothed by a random 0-1.0 s per table (a tracker smooths the ball by an
amount we do not know, so the models practise on a wide range, from a sharp ball to a very smooth one), its labels under the challenge definition (labels.py), the signals of both models
(features.py), and the same match with home and away swapped (so the team model learns 'the team
that ...' and not 'the home team').
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fc_anomaly import broadcast, features
from fc_anomaly.data import FPS, ROOT, load_idsse
from fc_anomaly.degrade import degrade
from fc_anomaly.labels import complete_passes, to_challenge_definition

DEAD_MODEL = ROOT / "models" / "dead.joblib"
TEAM_MODEL = ROOT / "models" / "team.joblib"
SECOND_MODEL = ROOT / "models" / "second.joblib"
SECOND_DEAD_MODEL = ROOT / "models" / "second_dead.joblib"
SEED = 2027
# The final models are trained several times, each time on the training matches degraded with a different
# seed (another random draw of the hidden ball, the smoothing and the position noise), and their probabilities
# are averaged. The gain is less noise, not more data: one model trained on two draws gains nothing.
# Offsets on SEED.
SEED_OFFSETS = (0, 100, 200, 300, 400)
SUBSAMPLE = 5          # train on every 5th frame; neighbouring frames are near-copies


def make_model(seed: int | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          early_stopping=False, random_state=SEED if seed is None else seed)


# How the ball is hidden in the training data (degrade.py). Every match is used twice, once per way:
# we cannot measure without labels how the BAR-ATH tracking loses the ball, so the models practise on
# both: at random, and during stoppages. Every number comes from the Metrica sample games (external
# data, same provider as BAR-ATH); nothing is measured on BAR-ATH.
BALL_MODES = ("ext_random", "ext_stoppages")
SMOOTH_MAX_S = 1.0     # training ball smoothed by a random 0 .. 1.0 s per table: the BAR-ATH ball is about as
                       # smooth as a 0.6-0.8 s moving average, and a model must have seen such a ball in training


def training_smoothing(i: int, ball: str, seed: int | None = None) -> float:
    """The random smoothing (seconds) of training match i with this way of hiding the ball."""
    rng = np.random.default_rng([(SEED if seed is None else seed) + i, BALL_MODES.index(ball)])
    return round(float(rng.uniform(0, SMOOTH_MAX_S)), 2)


def tables(match_id: str, seed: int, ball: str, smooth_s: float = 0.0) -> dict:
    """Everything both models need from one IDSSE match, with the ball hidden the given way."""
    m = degrade(load_idsse(match_id), seed, ball, smooth_s=smooth_s)
    gt = to_challenge_definition(m.owner, m.dead, m.segment, complete_passes(match_id))
    sw = features.swapped(m)
    return dict(m=m, gt=gt, gt_sw=np.where(gt < 2, 1 - gt, 2).astype(np.int8),
                Fd=with_broadcast(features.dead_table(m), gt, m.segment, seed),
                Fp=features.team_table(m), Fp_sw=features.team_table(sw))


def with_broadcast(Fd: pd.DataFrame, gt: np.ndarray, segment: np.ndarray, seed: int) -> pd.DataFrame:
    """Dead-ball signals plus the broadcast signals (broadcast.py). A training match has no broadcast,
    so it gets one made up from SoccerNet and SkillCorner, following its own dead frames."""
    pitch = broadcast.simulate(gt == 2, segment, seed)
    return pd.concat([Fd.reset_index(drop=True), broadcast.signals(pitch, segment)], axis=1)


def all_tables(matches, seed: int | None = None, jobs: int = 4) -> dict:
    """Tables per (match, way of hiding the ball); match i is degraded with seed (default SEED) + i."""
    from joblib import Parallel, delayed
    seed = SEED if seed is None else seed
    keys = [(mid, ball) for mid in matches for ball in BALL_MODES]
    built = Parallel(n_jobs=jobs)(delayed(tables)(mid, seed + i, ball, training_smoothing(i, ball, seed))
                                  for i, mid in enumerate(matches) for ball in BALL_MODES)
    return dict(zip(keys, built))


def both(matches) -> list:
    """The keys of all_tables for these matches."""
    return [(mid, ball) for mid in matches for ball in BALL_MODES]


# The dead-ball model sees only its own signals, not the team model's. With the team signals it scored
# +0.02 on IDSSE, but that was an artifact: in IDSSE the ball does not move at all during a stoppage, in
# the Metrica games it does, and with the ball in view that model got worse there (AUC 0.875 -> 0.743;
# experiments/README, "Tried and dropped").
def fit_dead(tabs: dict, train, cols=None, make=make_model):
    """Dead-ball model on the given matches. cols: which signals (default: all)."""
    X = pd.concat([tabs[m]["Fd"].iloc[::SUBSAMPLE] for m in train])
    y = np.concatenate([(tabs[m]["gt"] == 2)[::SUBSAMPLE] for m in train])
    return make().fit(X if cols is None else X[cols], y)


CHANGE_WEIGHT = 2.0    # a team-model frame within CHANGE_WINDOW_S of a true change of team counts double:
CHANGE_WINDOW_S = 1.0  # that is where the score is lost (experiments/README, finding 2)


def near_change(gt: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Frames within CHANGE_WINDOW_S of a true change of team (home to away or back, within one half)."""
    w = int(CHANGE_WINDOW_S * FPS)
    y = np.zeros(len(gt), dtype=bool)
    change = (gt[1:] != gt[:-1]) & (gt[1:] < 2) & (gt[:-1] < 2) & (segment[1:] == segment[:-1])
    for f in np.flatnonzero(change) + 1:
        y[max(0, f - w):f + w] = True
    return y


def last_touch(t: dict) -> np.ndarray:
    """What the team model learns, per frame of a training table: which team touched the ball last
    (0 home, 1 away, 2 no play). This is the provider's owner as it is, before the challenge rule.

    The training labels (labels.py) give a touch back to the other team when it lasted under 1.5 s.
    A model asked for those labels has to see the touch and judge the 1.5 s at once. Asked for the
    touch alone it answers what the tracking can show, and the 1.5 s is applied afterwards as a rule
    (stabilise.control_rule)."""
    owner = t["m"].owner
    return np.where((t["gt"] < 2) & (owner >= 0), owner, 2).astype(np.int8)


def fit_team(tabs: dict, train, cols=None, make=make_model):
    """Team model on the given matches: which team touched the ball last (last_touch), live frames
    only, both orientations; frames near a change weigh CHANGE_WEIGHT."""
    X, y, w = [], [], []
    for m in train:
        t = tabs[m]
        touch = last_touch(t)
        live = touch < 2
        X += [t["Fp"][live].iloc[::SUBSAMPLE], t["Fp_sw"][live].iloc[::SUBSAMPLE]]
        y += [(touch[live] == 0)[::SUBSAMPLE], (touch[live] == 1)[::SUBSAMPLE]]
        wk = np.where(near_change(touch, t["m"].segment), CHANGE_WEIGHT, 1.0)[live][::SUBSAMPLE]
        w += [wk, wk]
    X = pd.concat(X)
    return make().fit(X if cols is None else X[cols], np.concatenate(y), sample_weight=np.concatenate(w))


def fit_second(tabs: dict, probs: dict, train, make=make_model):
    """The second look, on the given tables: live frames, both orientations, the same target as the
    team model (last_touch).

    probs[key] = (p_dead, p_home) for that table from first models that did NOT train on that match.
    On its own training matches the team model is too sure of itself, and the second look would
    learn to trust it blindly."""
    X, y = [], []
    for key in train:
        t = tabs[key]
        touch = last_touch(t)
        live = touch < 2
        c = features.context_signals(*probs[key], t["m"].segment)[live].iloc[::SUBSAMPLE]
        X += [c, features.mirrored(c)]
        y += [(touch[live] == 0)[::SUBSAMPLE], (touch[live] == 1)[::SUBSAMPLE]]
    return make().fit(pd.concat(X), np.concatenate(y))


def second_look(model, p_dead: np.ndarray, p_home: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """p(the home team touched the ball last) for every frame after the second look."""
    return model.predict_proba(features.context_signals(p_dead, p_home, segment))[:, 1]


def fit_second_dead(tabs: dict, probs: dict, train, make=make_model):
    """The second look for the dead-ball probability: the same columns as fit_second (both first models'
    probabilities in the seconds around the frame), the target "is this frame a stoppage". Every frame,
    both orientations (a stoppage is a stoppage whichever team is called home). probs as in fit_second:
    from first models that did not train on that match."""
    X, y = [], []
    for key in train:
        t = tabs[key]
        c = features.context_signals(*probs[key], t["m"].segment).iloc[::SUBSAMPLE]
        dead = (t["gt"] == 2)[::SUBSAMPLE]
        X += [c, features.mirrored(c)]
        y += [dead, dead]
    return make().fit(pd.concat(X), np.concatenate(y))


def second_look_dead(model, p_dead: np.ndarray, p_home: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """p(the ball is dead) for every frame after its second look."""
    return model.predict_proba(features.context_signals(p_dead, p_home, segment))[:, 1]

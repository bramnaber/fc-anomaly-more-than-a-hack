"""The whole pipeline on BAR-ATH -> the submission file.

    python -m fc_anomaly.predict            # writes outputs/FC_Anomaly.csv

    tracking -> 1. dead-ball model -> p(the ball is dead)        for every frame (+ broadcast: pitch on screen?)
             -> 2. team model      -> p(home touched the ball last) for every frame
             -> 3. second look     -> a better p(home touched the ball last) and a better p(the ball is
                                      dead), each from the probabilities of steps 1 and 2 in the seconds
                                      before and after each frame
             -> 4. one decision    -> the most likely H / A / D sequence, where a change of team and
                                      the edge of a stoppage each cost a fixed penalty
             -> 5. the rule        -> a touch is a change of possession only if that team keeps the
                                      ball for 1.5 s (CHALLENGE.md: control)
    Steps 1-3 run once per seed (models.SEED_OFFSETS: the same models trained on differently degraded
    copies of the training matches); their two probabilities are averaged before step 4.
    X outside the graded halves. All models were trained on IDSSE only (fc_anomaly/train.py).
"""
from __future__ import annotations

import sys
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

from fc_anomaly import broadcast, features
from fc_anomaly.data import ROOT, load_barath
from fc_anomaly.models import (DEAD_MODEL, SECOND_DEAD_MODEL, SECOND_MODEL, TEAM_MODEL, second_look,
                               second_look_dead)
from fc_anomaly.stabilise import decide, to_letters

OUT = ROOT / "outputs" / "FC_Anomaly.csv"


def run() -> np.ndarray:
    from games import half_video_blocks
    m = load_barath()
    db = joblib.load(DEAD_MODEL)
    tb = joblib.load(TEAM_MODEL)
    seconds = joblib.load(SECOND_MODEL)["models"]
    seconds_dead = joblib.load(SECOND_DEAD_MODEL)["models"]

    # The kickoff frames are published by the organisers (FAQ.md, starter/load_smart_tagging.py); the
    # second half is graded from 110,808 but kicks off at 113,742.
    sys.path.insert(0, str(ROOT / "starter"))
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    kickoffs = sorted(KICKOFF_VIDEO_FRAMES.values())
    # The tracking also covers the minutes before each kickoff, the break and the minutes after the end.
    # The acceleration signals are measured against the match's own usual level (features.acceleration_signals),
    # and that level has to come from the match itself: each half from its kickoff to its end.
    in_match = np.zeros(m.n_frames, dtype=bool)
    for (_, a, b), kickoff in zip(half_video_blocks("BAR-ATH"), kickoffs):
        in_match[kickoff:b + 1] = True

    # Dead ball: the tracking signals plus whether the TV broadcast shows the pitch (broadcast.py;
    # the training matches got a made-up broadcast from SoccerNet and SkillCorner).
    Fd = pd.concat([features.dead_table(m, in_match).reset_index(drop=True),
                    broadcast.signals(broadcast.pitch_on_screen(), m.segment)], axis=1)[db["columns"]]
    p_deads = [model.predict_proba(Fd)[:, 1] for model in db["models"]]
    # Who has the ball: where the tactical camera lost the ball, the broadcast often still sees it
    # (broadcast.ball_on_tactical). Only for the team model: for the dead-ball model a missing ball is
    # itself a sign of a stoppage. Not testable on external data (no labelled two-camera match): a
    # football argument, more of the ball means a better call of who has it.
    bc_ball = broadcast.ball_on_tactical() * np.array(m.pitch, dtype=np.float32)
    mt = replace(m, ball=np.where(np.isnan(m.ball[:, :1]), bc_ball, m.ball).astype(np.float32))
    Fp = features.team_table(mt)[tb["columns"]]
    p_homes = [model.predict_proba(Fp)[:, 1] for model in tb["models"]]
    p_dead, p_home = np.mean(p_deads, axis=0), np.mean(p_homes, axis=0)

    # Rule: inside a graded half, everything before that half's kickoff is a dead ball.
    labels = np.full(m.n_frames, "X", dtype=object)
    for (_, a, b), kickoff in zip(half_video_blocks("BAR-ATH"), kickoffs):
        play = slice(kickoff, b + 1)        # the second looks start at the kickoff, like a half in training
        seg = m.segment[play]
        looked = [(second_look_dead(sd, pd_[play], ph[play], seg), second_look(sh, pd_[play], ph[play], seg))
                  for pd_, ph, sh, sd in zip(p_deads, p_homes, seconds, seconds_dead)]   # per seed, its own first models
        p_dead[play] = np.mean([d for d, _ in looked], axis=0)
        p_home[play] = np.mean([h for _, h in looked], axis=0)
        p_dead[a:kickoff] = 1.0
        sl = slice(a, b + 1)
        labels[sl] = to_letters(decide(p_dead[sl], p_home[sl], m.segment[sl], tb["team_cost"], tb["dead_cost"]))
    return labels


def main() -> None:
    labels = run()
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(OUT, index=False)
    g = labels[labels != "X"]
    print(f"wrote {OUT.relative_to(ROOT)}: graded H {np.mean(g == 'H'):.3f} "
          f"A {np.mean(g == 'A'):.3f} D {np.mean(g == 'D'):.3f}")


if __name__ == "__main__":
    main()

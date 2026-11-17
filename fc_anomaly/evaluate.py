"""How we measure, on IDSSE only: leave one match out, scored with the official metric code.

`leave_one_out` is the outer loop every experiment uses: train on the other matches, score on the
held-out one. Nested experiments call it again on the training matches to choose settings, so a
held-out match is never used for a choice.
"""
from __future__ import annotations

import sys

import numpy as np

from fc_anomaly.data import ROOT
from fc_anomaly.stabilise import to_letters

sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402


def leave_one_out(matches):
    """Yield (held_out, training_matches) for every match."""
    for test in matches:
        yield test, [m for m in matches if m != test]


def spell_f1(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Official spellF1 and macroF1 of 0/1/2 labels, plus the number of possessions."""
    r = pm.evaluate(pm.label_codes(to_letters(gt)), pm.label_codes(to_letters(pred)), bootstrap_n=0)
    runs = pm.valid_runs(pm.label_codes(to_letters(gt)))
    n_pred = len(pm.possession_spells(pm.label_codes(to_letters(pred)), runs))
    n_gt = len(pm.possession_spells(pm.label_codes(to_letters(gt)), runs))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss_pred": n_pred, "poss_gt": n_gt}

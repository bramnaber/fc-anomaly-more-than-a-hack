# experiments

What we measured to build the pipeline, and what we tried and dropped. Every number is measured on
external data, never on BAR-ATH: the seven IDSSE Bundesliga matches and the three Metrica sample games
(same tracking provider as BAR-ATH, never used for training). The score is the official spellF1.

## How we measure

- Leave one match out: train on six IDSSE matches and score the seventh, for each of the seven. Every
  match appears twice, with the ball hidden in each of the two ways used in training, so there are 14
  tests.
- Metrica as a second test: the three games as tracked, and with the ball smoothed over 0.8 s (about as
  smooth as the BAR-ATH ball).
- Noise: the same pipeline trained on five different random draws of the training data scores between
  0.756 and 0.769 (`python -m fc_anomaly.train` prints this). A difference under about 0.01 is therefore
  noise. A change was only kept if it gained more than that, against a bar written down before the
  run. Close to the bar, a second seed decided.

## Steps

| | change | gain |
|---|---|---|
| 1 | two gradient-boosting models trained on IDSSE: dead ball first, then which team | Metrica 0.68 |
| 2 | training labels: possession changes only when the new team completes a pass or keeps the ball 1.5 s (the two signs of control in CHALLENGE.md) | a definition, not a gain |
| 3 | IDSSE made to look like the BAR-ATH tracking (ball hidden in long gaps); one decision for H, A and D; signals for ball seen, touches and contact | Metrica 0.68 to 0.80 |
| 4 | every number used to degrade IDSSE measured on Metrica, none on BAR-ATH | -0.012, accepted for the rules |
| 5 | a second look at the seconds around each frame | +0.013 |
| 6 | whether the TV broadcast shows the pitch | +0.024 |
| 7 | team model: four more signal groups, double weight near a change of team, a smoothed training ball | +0.013 |
| 8 | team model: the ball's path as straight pieces, where a kink is a touch | +0.010 and +0.005 (two seeds) |
| 9 | training ball smoothed by 0 to 1.0 s instead of 0 to 0.6 s | +0.024 on a ball as smooth as BAR-ATH's |
| 10 | a second look for the dead-ball probability as well | +0.012 and +0.015 (two seeds) |
| 11 | five seeds averaged | +0.010; Metrica 0.842 (as tracked), 0.836 (smoothed) |
| 12 | the team model learns who touched the ball last; the 1.5 s rule is applied after the decision | leave one match out 0.753 to 0.764; Metrica 0.865 (as tracked), 0.837 (smoothed) |
| 13 | dead-ball model: how hard the players speed up, brake and turn (median and 75th percentile over the players, divided by the match's own median): the submission | leave one match out 0.764 to 0.777 (five seeds each step +0.012 to +0.017); Metrica 0.868 (as tracked), 0.840 (smoothed) |

## Findings

### Training data that looks like BAR-ATH matters more than the model

The largest gains came from imitating the BAR-ATH tracking: hiding the ball the way the provider loses
it (step 3, Metrica 0.68 to 0.80) and smoothing the ball the way its tracker does (step 9, +0.034 over
step 7 on a smooth ball). Every other kind of model we tried stayed within the noise (see below).

As a check, `domain_gap.py` trains a classifier to tell a BAR-ATH frame from a training frame. The
differences it still finds were already imitated or measured: Barcelona's own way of playing, extra and
missing players in the tracking, a slightly different TV director, and a ball that wanders a few
centimetres around its smoothed path.

### The score is lost on short possessions

Each kind of disagreement between the model and the truth, repaired one at a time (step 7, leave one
match out):

| mistake | per 90 min | spellF1 if repaired |
|---|---|---|
| a possession missed | 31 | +0.113 |
| a change of team found, but too early or too late | 138 | +0.050 |
| a possession that is not there | 15 | +0.043 |
| the start or end of a stoppage off | | +0.034 |

The missed possessions are short (median 2.4 s), belong to the other team, sit inside a longer
possession, and 85-90% of them show at most one touch. The model is confident exactly where it is
wrong, so methods that rely on its own confidence (self-training, a specialist for short possessions)
did not help.

Step 12 helped here: a model asked for the possession label has to see the touch and judge the 1.5 s
at once; asked for the last touch only, it answers what the tracking can show, and the 1.5 s becomes a
rule. What is still missed are duels: with an opponent within 1.4 m of the touch, 43% of the short
possessions are missed; after a long ball, 4%. In a duel the tracking does not show who touched the
ball.

### The label definition matters more than any model change

The challenge says possession changes only when the new team has *control*, for example when it
"keeps the ball" for a few seconds. That can be read in two ways:

- ours: the new team touched the ball last and the ball stays theirs for 1.5 s, even if none of
  their players is right at it (a rolling ball, a ball in the air);
- stricter: a player of the new team must actually be at the ball for some time.

The choice matters a lot. A model trained under one reading and scored under the other loses 0.05
to 0.16 spellF1, about ten times more than any model change we made.

CHALLENGE.md names the public five minutes as the reference for such cases ("if your reading of the
definition disagrees with it, the ground truth is right"), and RULES.md allows them for checking that
labels mean what you think. We used them for this one check only, decided in advance. We counted the
possessions in which the new team is hardly at the ball (a player within 1 m for less than 0.5 s). On
the Bundesliga data, our reading gives 4.6 such possessions per 17 on average and the stricter one
1.6. So we agreed beforehand: 0 or 1 would point to the stricter reading, 4 or more to ours. The
public five minutes contain 7 of 17, so we kept our reading. Nothing in the pipeline was tuned on
those five minutes.

## Tried and dropped

Gains are against the pipeline of the moment; the noise is about 0.01.

| | idea | result |
|---|---|---|
| models | random forest, XGBoost, small neural networks, logistic regression | within the noise or worse |
| | tuning the gradient boosting (larger leaves, regularisation, more leaves; checked again at step 11) | +0.004 at best, and worse on a smooth ball |
| | a network over time (TCN), a network over the players, pitch control | +0.003, +0.001, +0.001 |
| | one model for H, A and D together | +0.02 on IDSSE, but an artefact: in IDSSE the ball stands still during a stoppage, in BAR-ATH it does not |
| data | SkillCorner's 20 matches in training; the Metrica games in training | -0.037; no change |
| | players tracked under the wrong team in training; extra weight on short possessions; self-training on the match itself | within the noise |
| | no position noise on the players (five seeds) | Metrica 0.842 / 0.829 against 0.842 / 0.836 |
| signals | a learned touch detector; who stands at the ball; the ball estimated where it is hidden | within the noise, or worse on Metrica |
| | the ball found in the video | at most +0.02 even if perfect; not built |
| | the tactical tags (Smart Tagging) after a restart | -0.001 to -0.004 |
| decision | a wider second look; the two second looks reading each other; a decision that knows how long possessions last | within the noise |
| | a specialist for short possessions | +0.04 to +0.06 if it were perfect, +0.003 as learned |
| | also predicting on the mirrored match and averaging | +0.002 |
| | 24 other final decisions; the 1.5 s rule at another threshold (1.5 s was best); snapping changes of team to touches; a contact rule | +0.006 at best |
| | on top of step 12: soft kinks, push direction, a longer ball-path size, extra weight on missed possessions, second looks that see touches or who takes the restart, two kinds of model averaged | within the noise, or flat on Metrica |
| | on top of step 12, about 75 more variants: decision costs and rules, model sizes, the training ball, position noise, eight new signal groups for the team model | none passed a bar fixed beforehand |
| | step 13 in fixed units (m/s², metres to the line, share of players standing still) instead of against the match's own level | +0.010 to +0.015 on IDSSE, -0.007 to -0.15 on Metrica: the level differs per provider |
| simpler | without the outside-the-pitch rule in `degrade.py` | no change (two seeds); kept, so the submission stays as trained |
| | without the four signal groups of step 7, the rule touches, SoccerNet, the broadcast, the double weight, or the contact and ball-direction signals | each costs more than 0.005, so they stay |

## Files

| file | contents |
|---|---|
| `signal_groups.py` | the test bench (one seed, scored as `train.py` scores; leave one match out and Metrica), and leaving groups of signals out |
| `seeds_metrica.py` | the saved models on the Metrica games, per seed and averaged |
| `domain_gap.py` | how the BAR-ATH input still differs from the training data (input only, no labels) |
| `results/` | their results |

The earlier experiments ran on the code of their time. Their scripts are not kept; this file gives
their results.

# FC Anomaly: frame-by-frame ball possession, BAR-ATH

Team FC Anomaly. More than a Hack 2027, Phase 1. Code: https://github.com/bramnaber/fc-anomaly-more-than-a-hack

## 1. Approach

We split the question in two, because the two halves need different evidence: *is the ball dead?*
(the players' collective motion and how hard they change pace, whether the ball is seen, whether the
TV shows the pitch) and *which
team touched it last?* (the ball against both teams' positions, team shape, touches, pressing, the
ball's path). For every frame of BAR-ATH:

1. **Dead-ball model** gives p(dead).
2. **Team model** gives p(home touched the ball last). It is trained on live frames only, and on every
   match in both orientations, so it learns "the team that ..." rather than "the home team".
3. **Two second looks** re-read both probabilities in the 0.2 to 4 s before and after the frame.
4. **One decision** (Viterbi) finds the most likely H / A / D sequence; a change of team and the start
   or end of a stoppage each cost a fixed penalty. This removes the flicker that per-frame decisions
   have and that spellF1 punishes.
5. **The 1.5 s rule** (section 4): a team that has the ball for less than 1.5 s and loses it again
   showed no control, so that stretch goes back to the team that had it.

Steps 1-3 are trained five times on differently degraded copies of the training matches and the five
probabilities are averaged. All models are scikit-learn `HistGradientBoostingClassifier` with default
settings (fixed seed, 300 steps, learning rate 0.05).

**Why this and not something else.** Every number below is spellF1 on external data: leave one
IDSSE match out (14 tests), and the three Metrica games as tracked and with the ball smoothed over
0.8 s (as smooth as the BAR-ATH ball). The same pipeline on five random draws scores 0.756-0.769, so a
change counted only if it gained about 0.01, against a bar written down before the run.

- **Training data that looks like BAR-ATH mattered more than the model.** Hiding the ball the way this
  provider loses it took Metrica from 0.68 to 0.80; smoothing the training ball like this tracker does
  added 0.024 on a smooth ball. Random forests, XGBoost, neural networks over time (TCN) or over the
  players, and tuning the boosting all stayed within the noise.
- **Two models, not one.** One model for H/A/D together scored higher on IDSSE, but only because the
  IDSSE ball stands still during a stoppage and the BAR-ATH ball does not.
- **The team model learns the last touch, not the possession label.** Asked for possession it has to
  see the touch and judge the 1.5 s at once; asked for the last touch it answers what the tracking
  shows, and the 1.5 s becomes a rule.
- **Motion against the match's own level.** During play the players keep speeding up, braking and
  turning; in a stoppage they walk on at an even pace. Measured in m/s² this won on IDSSE and lost up
  to 0.15 on Metrica, because the level differs per provider. Divided by the match's own median it
  carries over: +0.013 leave one match out, +0.003 on Metrica. Before using it we checked on the
  BAR-ATH input alone (no labels) that the match does not read calmer than the training matches,
  the one case where this signal would hurt.
- **Result (submitted version):** leave one match out 0.777, Metrica 0.868 as tracked and 0.840
  smoothed.

The full log of steps and their gains is in `experiments/README.md`.

## 2. Assets used and dropped

**Provided assets used**

| asset | how |
|---|---|
| tactical tracking (players and ball) | the input of both models |
| broadcast homography | `valid` = the pitch is on screen; the director cuts away during stoppages (dead-ball model, +0.024) |
| broadcast tracking | aligned to the tactical recording per half by matching the ball (automatic, no labels); where the tactical camera lost the ball, the broadcast ball is used for the team model |
| kick-off frames (`starter/load_smart_tagging.py`) | inside a graded half, everything before the kick-off is dead ball |

**Tried and dropped** (gain against the pipeline of the moment; noise about 0.01)

| idea | result |
|---|---|
| Smart Tagging (who takes a restart) | -0.001 to -0.004 |
| a ball detector on both videos (pretrained, off the shelf; pixel to pitch with the homography) | on four test minutes it agrees with the tracked ball (median 1-1.6 m), but on the play frames where both tracking files lose the ball it finds one only 28% of the time: about 1% of play. Its other misses are mostly a spare ball by the touchline. Expected gain well below the noise; not built |
| SkillCorner's 20 matches as extra training matches | -0.037 |
| the Metrica games in training | no change (kept as a clean test instead) |
| other models: random forest, XGBoost, logistic regression, TCN, a network over the players, pitch control | within the noise |
| a learned touch detector; who stands at the ball; the ball estimated where it is hidden | within the noise, or worse on Metrica |
| a specialist for short possessions; self-training on BAR-ATH | +0.003; within the noise |
| 24 other final decisions; the 1.5 s rule at 1 / 2 / 3 s; snapping changes to touches | +0.006 at best |
| about 75 further variants (decision costs and rules, model sizes, the training ball, position noise, eight new signal groups for the team model) | none passed the bar fixed beforehand |
| the pace signals in fixed units instead of against the match's own level | +0.010 to +0.015 on IDSSE, -0.007 to -0.15 on Metrica |

## 3. Training data

No label, number or threshold in the pipeline comes from BAR-ATH. Everything that is trained or tuned
comes from the external data below.

| dataset | source | who labelled it | how much | used for |
|---|---|---|---|---|
| IDSSE Bundesliga tracking + events | Bassek et al. 2025, *Sci Data* 12:195 (CC BY 4.0) | tracking and ball status: the provider's system; passes: the provider's human event loggers. Our possession labels are computed from these by `labels.py` (section 4); nobody labelled anything by hand for us | 7 matches | the only training matches; leave-one-match-out test |
| Metrica sample games 1-3 | Metrica Sports, public sample data | Metrica (event data) | 3 games | how this provider's tracking loses the ball (the numbers in `degrade.py`); a second test. Never trained on |
| SkillCorner open data | github.com/SkillCorner/opendata (MIT), pinned commit | SkillCorner (phases of play) | 20 A-League matches | how often a TV broadcast shows the pitch during play and during stoppages |
| SoccerNet labels | SoccerNet (`Labels-cameras.json`, `Labels-v2.json`) | SoccerNet annotators | 204 games with camera labels | the same broadcast statistics, from more games |

BAR-ATH itself is only ever read as input (its tracking and its broadcast homography). Its public
first five minutes were never used to train, tune or choose a model. We used them once for one check
decided in advance: whether our reading of "control" matches the organisers' (see
`experiments/README.md`, "The label definition matters more than any model change"). The outcome was
to keep the reading we already had.

The IDSSE tracking is cleaner than BAR-ATH's, so `degrade.py` makes every training match look like
this provider's tracking: the ball hidden in long stretches with lengths measured on the Metrica
games (at random, or mostly during stoppages), 5 cm noise on player positions, most ball positions
outside the pitch removed, and the ball's path smoothed by a random 0 to 1.0 s. Each match is
degraded five times with different random draws (five seeds), and the five models are averaged.

## 4. Possession rules

Implemented in `fc_anomaly/labels.py` (training labels) and as the last step of the pipeline
(`stabilise.control_rule`).

- **Dead ball** follows the provider's ball status: a stoppage starts when the ball crosses a line or
  the referee stops play, and ends at the restart. A stoppage is never a possession, so `H D H` is two
  possessions. Inside a graded half, everything before the kick-off is dead ball.
- **Restarts.** The first possession after a stoppage always counts, however short: the team taking
  the throw-in, free kick, corner or goal kick has the ball.
- **Loose balls and challenges.** Possession changes only when the new team shows control, using the
  two signs CHALLENGE.md names: it completes a pass, or it keeps the ball for at least 1.5 s. A touch,
  a deflection, a block or a header won without control goes back to the team that had the ball. A
  ball rolling free after a touch stays with the team that touched it last until the other team
  gains control.
- **Why 1.5 s.** "A few seconds" leaves room. Of 1 / 1.5 / 2 / 3 s, 1.5 s lost least on the external
  tests if the organisers meant one of the others.
- On BAR-ATH passes are not known, so there only the 1.5 s test applies.

## 5. Limitations

- **Short possessions in duels.** Most lost score is possessions that are missed (median 2.4 s, the
  other team, inside a longer possession, mostly at most one touch). With an opponent within 1.4 m of
  the touch, 43% of them are missed; after a long ball, 4%. In a duel the tracking does not show who
  touched the ball, and the model is confident exactly there, so methods that rely on its confidence
  did not help.
- **The smoothed ball.** The BAR-ATH tracker smooths the ball (about a 0.6-0.8 s moving average), which
  hides touches. We imitate it in training, but a touch that is not in the data cannot be found.
- **External tests only.** We never scored on BAR-ATH beyond the public five minutes, which we used as
  a check and not to choose: there the submitted file scores spellF1 0.684, against 0.84-0.87 on the
  Metrica games. Five minutes and 17 possessions is too little to read much into, but it suggests
  BAR-ATH is harder than our external tests. The Metrica games are the closest test (same provider),
  but they are three games.
- **The broadcast ball** is used where the tactical camera lost it. That could not be tested on
  external data (no labelled match filmed by two cameras); it rests on the argument that seeing more of
  the ball gives a better call of who has it.
- **The definition of control.** "A few seconds" can be read more strictly (a player must actually be at
  the ball). A model trained under one reading and scored under the other loses 0.05-0.16, more than
  any model change.

**With more time:** a ball detector trained for football broadcasts (an off-the-shelf one finds the ball
on only 28% of the frames where the tracking loses it), a touch
detector for duels trained on a provider with sharp ball tracking, and a labelled two-camera match to
test the broadcast ball.

## 6. Reproduction

Tested with Python 3.9.6 on macOS; the versions are pinned in `requirements-fc_anomaly.txt`.

```bash
pip install -r requirements-fc_anomaly.txt
# unpack the organisers' data bundle into data/ (see data/README.md)
python -m fc_anomaly.predict      # writes outputs/FC_Anomaly.csv from the trained models in models/
```

The result is the submitted file exactly (md5 `f8ab01bab80f5250caab1162ee7ada47`). To retrain from scratch:

```bash
python -m fc_anomaly.download     # IDSSE, Metrica, SkillCorner and SoccerNet into data/external/
python -m fc_anomaly.train        # about 30-40 min; writes models/ and prints the leave-one-match-out scores
python -m fc_anomaly.predict
```

# fc_anomaly

The pipeline that produces `outputs/FC_Anomaly.csv`. How to run it: see the [README](../README.md) at
the top of the repository.

## Pipeline

For every frame of BAR-ATH:

1. The dead-ball model gives p(the ball is dead), from the players' collective motion, how hard they
   change pace (against the match's own usual level, so it means the same in every tracking), whether and
   where the ball is seen, and whether the TV broadcast shows the pitch (the director cuts away
   during stoppages).
2. The team model gives p(the home team touched the ball last), from the ball's position relative to
   both teams, team shape, touches, contact, pressing and the ball's path. It is trained on live
   frames only, and on every match in both orientations, so it learns "the team that ..." rather than
   "the home team". It learns the last touch, not the possession label: whether a touch became
   possession is decided in step 5.
3. Two second looks re-read both probabilities from 0.2 to 4 s before and after the frame. One
   improves p(home), the other p(dead).
4. One decision finds the most likely H / A / D sequence (Viterbi), where a change of team and the
   start or end of a stoppage each cost a fixed penalty. Inside a graded half, everything before the
   kick-off is D.
5. The 1.5 s rule of the possession rules below: a team that has the ball for less than 1.5 s and
   then loses it again showed no control, so that stretch goes back to the team that had the ball.
   The first possession after a stoppage always counts.

Steps 1 to 3 are trained five times, each time on differently degraded copies of the training
matches, and the five probabilities are averaged before step 4. All models are scikit-learn's
HistGradientBoostingClassifier with default settings, except a fixed seed and 300 steps at a learning
rate of 0.05 without early stopping.

| file | contents |
|---|---|
| `data.py` | loads BAR-ATH and IDSSE into the same arrays |
| `degrade.py` | makes the IDSSE tracking look like the BAR-ATH tracking |
| `labels.py` | training labels under the challenge's definition of possession |
| `features.py` | the signals of the dead-ball model, the team model and the second looks |
| `broadcast.py` | pitch on screen in the BAR-ATH broadcast, and a simulated broadcast for the training matches |
| `models.py` | training tables, the four models, the seeds |
| `stabilise.py` | the final decision |
| `evaluate.py` | leave-one-match-out and the official spellF1 |
| `train.py` | trains everything, reports leave-one-match-out, chooses the two penalties |
| `predict.py` | runs the pipeline on BAR-ATH |
| `metrica.py`, `skillcorner.py` | readers for external data |
| `download.py` | downloads the external data, pinned to the versions we used |

## Training data

| source | used for | licence |
|---|---|---|
| IDSSE (Bassek et al. 2025, *Sci Data* 12:195) | the seven training matches: tracking, ball status, ball-owning team, passes | CC BY 4.0 |
| Metrica sample games 1-3 | how the provider's tracking loses the ball (the numbers in `degrade.py`); a second test | attribution |
| SkillCorner open data | how often a broadcast shows the pitch during play and during stoppages | MIT |
| SoccerNet labels (`Labels-cameras.json`, `Labels-v2.json` of the games in `soccernet_games.txt`) | the same, from more games | SoccerNet terms |

`degrade.py` hides the ball in long stretches, either at random or mostly during stoppages, with
lengths measured on the Metrica games; adds 5 cm of noise to the player positions; removes most ball
positions outside the pitch; and smooths the ball's path by a random 0 to 1.0 s, because the BAR-ATH
tracker smooths it (about as much as a 0.6 to 0.8 s moving average). The training matches have no
broadcast, so each gets a simulated one, built from real SoccerNet and SkillCorner stretches of the
same kind (dead or live) and length.

## Possession rules

Implemented in `labels.py` for the training labels, and as step 5 of the pipeline
(`stabilise.control_rule`, without the pass test, since passes are not known on BAR-ATH).

- Dead ball follows the provider's ball status: a stoppage starts when the ball crosses a line or play
  is stopped, and ends at the restart.
- Possession changes only when the new team shows control, the two signs CHALLENGE.md names: it
  completes a pass, or it keeps the ball for at least 1.5 s. A touch, a deflection or a header won
  without control goes back to the team that had the ball. The first possession after a stoppage
  always counts.

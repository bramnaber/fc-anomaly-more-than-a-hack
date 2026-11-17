"""Metrica Sports sample data: 3 anonymised matches, same tracking provider as BAR-ATH.

    python -m fc_anomaly.metrica           # download (~180 MB) + summary per game

Source: https://github.com/metrica-sports/sample-data ("If you use it for anything public, please
acknowledge the source"). No licence file, so the files are downloaded, never committed.

There is no per-frame ball status or ball-owning team, so both are derived from the events, which
carry start and end frames on the tracking's frame axis:

- Dead ball: from a stoppage (BALL OUT; a SHOT that goes OUT or is a GOAL; BALL LOST by OFFSIDE,
  END HALF, HAND BALL or REFEREE HIT; a FAULT RECEIVED or a CHALLENGE with a FAULT) until the next
  restart (any SET PIECE, or a goal kick: PASS or BALL LOST with GOAL KICK). Also before the first
  and after the last event of each half. Derived from the events only, NOT from the tracking, so we
  can measure afterwards how well 'ball missing' matches 'dead ball'.
- Owner: the team of the latest on-ball event (PASS, BALL LOST, RECOVERY, SET PIECE, SHOT, CARRY).
  The challenge's definition of control (labels.py) is applied on top, as for IDSSE.
- Completed passes: Metrica's PASS is by definition a pass between two players of the same team.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from fc_anomaly.data import FPS, ROOT, Match, save_atomic

METRICA_DIR = ROOT / "data" / "external" / "metrica"
GAMES = [1, 2, 3]
PITCH = (105.0, 68.0)
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{g}/"
FILES = {1: ["RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"],
         2: ["RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"],
         3: ["events.json", "metadata.xml", "tracking.txt"]}

ON_BALL = {"PASS", "BALL LOST", "RECOVERY", "SET PIECE", "SHOT", "CARRY"}
MAX_STOPPAGE_S = 180      # a stoppage with no restart found within this time is not filled in


def path(game: int, name: str):
    return METRICA_DIR / f"Sample_Game_{game}_{name}"


def download() -> None:
    METRICA_DIR.mkdir(parents=True, exist_ok=True)
    for g, names in FILES.items():
        for name in names:
            if not path(g, name).exists():
                print(f"fetch {path(g, name).name}", flush=True)
                urllib.request.urlretrieve(BASE_URL.format(g=g) + f"Sample_Game_{g}_{name}", path(g, name))


# ---- tracking --------------------------------------------------------------------------------

def _tracking(game: int):
    """players (T,P,2), team (P,), ball (T,2) in metres; segment (T,) = half; frame_id (T,)."""
    cache = METRICA_DIR / "cache" / f"game{game}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["players"], z["team"], z["ball"], z["segment"], z["frame_id"]
    from kloppy import metrica
    if game in (1, 2):
        ds = metrica.load_tracking_csv(home_data=str(path(game, FILES[game][1])),
                                       away_data=str(path(game, FILES[game][2])), coordinates="kloppy")
    else:
        ds = metrica.load_tracking_epts(meta_data=str(path(3, "metadata.xml")),
                                        raw_data=str(path(3, "tracking.txt")), coordinates="kloppy")
    home, away = ds.metadata.teams
    plist = list(home.players) + list(away.players)
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    index = {p.player_id: i for i, p in enumerate(plist)}
    T = len(ds.frames)
    players = np.full((T, len(plist), 2), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    segment = np.zeros(T, dtype=np.int16)
    frame_id = np.zeros(T, dtype=np.int64)
    for t, fr in enumerate(ds.frames):
        segment[t], frame_id[t] = fr.period.id, fr.frame_id
        if fr.ball_coordinates is not None and not np.isnan(fr.ball_coordinates.x):
            ball[t] = (fr.ball_coordinates.x * PITCH[0], fr.ball_coordinates.y * PITCH[1])
        for p, pd_ in fr.players_data.items():
            c = pd_.coordinates
            if c is not None and not np.isnan(c.x):
                players[t, index[p.player_id]] = (c.x * PITCH[0], c.y * PITCH[1])
    save_atomic(cache, lambda fh: np.savez_compressed(fh, players=players, team=team, ball=ball, segment=segment,
                                                      frame_id=frame_id))
    return players, team, ball, segment, frame_id


# ---- events ----------------------------------------------------------------------------------

def events(game: int) -> pd.DataFrame:
    """One row per event: team (0 home, 1 away, -1 none), type, subtype, start/end frame id."""
    if game in (1, 2):
        e = pd.read_csv(path(game, "RawEventsData.csv"))
        return pd.DataFrame({"team": e["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
                             "type": e["Type"].str.upper(), "subtype": e["Subtype"].fillna("").str.upper(),
                             "start": e["Start Frame"].astype(int), "end": e["End Frame"].astype(int)})
    data = json.load(open(path(3, "events.json")))["data"]

    def subtype(x):
        s = x["subtypes"]
        s = s if isinstance(s, list) else [s] if s else []
        return "-".join(v["name"] for v in s).upper()
    home_id = "FIFATMA"                     # Team A is listed first in the metadata (home)
    return pd.DataFrame({"team": [0 if x["team"]["id"] == home_id else 1 for x in data],
                         "type": [x["type"]["name"].upper() for x in data],
                         "subtype": [subtype(x) for x in data],
                         "start": [int(x["start"]["frame"]) for x in data],
                         "end": [int(x["end"]["frame"] or x["start"]["frame"]) for x in data]})


def _is_stop(r) -> bool:
    s = r.subtype
    return (r.type == "BALL OUT" or (r.type == "SHOT" and ("OUT" in s or "GOAL" in s))
            or (r.type == "BALL LOST" and any(k in s for k in ("OFFSIDE", "END HALF", "HAND BALL", "REFEREE HIT")))
            or r.type == "FAULT RECEIVED" or (r.type == "CHALLENGE" and "FAULT" in s))


def _is_restart(r) -> bool:
    return r.type == "SET PIECE" or (r.type in ("PASS", "BALL LOST") and "GOAL KICK" in r.subtype)


def derived_labels(ev: pd.DataFrame, frame_id: np.ndarray, segment: np.ndarray):
    """dead (T,) bool, owner (T,) int8 and completed passes (n, 2) [row, team], from the events."""
    T = len(frame_id)
    row = lambda f: int(np.clip(np.searchsorted(frame_id, f), 0, T - 1))  # noqa: E731
    ev = ev.assign(a=[row(f) for f in ev.start], b=[row(f) for f in ev.end]).sort_values("a", kind="stable")

    dead = np.zeros(T, dtype=bool)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        e = ev[(ev.a >= idx[0]) & (ev.a <= idx[-1])]
        if len(e) == 0:
            dead[idx] = True
            continue
        dead[idx[0]:e.a.min()] = True                          # before the first event of the half
        dead[e.b.max() + 1:idx[-1] + 1] = True                 # after the last event of the half
        restarts = e[[_is_restart(r) for r in e.itertuples()]].a.to_numpy()
        for r in e[[_is_stop(r) for r in e.itertuples()]].itertuples():
            stop = r.a if r.type in ("FAULT RECEIVED", "CHALLENGE") else r.b
            nxt = restarts[restarts > stop]
            if len(nxt) and nxt[0] - stop <= MAX_STOPPAGE_S * FPS:
                dead[stop:nxt[0]] = True

    owner = np.full(T, -1, dtype=np.int8)
    on_ball = ev[ev.type.isin(ON_BALL) & (ev.team >= 0)]
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        e = on_ball[(on_ball.a >= idx[0]) & (on_ball.a <= idx[-1])]
        for (a, team), nxt in zip(e[["a", "team"]].to_numpy(), list(e.a.to_numpy()[1:]) + [idx[-1] + 1]):
            owner[a:nxt] = team
    passes = ev[(ev.type == "PASS") & (ev.team >= 0)][["a", "team"]].to_numpy().astype(np.int64)
    return dead, owner, passes


def load_metrica(game: int):
    """(Match with derived dead and owner, completed passes)."""
    players, team, ball, segment, frame_id = _tracking(game)
    dead, owner, passes = derived_labels(events(game), frame_id, segment)
    return Match(f"metrica{game}", players, team, ball, segment, PITCH, dead, owner), passes


def summary() -> None:
    for g in GAMES:
        m, passes = load_metrica(g)
        seen = ~np.isnan(m.ball[:, 0])
        print(f"game {g}: {m.n_frames} frames, halves {sorted(set(m.segment.tolist()))}, "
              f"{int((~np.isnan(m.players[..., 0])).sum(1).mean())} players/frame, {len(passes)} passes\n"
              f"  dead (from events) {m.dead.mean():.1%} | ball missing {1 - seen.mean():.1%}\n"
              f"  ball missing when dead {1 - seen[m.dead].mean():.1%}, when live {1 - seen[~m.dead].mean():.1%}\n"
              f"  dead when ball missing {m.dead[~seen].mean():.1%}", flush=True)


if __name__ == "__main__":
    download()
    summary()

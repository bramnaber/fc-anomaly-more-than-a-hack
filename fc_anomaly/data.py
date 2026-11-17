"""All data in, as the same simple arrays: the competition match (BAR-ATH) and the training data (IDSSE).

Both loaders return a `Match`:

    players   (T, P, 2) float32  positions in metres, origin at a pitch corner, NaN when untracked
    team      (P,)      int8     0 = home, 1 = away
    ball      (T, 2)    float32  metres, NaN when the ball is not tracked
    segment   (T,)      int16    contiguous recording pieces (halves); windows never cross them
    pitch     (length, width) in metres
    dead      (T,)      bool     IDSSE only: the provider's BallStatus == dead. None for BAR-ATH.
    owner     (T,)      int8     IDSSE only: team in possession per the provider (0 home, 1 away).

Everything downstream (features, models) only sees these arrays, so both datasets go through
exactly the same code.

IDSSE open data (7 Bundesliga matches, CC BY 4.0): Bassek, M., Rein, R., Weber, H. et al. "An
integrated dataset of spatiotemporal and event data in elite soccer." Sci Data 12, 195 (2025).
https://doi.org/10.1038/s41597-025-04505-y. Downloaded into data/external/idsse/ (gitignored):

    python -m fc_anomaly.data          # ~2.6 GB, already-downloaded files are skipped
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 25

IDSSE_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
IDSSE_DIR = ROOT / "data" / "external" / "idsse"
CACHE_DIR = IDSSE_DIR / "cache"


@dataclass
class Match:
    name: str
    players: np.ndarray
    team: np.ndarray
    ball: np.ndarray
    segment: np.ndarray
    pitch: tuple
    dead: Optional[np.ndarray] = None
    owner: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        return len(self.ball)


# --------------------------------------------------------------------------------------------
# BAR-ATH (the competition match)
# --------------------------------------------------------------------------------------------

def load_barath() -> Match:
    """The tactical ATD of BAR-ATH. Row i is video frame i (= submission row i)."""
    from games import GAMES, GAME, asset_paths

    paths = asset_paths()
    length, width = GAMES[GAME]["pitch"]

    root = ET.parse(paths["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team = np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)
    n_players = len(team)

    # Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...(46x):<x,y,z>". Flatten every separator to a comma
    # and let pandas parse the numbers: 1 + 3*46 + 3 columns.
    with open(paths["atd_raw"]) as fh:
        text = fh.read().replace(":", ",").replace(";", ",")
    from io import StringIO
    arr = pd.read_csv(StringIO(text), header=None, dtype=np.float64,
                      na_values=["NaN"]).to_numpy()
    del text
    frame_id = arr[:, 0].astype(np.int64)
    if not np.array_equal(frame_id, np.arange(1, len(frame_id) + 1)):
        raise ValueError("ATD frame ids are not 1..N in order")

    pl = arr[:, 1:1 + 3 * n_players].reshape(len(arr), n_players, 3)[:, :, :2]
    ball = arr[:, 1 + 3 * n_players:1 + 3 * n_players + 2]
    scale = np.array([length, width])
    players = (pl * scale).astype(np.float32)
    ball = (ball * scale).astype(np.float32)
    # Two segments, split in the middle of halftime, so that per-half quantities (attacking
    # direction) and rolling windows never mix the halves. Halftime itself is never graded.
    from games import half_video_blocks
    (_, _, end1), (_, start2, _) = half_video_blocks(GAME)
    segment = np.where(np.arange(len(arr)) <= (end1 + start2) // 2, 1, 2).astype(np.int16)
    return Match("BAR-ATH", players, team, ball, segment, (length, width))


# --------------------------------------------------------------------------------------------
# IDSSE (external training data, Bundesliga open data, CC BY 4.0)
# --------------------------------------------------------------------------------------------

def idsse_path(match_id: str, kind: str) -> Path:
    """Local file of one IDSSE match; kind is 'meta', 'tracking' or 'event'."""
    from kloppy._providers.sportec import get_IDSSE_url
    return IDSSE_DIR / get_IDSSE_url(match_id, kind).rsplit("/", 1)[-1]


def download_idsse() -> None:
    import urllib.request
    from kloppy._providers.sportec import get_IDSSE_url
    IDSSE_DIR.mkdir(parents=True, exist_ok=True)
    for match_id in IDSSE_MATCHES:
        for kind in ("meta", "tracking", "event"):
            target = idsse_path(match_id, kind)
            if target.exists() and target.stat().st_size > 0:
                print(f"have  {target.name}")
                continue
            print(f"fetch {target.name}", flush=True)
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(get_IDSSE_url(match_id, kind), tmp)
            tmp.rename(target)


def save_atomic(path: Path, write) -> None:
    """Write a cache file via a temporary name, so that a process running in parallel never reads a
    half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def _idsse_cache(match_id: str) -> Path:
    return CACHE_DIR / f"{match_id}.v2.npz"


def load_idsse(match_id: str) -> Match:
    """One IDSSE match. Parsed once with kloppy and cached as .npz next to the raw files."""
    cache = _idsse_cache(match_id)
    if cache.exists():
        z = np.load(cache)
        return Match(match_id, z["players"], z["team"], z["ball"], z["segment"],
                     tuple(z["pitch"]), z["dead"], z["owner"])

    from kloppy import sportec
    from kloppy.domain import BallState

    ds = sportec.load_tracking(raw_data=str(idsse_path(match_id, "tracking")),
                               meta_data=str(idsse_path(match_id, "meta")),
                               coordinates="kloppy")
    length = ds.metadata.pitch_dimensions.pitch_length
    width = ds.metadata.pitch_dimensions.pitch_width
    home, away = ds.metadata.teams
    plist = list(home.players) + list(away.players)
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    index = {p.player_id: i for i, p in enumerate(plist)}

    T = len(ds.frames)
    players = np.full((T, len(plist), 2), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    dead = np.zeros(T, dtype=bool)
    owner = np.full(T, -1, dtype=np.int8)
    home_id = home.team_id
    segment = np.zeros(T, dtype=np.int16)
    for t, fr in enumerate(ds.frames):
        segment[t] = fr.period.id
        dead[t] = fr.ball_state != BallState.ALIVE
        if fr.ball_owning_team is not None:
            owner[t] = 0 if fr.ball_owning_team.team_id == home_id else 1
        if fr.ball_coordinates is not None:
            ball[t] = (fr.ball_coordinates.x * length, fr.ball_coordinates.y * width)
        for p, pd_ in fr.players_data.items():
            c = pd_.coordinates
            if c is not None:
                players[t, index[p.player_id]] = (c.x * length, c.y * width)

    save_atomic(cache, lambda fh: np.savez_compressed(fh, players=players, team=team, ball=ball, segment=segment,
                                                      pitch=np.array([length, width]), dead=dead, owner=owner))
    return Match(match_id, players, team, ball, segment, (length, width), dead, owner)


if __name__ == "__main__":
    download_idsse()

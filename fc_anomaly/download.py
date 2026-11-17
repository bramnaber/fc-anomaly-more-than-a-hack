"""Download every piece of external data the pipeline uses, into data/external/. Files already there are kept.

    python -m fc_anomaly.download                 # everything (~5 GB)
    python -m fc_anomaly.download soccernet       # or one source: idsse, metrica, skillcorner, soccernet

| source      | what                                                     | size    | used by                    |
|-------------|----------------------------------------------------------|---------|----------------------------|
| idsse       | 7 Bundesliga matches: tracking, events (CC BY 4.0)       | ~2.6 GB | the training matches       |
| metrica     | 3 sample games (acknowledge the source)                  | ~180 MB | degrade.py numbers; a test |
| skillcorner | 20 A-League matches, broadcast tracking (MIT)            | ~2.1 GB | broadcast.py               |
| soccernet   | camera and action labels of 500 games (SoccerNet terms)  | ~93 MB  | broadcast.py               |

SkillCorner and SoccerNet are pinned to the versions we downloaded (13 Nov 2026), so a later change on
their side cannot change our models. The SoccerNet games are listed in fc_anomaly/soccernet_games.txt.
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from pathlib import Path

from fc_anomaly.data import ROOT

EXTERNAL = ROOT / "data" / "external"
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"     # github.com/SkillCorner/opendata, 14 Sep 2026
SOCCERNET_REVISION = "8e01649fe968da9f541c91019b65b3028b2425fc"     # huggingface.co/datasets/SoccerNet/SN-Labels, 7 Sep 2026
SKILLCORNER_RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{SKILLCORNER_COMMIT}/data/"
SKILLCORNER_LFS = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{SKILLCORNER_COMMIT}/data/"
SOCCERNET_URL = f"https://huggingface.co/datasets/SoccerNet/SN-Labels/resolve/{SOCCERNET_REVISION}/"
SOCCERNET_FILES = ("Labels-cameras.json", "Labels-v2.json")


def fetch(url: str, path: Path) -> None:
    """url -> path, unless path exists. Written to a temporary name first, so a broken download never
    looks like a finished file."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fc-anomaly-download"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.rename(path)


def skillcorner(root: Path = EXTERNAL / "skillcorner") -> None:
    """matches.json, and per match the two files skillcorner.py reads."""
    import json
    fetch(SKILLCORNER_RAW + "matches.json", root / "matches.json")
    for m in json.loads((root / "matches.json").read_text()):
        mid = str(m["id"])
        fetch(SKILLCORNER_RAW + f"matches/{mid}/{mid}_phases_of_play.csv", root / mid / f"{mid}_phases_of_play.csv")
        fetch(SKILLCORNER_LFS + f"matches/{mid}/{mid}_tracking_extrapolated.jsonl",     # stored in Git LFS
              root / mid / f"{mid}_tracking_extrapolated.jsonl")
        print(f"skillcorner {mid}", flush=True)


def soccernet(root: Path = EXTERNAL / "soccernet" / "games") -> None:
    games = (ROOT / "fc_anomaly" / "soccernet_games.txt").read_text().splitlines()
    for i, game in enumerate(games):
        for name in SOCCERNET_FILES:
            fetch(SOCCERNET_URL + urllib.parse.quote(f"{game}/{name}"), root / game / name)
        if (i + 1) % 50 == 0:
            print(f"soccernet {i + 1}/{len(games)}", flush=True)


def idsse() -> None:
    from fc_anomaly.data import download_idsse
    download_idsse()


def metrica() -> None:
    from fc_anomaly.metrica import download
    download()


SOURCES = {"idsse": idsse, "metrica": metrica, "skillcorner": skillcorner, "soccernet": soccernet}

if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        SOURCES[name]()

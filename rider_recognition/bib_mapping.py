"""Charge le mapping bib → coureur depuis un JSON de course (format ASO).

Format JSON attendu (cf signatureNG/test.json et exports ASO) :
    {
      "racecode": "TDF",
      "name": "Tour de France",
      "season": 2024,
      "teams": [
        {
          "code": "TVL", "name": "Team Visma | Lease a Bike", "nationality": "NED",
          "riders": [
            {"bib": 1, "lastname": "VINGEGAARD HANSEN", "firstname": "Jonas",
             "uciid": "10011208231", "nationality": "DEN"},
            ...
          ]
        },
        ...
      ]
    }

Sortie : `{bib(int): Rider}` où Rider expose name/uciid/team.

À terme remplacé par un client HTTP vers l'API signatureNG (cf
[[project_rider_recognition]]). Pour le test = lecture fichier local.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rider:
    """Coureur indexé par dossard pour une course donnée."""

    bib: int
    uciid: str
    firstname: str
    lastname: str
    nationality: str
    team_code: str      # ex: "TVL"
    team_name: str      # ex: "Team Visma | Lease a Bike"

    @property
    def display_name(self) -> str:
        """Format affichage broadcast : 'LASTNAME Firstname'."""
        return f"{self.lastname} {self.firstname}"


def load_bib_mapping(json_path: Path) -> dict[int, Rider]:
    """Charge un JSON de course ASO et retourne {bib: Rider}.

    Skip silencieusement les riders sans `bib` (substituts, etc.).
    """
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    out: dict[int, Rider] = {}
    for team in data.get("teams", []):
        for r in team.get("riders", []):
            bib = r.get("bib")
            if bib is None:
                continue
            out[int(bib)] = Rider(
                bib=int(bib),
                uciid=str(r.get("uciid", "")),
                firstname=str(r.get("firstname", "")),
                lastname=str(r.get("lastname", "")),
                nationality=str(r.get("nationality", "")),
                team_code=str(team.get("code", "")),
                team_name=str(team.get("name", "")),
            )
    return out

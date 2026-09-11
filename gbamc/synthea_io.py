"""Readers for Synthea CSV exports (stdlib csv only).

Synthea (https://github.com/synthetichealth/synthea) is Apache-2.0 and
generates realistic but fully synthetic patients without credentials.
Expected files follow the standard Synthea CSV export layout.
"""

from __future__ import annotations

import csv
import os


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_patients(dirpath: str) -> list[dict]:
    return _read_csv(os.path.join(dirpath, "patients.csv"))


def load_conditions(dirpath: str) -> list[dict]:
    return _read_csv(os.path.join(dirpath, "conditions.csv"))


def load_observations(dirpath: str) -> list[dict]:
    return _read_csv(os.path.join(dirpath, "observations.csv"))


def load_medications(dirpath: str) -> list[dict]:
    return _read_csv(os.path.join(dirpath, "medications.csv"))


def load_encounters(dirpath: str) -> list[dict]:
    return _read_csv(os.path.join(dirpath, "encounters.csv"))

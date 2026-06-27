"""Dependency contract checks for import-time APIs used by the scripts."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TRL_WITH_SFT_CONFIG = (0, 9, 2)


def parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractTests(TestCase):
    def test_trl_lower_bound_includes_sft_config(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
        trl_requirement = next(line.strip() for line in requirements if line.startswith("trl"))
        lower_bound = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", trl_requirement)

        self.assertIsNotNone(lower_bound)
        self.assertGreaterEqual(
            parse_version(lower_bound.group(1)),
            MIN_TRL_WITH_SFT_CONFIG,
        )


"""Dependency contract checks for import-time APIs used by the project."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import TestCase


class DependencyContractsTest(TestCase):
    def test_trl_lower_bound_exports_sft_config(self) -> None:
        requirements = Path("requirements.txt").read_text().splitlines()
        trl_requirement = next(
            line.strip()
            for line in requirements
            if line.strip() and not line.strip().startswith("#") and line.startswith("trl")
        )

        match = re.fullmatch(r"trl\s*(>=|==)\s*([0-9]+(?:\.[0-9]+){1,2})", trl_requirement)

        self.assertIsNotNone(match, "TRL must have an explicit minimum supported version")
        self.assertGreaterEqual(
            tuple(int(part) for part in match.group(2).split(".")),
            (0, 9, 2),
            "src.train_lora imports trl.SFTConfig, which is not exported by TRL 0.8.x",
        )

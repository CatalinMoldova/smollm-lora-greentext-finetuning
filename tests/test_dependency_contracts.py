"""Dependency contracts for import-time APIs used by the training scripts."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DependencyContractsTest(unittest.TestCase):
    def test_trl_minimum_version_exports_sft_config(self):
        train_source = (ROOT / "src" / "train_lora.py").read_text()
        imports_sft_config = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "trl"
            and any(alias.name == "SFTConfig" for alias in node.names)
            for node in ast.walk(ast.parse(train_source))
        )
        self.assertTrue(imports_sft_config, "training code should import SFTConfig")

        requirements = (ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+(?:\.[0-9]+)*)$", requirements, re.MULTILINE)
        self.assertIsNotNone(match, "requirements.txt must pin a TRL lower bound")

        minimum_version = tuple(int(part) for part in match.group(1).split("."))
        self.assertGreaterEqual(
            minimum_version,
            (0, 9, 2),
            "TRL versions below 0.9.2 do not export SFTConfig used by train_lora.py",
        )


if __name__ == "__main__":
    unittest.main()

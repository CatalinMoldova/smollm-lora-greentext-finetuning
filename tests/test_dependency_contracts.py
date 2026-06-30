"""Dependency contract tests for imports used by the training code."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _minimum_version(requirements_text: str, package: str) -> tuple[int, ...]:
    pattern = re.compile(rf"^{re.escape(package)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)", re.MULTILINE)
    match = pattern.search(requirements_text)
    if not match:
        raise AssertionError(f"{package!r} must declare a minimum supported version")
    return tuple(int(part) for part in match.group(1).split("."))


class DependencyContractTests(unittest.TestCase):
    def test_trl_minimum_supports_sft_config_import(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text()
        train_source = (ROOT / "src" / "train_lora.py").read_text()

        self.assertIn("from trl import SFTConfig", train_source)
        self.assertGreaterEqual(
            _minimum_version(requirements, "trl"),
            (0, 9, 2),
            "src.train_lora imports trl.SFTConfig, which is not exported by TRL 0.8.x.",
        )


if __name__ == "__main__":
    unittest.main()

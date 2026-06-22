"""Dependency contracts required by the source code."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractTests(unittest.TestCase):
    def test_trl_version_supports_sft_config_import(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+(?:\.[0-9]+)+)$", requirements, re.MULTILINE)

        self.assertIsNotNone(match, "requirements.txt must pin a TRL lower bound")
        self.assertGreaterEqual(version_tuple(match.group(1)), version_tuple("0.9.2"))


if __name__ == "__main__":
    unittest.main()

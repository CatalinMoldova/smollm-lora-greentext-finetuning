"""Dependency contract tests for import-time training requirements."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_requirement_floor(package_name: str) -> tuple[int, ...]:
    requirements = (REPO_ROOT / "requirements.txt").read_text()
    match = re.search(rf"^{re.escape(package_name)}>=([0-9]+(?:\.[0-9]+)*)$", requirements, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{package_name} lower bound is missing from requirements.txt")
    return tuple(int(part) for part in match.group(1).split("."))


class DependencyContractsTest(unittest.TestCase):
    def test_trl_floor_exports_sft_config(self):
        """src.train_lora imports SFTConfig, first exported by TRL 0.9.2."""
        self.assertGreaterEqual(parse_requirement_floor("trl"), (0, 9, 2))


if __name__ == "__main__":
    unittest.main()

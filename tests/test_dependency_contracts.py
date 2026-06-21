"""Dependency contract checks for documented entry points."""

from pathlib import Path
import unittest


def parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_exports_sft_config(self):
        requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
        trl_requirement = next(
            line.strip()
            for line in requirements.read_text().splitlines()
            if line.strip().startswith("trl>=")
        )
        lower_bound = trl_requirement.split(">=", 1)[1]

        self.assertGreaterEqual(parse_version(lower_bound), parse_version("0.9.2"))


if __name__ == "__main__":
    unittest.main()

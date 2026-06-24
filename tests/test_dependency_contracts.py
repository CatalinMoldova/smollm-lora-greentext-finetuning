"""Dependency contract tests for import-time API requirements."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements_by_package() -> dict[str, str]:
    requirements = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        package = re.split(r"[<>=!~]", requirement, maxsplit=1)[0].strip().lower()
        requirements[package] = requirement
    return requirements


def _imports_name(module_path: Path, package: str, name: str) -> bool:
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == package:
            if any(alias.name == name for alias in node.names):
                return True
    return False


class DependencyContractsTest(unittest.TestCase):
    def test_trl_lower_bound_supports_sft_config_import(self) -> None:
        train_lora_path = REPO_ROOT / "src" / "train_lora.py"
        self.assertTrue(_imports_name(train_lora_path, "trl", "SFTConfig"))

        trl_requirement = _requirements_by_package()["trl"]
        self.assertRegex(
            trl_requirement,
            r"^trl>=0\.9\.2$",
            "SFTConfig is first exported by TRL 0.9.2; lower versions crash on import.",
        )


if __name__ == "__main__":
    unittest.main()

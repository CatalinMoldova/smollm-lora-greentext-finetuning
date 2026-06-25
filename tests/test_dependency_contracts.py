"""Dependency contract checks for imported third-party APIs."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirement_floor(package_name: str) -> tuple[int, ...]:
    requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
    pattern = re.compile(rf"^{re.escape(package_name)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*$")

    for requirement in requirements:
        match = pattern.match(requirement.strip())
        if match:
            return tuple(int(part) for part in match.group(1).split("."))

    raise AssertionError(f"No >= requirement found for {package_name!r}")


def _imports_name(module_path: str, name: str) -> bool:
    source = (REPO_ROOT / module_path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "trl":
            if any(alias.name == name for alias in node.names):
                return True
    return False


class DependencyContractTests(unittest.TestCase):
    def test_trl_floor_supports_sft_config_import(self) -> None:
        self.assertTrue(
            _imports_name("src/train_lora.py", "SFTConfig"),
            "Update this test if train_lora.py no longer imports trl.SFTConfig.",
        )
        self.assertGreaterEqual(
            _requirement_floor("trl"),
            (0, 9, 2),
            "trl.SFTConfig is not exported by the allowed 0.8.x releases.",
        )


if __name__ == "__main__":
    unittest.main()

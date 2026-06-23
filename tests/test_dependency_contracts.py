"""Dependency version contracts for imported public APIs."""

from __future__ import annotations

from pathlib import Path

REQUIRED_TRL_VERSION = (0, 9, 2)


def _requirement_named(name: str) -> str:
    requirements = Path("requirements.txt").read_text().splitlines()
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == name or line.startswith(f"{name}>="):
            return line
    raise AssertionError(f"Missing dependency: {name}")


def _version_tuple(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split(".", maxsplit=2))


def test_trl_version_supports_sft_config_import() -> None:
    requirement = _requirement_named("trl")

    assert requirement.startswith("trl>="), "TRL must have a lower bound"
    lower_bound = requirement.removeprefix("trl>=")
    assert (
        _version_tuple(lower_bound) >= REQUIRED_TRL_VERSION
    ), "src.train_lora imports trl.SFTConfig, which is absent from TRL 0.8.x"

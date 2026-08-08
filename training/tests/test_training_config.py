from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

from train_synthetic import validate_training_config  # noqa: E402


def load_config() -> dict[str, object]:
    value = json.loads((TRAINING_ROOT / "configs/hcds31.json").read_text())
    assert isinstance(value, dict)
    return value


def test_shipped_training_config_matches_implemented_contract() -> None:
    validate_training_config(load_config())


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("model", "output_stride", 8),
        ("model", "output_encoding", "semantic"),
        ("targets", "gaussian_combine", "sum"),
        ("targets", "sigma_box_scale", 0.2),
        ("optimizer", "warmup_epochs", 5),
        ("optimizer", "schedule", "constant"),
    ],
)
def test_declarative_contract_drift_is_rejected(
    section: str, key: str, replacement: object
) -> None:
    config = copy.deepcopy(load_config())
    section_value = config[section]
    assert isinstance(section_value, dict)
    section_value[key] = replacement
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        validate_training_config(config)

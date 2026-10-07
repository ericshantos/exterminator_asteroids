from pathlib import Path
from typing import Any, cast

import yaml
from dacite import from_dict

from .schema import ExterminatorConfig


def load_yaml(filename: str | Path) -> dict[str, Any]:
    path = Path(__file__).parent / filename

    with open(path, encoding="utf-8") as file:
        return cast(dict[str, Any], yaml.safe_load(file))


class ConfigLoader:
    @staticmethod
    def load() -> ExterminatorConfig:
        path = Path(__file__).parent / "yaml"

        data = {}

        data.update(load_yaml(path / "environment.yaml"))
        data.update(load_yaml(path / "dqn.yaml"))
        data.update(load_yaml(path / "ppo.yaml"))
        data.update(load_yaml(path / "training.yaml"))

        return cast(
            ExterminatorConfig, from_dict(data_class=ExterminatorConfig, data=data)
        )

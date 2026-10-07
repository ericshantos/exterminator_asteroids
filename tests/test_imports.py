import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ["entities", "env", "agent", "arena", "configs", "rendering", "training"]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports_first_in_a_fresh_interpreter(package: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {package}"],
        cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

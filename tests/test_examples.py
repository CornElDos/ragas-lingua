"""Keep the examples from rotting: compile them all, run the no-API-key ones."""

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_OFFLINE = ["04_from_ragas.py", "05_add_a_language.py", "06_offline_with_fakejudge.py"]


def test_all_examples_compile():
    files = sorted(_EXAMPLES.glob("*.py"))
    assert files, "no example scripts found"
    for path in files:
        py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("name", _OFFLINE)
def test_offline_example_runs(name):
    # Run in a subprocess so registering a demo language doesn't leak into the suite.
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES / name)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

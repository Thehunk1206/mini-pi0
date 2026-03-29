import tempfile
from pathlib import Path

from mini_pi0.utils.runs import create_run_dir


def test_create_run_dir_is_sequential():
    with tempfile.TemporaryDirectory() as d:
        p1 = create_run_dir(d, "exp")
        p2 = create_run_dir(d, "exp")
        p3 = create_run_dir(d, "exp")

        assert p1.name == "run1"
        assert p2.name == "run2"
        assert p3.name == "run3"

        assert (Path(d) / "exp" / "run1").exists()
        assert (Path(d) / "exp" / "run2").exists()
        assert (Path(d) / "exp" / "run3").exists()


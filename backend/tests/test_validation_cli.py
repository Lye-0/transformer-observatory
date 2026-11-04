import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["validate_model.py", "validate_tuned_lens.py"])
def test_validation_cli_resolves_relative_model_path(tmp_path, monkeypatch, script):
    from transformer_observatory import models

    script_path = Path(__file__).resolve().parents[2] / "scripts" / script
    monkeypatch.chdir(tmp_path)
    received = []

    class RegistrationReached(Exception):
        pass

    def register(path, *args):
        received.append(path)
        raise RegistrationReached()

    monkeypatch.setattr(models, "register_model", register)
    args = [script, "--model", "relative-model", "--output", str(tmp_path / "output")]
    if script == "validate_tuned_lens.py":
        args += ["--lens", "relative-lens"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(RegistrationReached):
        runpy.run_path(str(script_path), run_name="__main__")
    assert received == [str(tmp_path / "relative-model")]

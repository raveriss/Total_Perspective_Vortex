import sys
from typing import Any

import mybci


def test_parse_args_returns_expected_namespace():
    args = mybci.parse_args(["S01", "R01", "train"])

    assert args.subject == "S01"
    assert args.run == "R01"
    assert args.mode == "train"


def test_main_invokes_train_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, subject: str, run: str) -> int:
        called["args"] = (module_name, subject, run)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S01", "R02", "train"])

    assert exit_code == 0
    assert called["args"] == ("tpv.train", "S01", "R02")


def test_main_invokes_predict_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, subject: str, run: str) -> int:
        called["args"] = (module_name, subject, run)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S02", "R03", "predict"])

    assert exit_code == 0
    assert called["args"] == ("tpv.predict", "S02", "R03")


def test_call_module_executes_python_module(monkeypatch):
    recorded_command: list[str] = []

    class DummyCompletedProcess:
        returncode = 0

    def fake_run(command: list[str], check: bool) -> DummyCompletedProcess:
        recorded_command.extend(command)
        assert check is False
        return DummyCompletedProcess()

    monkeypatch.setattr(mybci.subprocess, "run", fake_run)

    exit_code = mybci._call_module("tpv.train", "S03", "R04")

    assert exit_code == 0
    assert recorded_command == [sys.executable, "-m", "tpv.train", "S03", "R04"]

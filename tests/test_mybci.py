import sys
from typing import Any

import pytest

import mybci

EXIT_USAGE = 2


def test_parse_args_returns_expected_namespace():
    args = mybci.parse_args(["S01", "R01", "train"])

    assert args.subject == "S01"
    assert args.run == "R01"
    assert args.mode == "train"


def test_build_parser_metadata():
    parser = mybci.build_parser()

    assert (
        parser.description == "Pilote un workflow d'entraînement ou de prédiction TPV"
    )
    assert (
        parser.usage or ""
    ).strip() == "python mybci.py <subject> <run> {train,predict}"


def test_build_parser_defines_expected_arguments():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    subject_arg = get_action("subject")
    run_arg = get_action("run")
    mode_arg = get_action("mode")

    assert subject_arg.help == "Identifiant du sujet (ex: S01)"
    assert run_arg.help == "Identifiant du run (ex: R01)"
    assert mode_arg.help == "Choix du pipeline à lancer"
    assert tuple(mode_arg.choices) == ("train", "predict")


def test_parse_args_rejects_invalid_mode(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(["S01", "R01", "invalid"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr


def test_parse_args_requires_all_positional_arguments(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(["S01", "R01"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "usage: python mybci.py <subject> <run> {train,predict}" in stderr


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

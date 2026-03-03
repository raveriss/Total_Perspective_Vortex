# Vérifie le script sanitizer et ses sorties structurées.

import json
import sys
from pathlib import Path

import pytest

from scripts import sanitizer


def _build_config(tmp_path: Path, probes: tuple[str, ...]) -> sanitizer.SanitizerConfig:
    return sanitizer.SanitizerConfig(
        command=(sys.executable, "-c", "print(42)"),
        direct_python_command=(sys.executable, "-c", "print(42)"),
        output_dir=tmp_path,
        selected_probes=probes,
        timeout_seconds=5,
        pyperf_warmups=1,
        pyperf_runs=1,
        hyperfine_warmups=1,
        hyperfine_runs=1,
        cpu_core=0,
        ps_interval_seconds=0.1,
        psrecord_interval_seconds=0.1,
        memory_limit_kib=1024,
        time_csv_runs=2,
    )


def test_infer_python_command_supports_make_mybci_wavelet() -> None:
    inferred = sanitizer.infer_python_command(["make", "-j1", "mybci", "wavelet"])

    assert inferred == [
        sys.executable,
        "mybci.py",
        "--feature-strategy",
        "wavelet",
    ]


def test_infer_python_command_supports_compute_mean_of_means() -> None:
    inferred = sanitizer.infer_python_command(["make", "compute-mean-of-means"])

    assert inferred == [sys.executable, "scripts/aggregate_experience_scores.py"]


def test_run_probe_skips_when_hyperfine_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_config(tmp_path, ("A4",))
    catalog = {
        probe.identifier: probe for probe in sanitizer._build_probe_catalog()
    }
    monkeypatch.setattr(
        sanitizer,
        "_command_available",
        lambda name: False if name == "hyperfine" else True,
    )

    result = sanitizer._run_probe(config, catalog["A4"])

    assert result.status == "skipped"
    assert result.action is not None
    assert "hyperfine" in result.action
    assert Path(result.output_dir, "result.json").exists()


def test_main_runs_a2_and_writes_csv_jsonl_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "sanitizer"

    exit_code = sanitizer.main(
        [
            "--output-dir",
            str(output_dir),
            "--probe",
            "A2",
            "--time-csv-runs",
            "2",
            "--timeout-seconds",
            "5",
            "--",
            sys.executable,
            "-c",
            "print(42)",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "A2" / "time.csv").exists()
    assert (output_dir / "A2" / "time.jsonl").exists()
    assert (output_dir / "A2" / "time_summary.json").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall_status"] == "ok"
    assert summary["results"][0]["identifier"] == "A2"


def test_main_runs_h2_and_captures_meta(tmp_path: Path) -> None:
    output_dir = tmp_path / "meta"

    exit_code = sanitizer.main(
        [
            "--output-dir",
            str(output_dir),
            "--probe",
            "H2",
            "--",
            sys.executable,
            "-c",
            "print(1)",
        ]
    )

    assert exit_code == 0
    meta = json.loads((output_dir / "H2" / "meta.json").read_text(encoding="utf-8"))
    assert meta["target_command"] == [sys.executable, "-c", "print(1)"]
    assert meta["python"] is not None


def test_build_probe_catalog_returns_shell_commands_for_all_builders(
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path, ())

    for probe in sanitizer._build_probe_catalog():
        if probe.command_builder is None:
            continue
        command = probe.command_builder(config, tmp_path / probe.identifier)
        assert isinstance(command, str)
        assert command


def test_preflight_helpers_report_missing_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_config(tmp_path, ("A4",))
    missing_path = tmp_path / "missing-tool"
    monkeypatch.setattr(sanitizer, "_command_available", lambda _name: False)
    monkeypatch.setattr(sanitizer, "_python_module_available", lambda _name: False)

    command_result = sanitizer._require_command("hyperfine", "install hyperfine")(config)
    module_result = sanitizer._require_python_module("pyperf", "install pyperf")(config)
    path_result = sanitizer._require_path(missing_path, "create path")(config)
    poetry_result = sanitizer._require_poetry("install poetry")(config)
    direct_python_result = sanitizer._require_direct_python("pass command")(
        sanitizer.SanitizerConfig(
            command=("make", "-j1", "mybci"),
            direct_python_command=None,
            output_dir=tmp_path,
            selected_probes=("F1",),
            timeout_seconds=5,
            pyperf_warmups=1,
            pyperf_runs=1,
            hyperfine_warmups=1,
            hyperfine_runs=1,
            cpu_core=0,
            ps_interval_seconds=0.1,
            psrecord_interval_seconds=0.1,
            memory_limit_kib=1024,
            time_csv_runs=1,
        )
    )
    merged = sanitizer._merge_preflights(command_result, module_result)

    assert not command_result.ready
    assert not module_result.ready
    assert not path_result.ready
    assert not poetry_result.ready
    assert not direct_python_result.ready
    assert not merged.ready


def test_resolve_python_invocation_supports_poetry_and_rejects_non_python() -> None:
    poetry_python = sanitizer._resolve_python_invocation(
        ["poetry", "run", sys.executable, "-c", "print(1)"]
    )
    not_python = sanitizer._resolve_python_invocation(["make", "mybci"])

    assert poetry_python == (["poetry", "run"], [sys.executable, "-c", "print(1)"])
    assert not_python is None


def test_build_cprofile_command_skips_non_python_direct_command(tmp_path: Path) -> None:
    config = sanitizer.SanitizerConfig(
        command=("make", "-j1", "mybci"),
        direct_python_command=("make", "mybci"),
        output_dir=tmp_path,
        selected_probes=("F1",),
        timeout_seconds=5,
        pyperf_warmups=1,
        pyperf_runs=1,
        hyperfine_warmups=1,
        hyperfine_runs=1,
        cpu_core=0,
        ps_interval_seconds=0.1,
        psrecord_interval_seconds=0.1,
        memory_limit_kib=1024,
        time_csv_runs=1,
    )

    command = sanitizer._build_cprofile_command(config, tmp_path / "F1")

    assert "non compatible" in command
    assert str(sanitizer.SKIP_EXIT_CODE) in command


def test_run_shell_probe_marks_timeout_as_warn(tmp_path: Path) -> None:
    config = sanitizer.SanitizerConfig(
        command=(sys.executable, "-c", "print(0)"),
        direct_python_command=(sys.executable, "-c", "print(0)"),
        output_dir=tmp_path,
        selected_probes=("custom-timeout",),
        timeout_seconds=1,
        pyperf_warmups=1,
        pyperf_runs=1,
        hyperfine_warmups=1,
        hyperfine_runs=1,
        cpu_core=0,
        ps_interval_seconds=0.1,
        psrecord_interval_seconds=0.1,
        memory_limit_kib=1024,
        time_csv_runs=1,
    )
    probe = sanitizer.ProbeDefinition(
        identifier="custom-timeout",
        title="timeout",
        category="Z",
        highlights=(),
        command_builder=lambda _config, _dir: "sleep 2",
    )

    result = sanitizer._run_shell_probe(config, probe)

    assert result.status == "warn"
    assert "Timeout" in result.summary


def test_write_session_summary_writes_markdown_and_json(tmp_path: Path) -> None:
    config = _build_config(tmp_path, ("A2",))
    result = sanitizer.ProbeResult(
        identifier="A2",
        title="time csv",
        category="A",
        status="ok",
        summary="done",
        action=None,
        exit_code=0,
        command="echo ok",
        output_dir=str(tmp_path / "A2"),
        stdout_log=None,
        stderr_log=None,
    )

    overall_status, summary_json, summary_md = sanitizer._write_session_summary(
        config=config,
        results=[result],
        started_at=sanitizer.datetime.utcnow(),
        finished_at=sanitizer.datetime.utcnow(),
    )

    assert overall_status == "ok"
    assert summary_json.exists()
    assert summary_md.exists()


def test_misc_helper_branches_are_exercised(tmp_path: Path) -> None:
    config = sanitizer.SanitizerConfig(
        command=("make", "-j1", "mybci"),
        direct_python_command=None,
        output_dir=tmp_path,
        selected_probes=("demo",),
        timeout_seconds=7,
        pyperf_warmups=2,
        pyperf_runs=4,
        hyperfine_warmups=3,
        hyperfine_runs=5,
        cpu_core=1,
        ps_interval_seconds=0.1,
        psrecord_interval_seconds=0.1,
        memory_limit_kib=1024,
        time_csv_runs=1,
    )
    probe = sanitizer.ProbeDefinition(
        identifier="demo",
        title="demo",
        category="Z",
        highlights=(),
        command_builder=lambda _config, _dir: "true",
        timeout_resolver=lambda current: current.timeout_seconds + 1,
    )

    assert sanitizer._with_poetry(("poetry", "run", "make")) == (
        "poetry",
        "run",
        "make",
    )
    assert sanitizer._preferred_profile_shell(config) == "make -j1 mybci"
    assert sanitizer._probe_timeout(config, probe) == 8
    assert sanitizer._extract_tagged_line("SUMMARY: ok", "SUMMARY:") == "ok"
    assert sanitizer._extract_tagged_line("plain text", "SUMMARY:") is None
    assert sanitizer._normalize_process_output(b"hello") == "hello"
    assert sanitizer._timeout_for_pyperf(config) == 42
    assert sanitizer._timeout_for_hyperfine(config) == 56


def test_infer_python_command_covers_remaining_make_targets() -> None:
    assert sanitizer.infer_python_command([]) is None
    assert sanitizer.infer_python_command(["python", "tool.py"]) == [
        "python",
        "tool.py",
    ]
    assert sanitizer.infer_python_command(["make", "-j1"]) is None
    assert sanitizer.infer_python_command(["make", "train", "1", "6", "wavelet"]) == [
        sys.executable,
        "scripts/train.py",
        "1",
        "6",
        "--feature-strategy",
        "wavelet",
    ]
    assert sanitizer.infer_python_command(["make", "visualizer", "1", "6"]) == [
        sys.executable,
        "scripts/visualize_raw_filtered.py",
        "1",
        "6",
    ]
    assert sanitizer.infer_python_command(["make", "realtime", "1", "6"]) == [
        sys.executable,
        "src/tpv/realtime.py",
        "1",
        "6",
    ]


def test_main_list_probes_and_unknown_probe_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sanitizer.main(["--list-probes"]) == 0
    listed = capsys.readouterr().out
    assert "A1.make" in listed

    with pytest.raises(SystemExit):
        sanitizer.main(
            [
                "--output-dir",
                str(tmp_path),
                "--probe",
                "UNKNOWN",
                "--",
                sys.executable,
                "-c",
                "print(1)",
            ]
        )


def test_run_a2_probe_handles_preflight_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config(tmp_path, ("A2",))
    catalog = {
        probe.identifier: probe for probe in sanitizer._build_probe_catalog()
    }
    monkeypatch.setattr(
        sanitizer,
        "_require_path",
        lambda _path, _action: (lambda _config: sanitizer.PreflightResult(False, "skip", "fix")),
    )

    result = sanitizer._run_a2_probe(config, catalog["A2"])

    assert result.status == "skipped"
    assert result.action == "fix"


def test_run_a2_probe_marks_missing_rows_and_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _build_config(tmp_path, ("A2",))
    catalog = {
        probe.identifier: probe for probe in sanitizer._build_probe_catalog()
    }
    results = iter(
        [
            sanitizer.TimedRunResult(stdout="one", stderr="", row=None, action=None),
            sanitizer.TimedRunResult(
                stdout="two",
                stderr="",
                row=sanitizer.TimeCsvRow(
                    run=2,
                    wall_s=0.1,
                    user_s=0.1,
                    sys_s=0.0,
                    cpu="10%",
                    rss_kb=12,
                    exit=3,
                ),
            ),
        ]
    )
    monkeypatch.setattr(
        sanitizer,
        "_run_a2_iteration",
        lambda **_kwargs: next(results),
    )

    result = sanitizer._run_a2_probe(config, catalog["A2"])

    assert result.status == "warn"
    assert result.action is not None
    assert "code non nul" in result.action


def test_write_session_summary_includes_next_actions(tmp_path: Path) -> None:
    config = _build_config(tmp_path, ("A2",))
    result = sanitizer.ProbeResult(
        identifier="A2",
        title="time csv",
        category="A",
        status="warn",
        summary="needs action",
        action="fix me",
        exit_code=1,
        command="echo ko",
        output_dir=str(tmp_path / "A2"),
        stdout_log=None,
        stderr_log=None,
    )

    _, _, summary_md = sanitizer._write_session_summary(
        config=config,
        results=[result],
        started_at=sanitizer.datetime.utcnow(),
        finished_at=sanitizer.datetime.utcnow(),
    )

    assert "## Next Actions" in summary_md.read_text(encoding="utf-8")

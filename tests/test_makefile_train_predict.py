import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fake_poetry(script_path: Path, body: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "#!/bin/bash\nset -euo pipefail\n" + body,
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def write_blocked_script(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('blocked')\n", encoding="utf-8")
    script_path.chmod(0)


def run_make_goals_with_blocked_tpv_dir(
    tmp_path: Path,
    goals: list[str],
    *,
    fake_poetry_body: str,
    make_vars: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    blocked_tpv_dir.mkdir()
    blocked_tpv_dir.chmod(0)
    try:
        effective_make_vars = {"TPV_SRC_DIR": str(blocked_tpv_dir)}
        if make_vars is not None:
            effective_make_vars.update(make_vars)
        return run_make_goals(
            tmp_path,
            goals,
            fake_poetry_body=fake_poetry_body,
            make_vars=effective_make_vars,
        )
    finally:
        blocked_tpv_dir.chmod(0o755)


def run_make_goals_with_blocked_src_dir(
    tmp_path: Path,
    goals: list[str],
    *,
    fake_poetry_body: str,
) -> subprocess.CompletedProcess[str]:
    blocked_src_dir = tmp_path / "blocked-src"
    blocked_src_dir.mkdir()
    blocked_src_dir.chmod(0)
    try:
        return run_make_goals(
            tmp_path,
            goals,
            fake_poetry_body=fake_poetry_body,
            make_vars={
                "SRC_DIR": str(blocked_src_dir),
                "TPV_SRC_DIR": str(blocked_src_dir / "tpv"),
            },
        )
    finally:
        blocked_src_dir.chmod(0o755)


def run_make_goals_with_blocked_artifacts_dir(
    tmp_path: Path,
    goals: list[str],
    *,
    fake_poetry_body: str,
    block_subject_dir: bool = False,
) -> subprocess.CompletedProcess[str]:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    blocked_dir = artifacts_dir
    if block_subject_dir:
        blocked_dir = artifacts_dir / "S001"
        blocked_dir.mkdir()
    blocked_dir.chmod(0)
    try:
        return run_make_goals(
            tmp_path,
            goals,
            fake_poetry_body=fake_poetry_body,
            make_vars={
                "ARTIFACTS_DIR": str(artifacts_dir),
                "BENCH_DIR": str(artifacts_dir / "benchmarks"),
            },
        )
    finally:
        blocked_dir.chmod(0o755)


def run_make_goals(
    tmp_path: Path,
    goals: list[str],
    *,
    fake_poetry_body: str,
    fake_sudo_body: str | None = None,
    make_vars: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_poetry = tmp_path / "bin" / "fake-poetry"
    write_fake_poetry(fake_poetry, fake_poetry_body)
    if fake_sudo_body is not None:
        write_fake_poetry(tmp_path / "bin" / "sudo", fake_sudo_body)
    env = os.environ.copy()
    env["PATH"] = f"{fake_poetry.parent}:{env['PATH']}"
    make_args = [
        "make",
        "--no-print-directory",
        "-C",
        str(REPO_ROOT),
        *goals,
        f"POETRY={fake_poetry}",
        f"BENCH_DIR={tmp_path / 'benchmarks'}",
        "VENV_PY=/bin/true",
        "STAMP=/bin/true",
        "PYPROJECT=/bin/true",
        "LOCKFILE=/bin/true",
    ]
    if make_vars is not None:
        make_args.extend(f"{name}={value}" for name, value in make_vars.items())
    return subprocess.run(
        make_args,
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def run_make_target(
    tmp_path: Path,
    target: str,
    *,
    fake_poetry_body: str,
    make_vars: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_make_goals(
        tmp_path,
        [target, "1", "6"],
        fake_poetry_body=fake_poetry_body,
        make_vars=make_vars,
    )


def test_make_train_hides_make_footer_for_handled_cli_error(tmp_path: Path) -> None:
    result = run_make_target(
        tmp_path,
        "train",
        fake_poetry_body="""
printf 'INFO: lecture EDF impossible pour S001 R06\n'
printf 'Action: donnez les droits de lecture aux fichiers nécessaires : `chmod a+r data/S001/S001R06.edf data/S001/S001R06.edf.event`\n'
exit 2
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture EDF impossible pour S001 R06" in combined_output
    assert "Action: donnez les droits de lecture" in combined_output
    assert "make: ***" not in combined_output


def test_make_predict_keeps_make_footer_for_unexpected_failure(tmp_path: Path) -> None:
    result = run_make_target(
        tmp_path,
        "predict",
        fake_poetry_body="""
printf 'boom\n' >&2
exit 7
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "boom" in combined_output
    assert "predict] Error 7" in combined_output


def test_make_mybci_hides_make_footer_for_handled_cli_error(tmp_path: Path) -> None:
    result = run_make_goals(
        tmp_path,
        ["mybci"],
        fake_poetry_body="""
printf 'INFO: lecture EDF impossible pour S001 R06\n'
printf 'Action: donnez les droits de lecture aux fichiers nécessaires : `chmod a+r data/S001/S001R06.edf data/S001/S001R06.edf.event`\n'
exit 2
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture EDF impossible pour S001 R06" in combined_output
    assert "Action: donnez les droits de lecture" in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_hides_make_footer_for_handled_cli_error(
    tmp_path: Path,
) -> None:
    result = run_make_goals(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
if [[ "$*" != "python mybci.py --feature-strategy wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'INFO: lecture EDF impossible pour S001 R06\n'
printf 'Action: donnez les droits de lecture aux fichiers nécessaires : `chmod a+r data/S001/S001R06.edf data/S001/S001R06.edf.event`\n'
exit 2
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture EDF impossible pour S001 R06" in combined_output
    assert "unexpected args" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_hides_make_footer_for_data_directory_permission_error(
    tmp_path: Path,
) -> None:
    result = run_make_goals(
        tmp_path,
        ["mybci"],
        fake_poetry_body=r"""
printf 'INFO: lecture du dossier data impossible\n'
printf '%s\n' "Action: donnez les droits d'accès au dossier data : \`chmod a+rx data\`"
exit 2
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture du dossier data impossible" in combined_output
    assert "chmod a+rx data" in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_hides_make_footer_for_data_directory_permission_error(
    tmp_path: Path,
) -> None:
    result = run_make_goals(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body=r"""
if [[ "$*" != "python mybci.py --feature-strategy wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'INFO: lecture du dossier data impossible\n'
printf '%s\n' "Action: donnez les droits d'accès au dossier data : \`chmod a+rx data\`"
exit 2
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture du dossier data impossible" in combined_output
    assert "unexpected args" not in combined_output
    assert "chmod a+rx data" in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_skips_bench_log_when_sanitizer_mode_is_enabled(
    tmp_path: Path,
) -> None:
    bench_dir = tmp_path / "benchmarks"
    result = run_make_goals(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
if [[ "$*" != "python mybci.py --feature-strategy wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'sanitizer run\n'
exit 0
""",
        make_vars={"TPV_SANITIZER": "1"},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "sanitizer run" in combined_output
    assert not bench_dir.exists()
    assert "make: ***" not in combined_output


def test_make_mybci_reports_artifacts_subject_directory_permission_error(
    tmp_path: Path,
) -> None:
    result = run_make_goals_with_blocked_artifacts_dir(
        tmp_path,
        ["mybci"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        block_subject_dir=True,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture du dossier" in combined_output
    assert "artifacts/S001 impossible" in combined_output
    assert "chmod a+rx" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_reports_artifacts_root_permission_error(
    tmp_path: Path,
) -> None:
    result = run_make_goals_with_blocked_artifacts_dir(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "INFO: lecture du dossier" in combined_output
    assert "artifacts impossible" in combined_output
    assert "chmod a+rx" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_reports_script_permission_error(tmp_path: Path) -> None:
    blocked_script = tmp_path / "blocked-mybci.py"
    write_blocked_script(blocked_script)

    result = run_make_goals(
        tmp_path,
        ["mybci"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={"MYBCI_SCRIPT": str(blocked_script)},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_reports_script_permission_error(tmp_path: Path) -> None:
    blocked_script = tmp_path / "blocked-mybci-wavelet.py"
    write_blocked_script(blocked_script)

    result = run_make_goals(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={"MYBCI_SCRIPT": str(blocked_script)},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_realtime_reports_script_permission_error(tmp_path: Path) -> None:
    blocked_script = tmp_path / "blocked-realtime.py"
    write_blocked_script(blocked_script)

    result = run_make_target(
        tmp_path,
        "realtime",
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={"REALTIME_SCRIPT": str(blocked_script)},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_visualizer_reports_script_permission_error(tmp_path: Path) -> None:
    blocked_script = tmp_path / "blocked-visualizer.py"
    write_blocked_script(blocked_script)

    result = run_make_target(
        tmp_path,
        "visualizer",
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={"VISUALIZER_SCRIPT": str(blocked_script)},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_compute_mean_of_means_reports_script_permission_error(
    tmp_path: Path,
) -> None:
    blocked_script = tmp_path / "blocked-aggregate.py"
    write_blocked_script(blocked_script)

    result = run_make_goals(
        tmp_path,
        ["compute-mean-of-means"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={
            "AGGREGATE_EXPERIENCE_SCORES_SCRIPT": str(blocked_script),
        },
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_train_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["train", "1", "6"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_predict_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["predict", "1", "6"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["mybci"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_realtime_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["realtime", "1", "6"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_visualizer_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["visualizer", "1", "6"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_compute_mean_of_means_reports_tpv_source_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_tpv_dir = tmp_path / "blocked-tpv"
    result = run_make_goals_with_blocked_tpv_dir(
        tmp_path,
        ["compute-mean-of-means"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_tpv_dir} impossible" in combined_output
    assert f"chmod -R a+rX {blocked_tpv_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_reports_src_directory_permission_error(tmp_path: Path) -> None:
    blocked_src_dir = tmp_path / "blocked-src"
    result = run_make_goals_with_blocked_src_dir(
        tmp_path,
        ["mybci"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_src_dir} impossible" in combined_output
    assert f"chmod a+rx {blocked_src_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_mybci_wavelet_reports_src_directory_permission_error(
    tmp_path: Path,
) -> None:
    blocked_src_dir = tmp_path / "blocked-src"
    result = run_make_goals_with_blocked_src_dir(
        tmp_path,
        ["mybci", "wavelet"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du dossier {blocked_src_dir} impossible" in combined_output
    assert f"chmod a+rx {blocked_src_dir}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output


def test_make_sanitizer_relays_default_command_to_script(tmp_path: Path) -> None:
    result = run_make_goals(
        tmp_path,
        ["sanitizer"],
        fake_poetry_body="""
if [[ "$*" != "-- python scripts/sanitizer.py -- make -j1 mybci wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'sanitizer ok\n'
""",
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "sanitizer ok" in combined_output
    assert "unexpected args" not in combined_output


def test_make_sanitizer_relays_probe_args_to_script(tmp_path: Path) -> None:
    result = run_make_goals(
        tmp_path,
        ["sanitizer"],
        fake_poetry_body="""
if [[ "$*" != "-- python scripts/sanitizer.py --probe A3.make --pyperf-warmups 1 --pyperf-runs 1 -- make -j1 mybci wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'sanitizer args ok\n'
""",
        make_vars={
            "SANITIZER_ARGS": "--probe A3.make --pyperf-warmups 1 --pyperf-runs 1"
        },
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "sanitizer args ok" in combined_output
    assert "unexpected args" not in combined_output


def test_make_sanitizer_prepares_sudo_and_flag_for_privileged_mode(
    tmp_path: Path,
) -> None:
    sudo_log = tmp_path / "sudo.log"

    result = run_make_goals(
        tmp_path,
        ["sanitizer"],
        fake_poetry_body="""
if [[ "$*" != "-- python scripts/sanitizer.py --allow-privileged-tools --probe P1.make -- make -j1 mybci wavelet" ]]; then
  printf 'unexpected args: %s\n' "$*" >&2
  exit 9
fi
printf 'sanitizer privileged ok\n'
""",
        fake_sudo_body=f"""
printf '%s\\n' "$*" >> {sudo_log}
exit 0
""",
        make_vars={
            "SANITIZER_ALLOW_PRIVILEGED_TOOLS": "1",
            "SANITIZER_ARGS": "--probe P1.make",
        },
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "sanitizer privileged ok" in combined_output
    assert "unexpected args" not in combined_output
    assert sudo_log.read_text(encoding="utf-8").strip() == "-v"


def test_make_sanitizer_reports_script_permission_error(tmp_path: Path) -> None:
    blocked_script = tmp_path / "blocked-sanitizer.py"
    write_blocked_script(blocked_script)

    result = run_make_goals(
        tmp_path,
        ["sanitizer"],
        fake_poetry_body="""
printf 'poetry should not run\n'
exit 99
""",
        make_vars={"SANITIZER_SCRIPT": str(blocked_script)},
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert f"INFO: lecture du script {blocked_script} impossible" in combined_output
    assert f"chmod a+rwx {blocked_script}" in combined_output
    assert "poetry should not run" not in combined_output
    assert "make: ***" not in combined_output

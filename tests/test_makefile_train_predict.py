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


def run_make_target(
    tmp_path: Path,
    target: str,
    *,
    fake_poetry_body: str,
) -> subprocess.CompletedProcess[str]:
    fake_poetry = tmp_path / "bin" / "fake-poetry"
    write_fake_poetry(fake_poetry, fake_poetry_body)
    env = os.environ.copy()
    env["PATH"] = f"{fake_poetry.parent}:{env['PATH']}"
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(REPO_ROOT),
            target,
            "1",
            "6",
            f"POETRY={fake_poetry}",
            "VENV_PY=/bin/true",
            "STAMP=/bin/true",
            "PYPROJECT=/bin/true",
            "LOCKFILE=/bin/true",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
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

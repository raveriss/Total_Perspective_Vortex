import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fake_python(script_path: Path, args_file: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"printf '%s\\n' \"$@\" > {args_file}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def run_make_download_dataset(
    tmp_path: Path, *extra_make_args: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    args_file = tmp_path / "python_args.txt"
    write_fake_python(fake_bin / "python3", args_file)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(REPO_ROOT),
            "download_dataset",
            "EEGMMIDB_DATA_DIR=fake-data",
            "EEGMMIDB_SUBJECT_COUNT=3",
            "EEGMMIDB_RUN_COUNT=2",
            *extra_make_args,
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def test_make_download_dataset_delegates_to_python_script(tmp_path: Path) -> None:
    result = run_make_download_dataset(tmp_path)

    assert result.returncode == 0
    recorded_args = (
        (tmp_path / "python_args.txt").read_text(encoding="utf-8").splitlines()
    )
    assert recorded_args[0] == "scripts/download_dataset.py"
    assert "--destination" in recorded_args
    assert "fake-data" in recorded_args
    assert "--subject-count" in recorded_args
    assert "3" in recorded_args
    assert "--run-count" in recorded_args
    assert "2" in recorded_args


def test_make_download_dataset_ignores_url_overrides(tmp_path: Path) -> None:
    result = run_make_download_dataset(
        tmp_path,
        "EEGMMIDB_URL=https://physionet.org/files/eegmmidb/9.9.9/",
        "EEGMMIDB_ALLOW_UNSAFE_URL=1",
    )

    assert result.returncode == 0
    recorded_args = (tmp_path / "python_args.txt").read_text(encoding="utf-8")
    assert "9.9.9" not in recorded_args
    assert "ALLOW_UNSAFE" not in recorded_args

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fake_wget(script_path: Path, body: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "#!/bin/bash\nset -euo pipefail\n" + body,
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def run_make_download_dataset(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    data_dir = tmp_path / "data"
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(REPO_ROOT),
            "download_dataset",
            f"EEGMMIDB_DATA_DIR={data_dir}",
            "EEGMMIDB_SUBJECT_COUNT=1",
            "EEGMMIDB_RUN_COUNT=1",
            "EEGMMIDB_URL=https://example.invalid/files/eegmmidb/1.0.0/",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def test_download_dataset_accepts_wget_error_when_dataset_is_complete(
    tmp_path: Path,
) -> None:
    fake_wget = tmp_path / "bin" / "wget"
    write_fake_wget(
        fake_wget,
        """
dest=""
while (($#)); do
    if [[ "$1" == "-P" ]]; then
        dest="$2"
        shift 2
        continue
    fi
    shift
done
mkdir -p "$dest/S001"
printf 'edf' > "$dest/S001/S001R01.edf"
printf 'event' > "$dest/S001/S001R01.edf.event"
exit 8
""",
    )

    result = run_make_download_dataset(tmp_path)

    assert result.returncode == 0
    assert "wget a retourné 8" in result.stderr
    assert (tmp_path / "data" / ".eegmmidb.ok").exists()


def test_download_dataset_fails_when_wget_errors_and_files_are_missing(
    tmp_path: Path,
) -> None:
    fake_wget = tmp_path / "bin" / "wget"
    write_fake_wget(
        fake_wget,
        """
exit 8
""",
    )

    result = run_make_download_dataset(tmp_path)

    assert result.returncode != 0
    assert "toujours incomplet" in result.stderr
    assert not (tmp_path / "data" / ".eegmmidb.ok").exists()

import subprocess
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import download_dataset


def test_collect_runtime_network_diagnostics_reports_network_unreachable() -> None:
    command_results = {
        ("ping", "-c", "1", "1.1.1.1"): subprocess.CompletedProcess(
            args=["ping", "-c", "1", "1.1.1.1"],
            returncode=2,
            stdout="",
            stderr="ping: connect: Network is unreachable\n",
        ),
        ("getent", "hosts", "physionet.org"): subprocess.CompletedProcess(
            args=["getent", "hosts", "physionet.org"],
            returncode=2,
            stdout="",
            stderr="",
        ),
    }

    def fake_runner(*args, **kwargs):
        del kwargs
        return command_results[tuple(args[0])]

    lines = download_dataset.collect_runtime_network_diagnostics(runner=fake_runner)

    assert "Diagnostic local automatique:" in lines[0]
    assert "- ping -c 1 1.1.1.1 -> code retour 2" in lines
    assert "  stderr: ping: connect: Network is unreachable" in lines
    assert "- getent hosts physionet.org -> code retour 2" in lines
    assert "  sortie: aucune réponse" in lines
    assert "Cause probable: la machine n'a plus d'accès réseau sortant." in lines


def test_select_official_source_falls_back_to_second_candidate() -> None:
    attempted: list[str] = []

    def fake_probe(url: str, opener=None) -> None:
        attempted.append(url)
        if url == download_dataset.OFFICIAL_SOURCE_CANDIDATES[0]:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=Message(), fp=None)

    source_url = download_dataset.select_official_source(
        opener=None, probe_func=fake_probe
    )

    assert source_url == download_dataset.OFFICIAL_SOURCE_CANDIDATES[1]
    assert attempted == list(download_dataset.OFFICIAL_SOURCE_CANDIDATES)


def test_select_official_source_reports_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_probe(url: str, opener=None) -> None:
        raise urllib.error.URLError("Temporary failure in name resolution")

    monkeypatch.setattr(
        download_dataset,
        "collect_runtime_network_diagnostics",
        lambda runner=subprocess.run: [
            "Diagnostic local automatique:",
            "- ping -c 1 1.1.1.1 -> code retour 2",
            "  stderr: ping: connect: Network is unreachable",
            "- getent hosts physionet.org -> code retour 2",
            "  sortie: aucune réponse",
            "Cause probable: la machine n'a plus d'accès réseau sortant.",
            "Action: rétablissez la connexion réseau puis relancez make download_dataset.",
        ],
    )

    with pytest.raises(download_dataset.HandledDownloadError) as error_info:
        download_dataset.select_official_source(opener=None, probe_func=fake_probe)

    assert "Connexion internet indisponible ou instable" in error_info.value.lines[0]
    assert "Network is unreachable" in "\n".join(error_info.value.lines)


def test_main_returns_zero_when_dataset_is_already_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    subject_dir = data_dir / "S001"
    subject_dir.mkdir(parents=True)
    (subject_dir / "S001R01.edf").write_bytes(b"edf")
    (subject_dir / "S001R01.edf.event").write_bytes(b"event")
    args = SimpleNamespace(destination=str(data_dir), subject_count=1, run_count=1)

    monkeypatch.setattr(download_dataset, "parse_args", lambda: args)

    exit_code = download_dataset.main()

    assert exit_code == 0
    assert (data_dir / ".eegmmidb.ok").exists()
    assert "déjà complet" in capsys.readouterr().out


def test_main_downloads_dataset_via_selected_official_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    args = SimpleNamespace(destination=str(data_dir), subject_count=1, run_count=1)

    def fake_run_wget(source_url: str, destination_dir: Path) -> tuple[int, str]:
        subject_dir = destination_dir / "S001"
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "S001R01.edf").write_bytes(b"edf")
        (subject_dir / "S001R01.edf.event").write_bytes(b"event")
        return 0, ""

    monkeypatch.setattr(download_dataset, "parse_args", lambda: args)
    monkeypatch.setattr(download_dataset.shutil, "which", lambda _: "/usr/bin/wget")
    monkeypatch.setattr(
        download_dataset,
        "select_official_source",
        lambda: download_dataset.OFFICIAL_SOURCE_CANDIDATES[0],
    )
    monkeypatch.setattr(download_dataset, "run_wget_download", fake_run_wget)

    exit_code = download_dataset.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Source: {download_dataset.OFFICIAL_SOURCE_CANDIDATES[0]}" in captured.out
    assert "complet et validé" in captured.out
    assert (data_dir / ".eegmmidb.ok").exists()


def test_main_returns_zero_with_actionable_message_on_handled_source_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    args = SimpleNamespace(destination=str(data_dir), subject_count=1, run_count=1)

    def fail_select_official_source() -> str:
        raise download_dataset.HandledDownloadError(
            [
                "❌ Sources officielles EEGMMIDB indisponibles sur PhysioNet.",
                "Action 1: relancez make download_dataset plus tard.",
            ]
        )

    monkeypatch.setattr(download_dataset, "parse_args", lambda: args)
    monkeypatch.setattr(download_dataset.shutil, "which", lambda _: "/usr/bin/wget")
    monkeypatch.setattr(
        download_dataset,
        "select_official_source",
        fail_select_official_source,
    )

    exit_code = download_dataset.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Sources officielles EEGMMIDB indisponibles" in captured.err


def test_classify_wget_error_includes_runtime_network_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        download_dataset,
        "collect_runtime_network_diagnostics",
        lambda runner=subprocess.run: [
            "Diagnostic local automatique:",
            "- ping -c 1 1.1.1.1 -> code retour 2",
            "  stderr: ping: connect: Network is unreachable",
            "Cause probable: la machine n'a plus d'accès réseau sortant.",
            "Action: rétablissez la connexion réseau puis relancez make download_dataset.",
        ],
    )

    lines = download_dataset.classify_wget_error(
        4, "Temporary failure in name resolution"
    )

    assert lines is not None
    assert "Diagnostic wget: Temporary failure in name resolution" in lines
    assert "Diagnostic local automatique:" in lines
    assert "Network is unreachable" in "\n".join(lines)

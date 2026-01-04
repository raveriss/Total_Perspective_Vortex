"""Tests ciblés sur scripts.predict._load_data pour sécuriser la reconstruction."""

import sys
from pathlib import Path

import numpy as np

from scripts import predict


# Verrouille l'initialisation booléenne de needs_rebuild (mutant équivalent sinon)
def test_predict_load_data_initializes_needs_rebuild_as_false(
    tmp_path, monkeypatch
) -> None:
    """Verrouille needs_rebuild: bool False dès l'entrée dans l'implémentation."""

    subject = "S011"
    run = "R03"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)

    # Prépare des fichiers valides pour forcer le chemin "pas de rebuild".
    expected_X = np.arange(12).reshape(3, 2, 2)
    expected_y = np.array([0, 1, 0])
    np.save(subject_dir / f"{run}_X.npy", expected_X)
    np.save(subject_dir / f"{run}_y.npy", expected_y)

    # Verrouille l'absence de rebuild dans ce scénario.
    def _forbid_rebuild(*args, **kwargs):
        raise AssertionError("_build_npy_from_edf ne doit pas être appelé ici")

    monkeypatch.setattr(predict, "_build_npy_from_edf", _forbid_rebuild)

    captured: dict[str, object] = {}

    # Capture la première valeur observée de needs_rebuild dans une frame load_data.
    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        if not filename.endswith("scripts/predict.py"):
            return tracer
        if "load_data" not in name:
            return tracer
        if "needs_rebuild" not in frame.f_locals:
            return tracer
        if "value" not in captured:
            captured["value"] = frame.f_locals["needs_rebuild"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        predict._load_data(subject, run, data_dir, raw_dir)
    finally:
        sys.settrace(previous_tracer)

    assert "value" in captured
    assert captured["value"] is False
    assert type(captured["value"]) is bool


# Vérifie que _build_npy_from_edf est invoqué dès que les .npy sont invalides
def test_predict_load_data_rebuilds_invalid_numpy_payloads(tmp_path, monkeypatch):
    """Force la reconstruction sur X 2D et sur un y désaligné."""

    # Prépare le sujet/run pour isoler les fichiers temporaires
    subject = "S010"
    run = "R02"
    # Construit les répertoires attendus par la signature
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    # Crée l'arborescence pour simuler des .npy déjà présents
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    # Prépare les tenseurs régénérés par la reconstruction simulée
    rebuilt_X = np.full((2, 3, 4), fill_value=5)
    rebuilt_y = np.array([0, 1])
    # Trace les appels de reconstruction pour vérifier la propagation des arguments
    calls: list[tuple[str, str, Path, Path]] = []

    # Stub de reconstruction qui remplace l'EDF pendant le test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        calls.append((subject_arg, run_arg, data_arg, raw_arg))
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        np.save(features_path, rebuilt_X)
        np.save(labels_path, rebuilt_y)
        return features_path, labels_path

    monkeypatch.setattr(predict, "_build_npy_from_edf", fake_build_npy)

    # 1) X 2D pour déclencher la reconstruction liée à la mauvaise dimension
    np.save(subject_dir / f"{run}_X.npy", np.ones((2, 4)))
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, data_dir, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

    # 2) y de longueur différente pour forcer la régénération
    calls.clear()
    np.save(subject_dir / f"{run}_X.npy", np.ones((3, 2, 2)))
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1, 1, 0]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, data_dir, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)


# Vérifie que la reconstruction n'est pas déclenchée si les .npy sont valides
def test_predict_load_data_skips_rebuild_for_valid_files(tmp_path, monkeypatch):
    """Couvre le chemin nominal lorsque X et y sont déjà conformes."""

    subject = "S011"
    run = "R03"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    # Génère des tensors valides pour éviter toute reconstruction
    expected_X = np.arange(12).reshape(3, 2, 2)
    expected_y = np.array([0, 1, 0])
    np.save(subject_dir / f"{run}_X.npy", expected_X)
    np.save(subject_dir / f"{run}_y.npy", expected_y)

    # Espionne np.load pour imposer mmap_mode="r" sur les chargements "candidate"
    real_load = predict.np.load
    load_calls: list[tuple[Path, str]] = []

    def spy_load(path, *args, **kwargs):
        mode = kwargs.get("mmap_mode", "__missing__")
        load_calls.append((Path(path), str(mode)))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(predict.np, "load", spy_load)

    # Stub qui provoquerait un échec si _build_npy_from_edf était appelé
    def fail_build_npy(*args, **kwargs):
        raise AssertionError("_build_npy_from_edf ne doit pas être invoqué")

    monkeypatch.setattr(predict, "_build_npy_from_edf", fail_build_npy)

    X, y = predict._load_data(subject, run, data_dir, raw_dir)

    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"
    features_modes = [mode for path, mode in load_calls if path == features_path]
    labels_modes = [mode for path, mode in load_calls if path == labels_path]
    assert sorted(features_modes) == ["__missing__", "r"]
    assert sorted(labels_modes) == ["__missing__", "r"]


# Vérifie que la reconstruction est déclenchée si un seul des deux fichiers manque
def test_predict_load_data_rebuilds_when_one_numpy_file_missing(tmp_path, monkeypatch):
    """Verrouille le OR de présence (un seul manquant doit reconstruire)."""

    subject = "S012"
    run = "R04"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)

    rebuilt_X = np.zeros((2, 3, 4))
    rebuilt_y = np.array([0, 1])
    calls: list[tuple[str, str, Path]] = []

    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        calls.append((subject_arg, run_arg, raw_arg))
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        np.save(features_path, rebuilt_X)
        np.save(labels_path, rebuilt_y)
        return features_path, labels_path

    monkeypatch.setattr(predict, "_build_npy_from_edf", fake_build_npy)

    # Cas 1: X présent, y absent => rebuild obligatoire
    np.save(subject_dir / f"{run}_X.npy", np.ones((2, 3, 4)))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

    # Cas 2: y présent, X absent => rebuild obligatoire
    calls.clear()
    (subject_dir / f"{run}_X.npy").unlink()
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

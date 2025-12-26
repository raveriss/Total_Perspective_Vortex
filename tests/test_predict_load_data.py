"""Tests ciblés sur scripts.predict._load_data pour sécuriser la reconstruction."""

import numpy as np

from scripts import predict


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
    calls: list[tuple[str, str]] = []

    # Stub de reconstruction qui remplace l'EDF pendant le test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        calls.append((subject_arg, run_arg))
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
    assert calls == [(subject, run)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

    # 2) y de longueur différente pour forcer la régénération
    calls.clear()
    np.save(subject_dir / f"{run}_X.npy", np.ones((3, 2, 2)))
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1, 1, 0]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run)]
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

    # Stub qui provoquerait un échec si _build_npy_from_edf était appelé
    def fail_build_npy(*args, **kwargs):
        raise AssertionError("_build_npy_from_edf ne doit pas être invoqué")

    monkeypatch.setattr(predict, "_build_npy_from_edf", fail_build_npy)

    X, y = predict._load_data(subject, run, data_dir, raw_dir)

    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)

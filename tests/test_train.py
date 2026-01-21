# Importe argparse pour typer les erreurs attendues
import argparse

# Importe NumPy pour fabriquer des labels synthétiques
import numpy as np

# Importe pytest pour vérifier les exceptions attendues
import pytest

# Offre les classifieurs scikit-learn pour le contrôle de la grille
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Importe la config et les helpers de training pour les tester
from scripts.train import (
    EpochWindowContext,
    PipelineConfig,
    _adapt_pipeline_config_for_samples,
    _build_epochs_for_window,
    _build_grid_search_grid,
    _build_window_search_pipeline,
    _normalize_identifier,
    _read_epoch_window_metadata,
    _resolve_epoch_window_path,
    _score_epoch_window,
    _select_best_epoch_window,
    _write_epoch_window_metadata,
    resolve_sampling_rate,
)


# Vérifie la détection des effectifs trop faibles pour LDA
def test_adapt_pipeline_config_switches_lda_for_small_sample():
    """Vérifie que LDA est remplacé lorsque l'effectif est insuffisant."""

    # Prépare une configuration LDA standard
    config = PipelineConfig(sfreq=128.0, classifier="lda")
    # Crée un jeu de labels minimal avec deux classes et deux samples
    y = np.array([0, 1])
    # Applique l'adaptation sur effectif réduit
    adapted = _adapt_pipeline_config_for_samples(config, y)
    # Vérifie que le classifieur est basculé vers centroid
    assert adapted.classifier == "centroid"


# Vérifie que LDA est conservé lorsque l'effectif est suffisant
def test_adapt_pipeline_config_keeps_lda_for_valid_sample():
    """Vérifie que LDA reste utilisé quand l'effectif le permet."""

    # Prépare une configuration LDA standard
    config = PipelineConfig(sfreq=128.0, classifier="lda")
    # Crée un jeu de labels avec trois samples et deux classes
    y = np.array([0, 1, 0])
    # Applique l'adaptation sur effectif suffisant
    adapted = _adapt_pipeline_config_for_samples(config, y)
    # Vérifie que le classifieur reste LDA
    assert adapted.classifier == "lda"


# Vérifie que la grille peut exclure LDA si nécessaire
def test_build_grid_search_grid_respects_lda_flag():
    """Vérifie que la grille inclut LDA seulement si autorisé."""

    # Prépare une configuration de pipeline de référence
    config = PipelineConfig(sfreq=128.0, classifier="lda")
    # Construit une grille sans LDA autorisé
    grid_without_lda = _build_grid_search_grid(config, allow_lda=False)
    # Extrait les classes présentes dans la grille de classifieurs
    types_without_lda = {type(item) for item in grid_without_lda["classifier"]}
    # Vérifie l'absence du classifieur LDA dans la grille
    assert LinearDiscriminantAnalysis not in types_without_lda
    # Construit une grille avec LDA autorisé
    grid_with_lda = _build_grid_search_grid(config, allow_lda=True)
    # Extrait les classes présentes dans la grille de classifieurs
    types_with_lda = {type(item) for item in grid_with_lda["classifier"]}
    # Vérifie que LDA est bien présent dans la grille
    assert LinearDiscriminantAnalysis in types_with_lda


# Vérifie que les identifiants vides sont rejetés
def test_normalize_identifier_rejects_empty_value():
    """Vérifie le rejet des identifiants vides."""

    # Vérifie que l'erreur ArgumentTypeError est levée
    with pytest.raises(argparse.ArgumentTypeError):
        # Exécute la normalisation avec une valeur vide
        _normalize_identifier(value=" ", prefix="S", width=3, label="Sujet")


# Vérifie que les identifiants non numériques sont rejetés
def test_normalize_identifier_rejects_non_numeric():
    """Vérifie le rejet des identifiants non numériques."""

    # Vérifie que l'erreur ArgumentTypeError est levée
    with pytest.raises(argparse.ArgumentTypeError):
        # Exécute la normalisation avec une valeur non numérique
        _normalize_identifier(value="S0X", prefix="S", width=3, label="Sujet")


# Vérifie que les identifiants négatifs sont rejetés
def test_normalize_identifier_rejects_non_positive():
    """Vérifie le rejet des identifiants non positifs."""

    # Vérifie que l'erreur ArgumentTypeError est levée
    with pytest.raises(argparse.ArgumentTypeError):
        # Exécute la normalisation avec une valeur non positive
        _normalize_identifier(value="0", prefix="S", width=3, label="Sujet")


# Vérifie le fallback si l'EDF est absent
def test_resolve_sampling_rate_returns_requested_when_missing(tmp_path):
    """Vérifie la valeur par défaut si l'EDF est absent."""

    # Appelle la résolution avec un répertoire sans EDF
    result = resolve_sampling_rate("S001", "R01", tmp_path, 50.0)
    # Vérifie que la fréquence demandée est conservée
    assert result == 50.0


# Vérifie le fallback si la lecture EDF échoue
def test_resolve_sampling_rate_handles_read_error(tmp_path, monkeypatch):
    """Vérifie que l'échec de lecture EDF retourne la valeur demandée."""

    # Crée le dossier sujet pour simuler l'EDF présent
    subject_dir = tmp_path / "S001"
    # Assure la présence du dossier pour le chemin attendu
    subject_dir.mkdir()
    # Crée un fichier EDF factice pour passer la garde d'existence
    (subject_dir / "S001R01.edf").write_text("dummy")

    # Force le loader à lever une erreur de lecture
    monkeypatch.setattr(
        "scripts.train.preprocessing.load_physionet_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad edf")),
    )

    # Appelle la résolution pour couvrir le bloc d'exception
    result = resolve_sampling_rate("S001", "R01", tmp_path, 50.0)
    # Vérifie que la fréquence demandée est conservée
    assert result == 50.0


# Vérifie le fallback quand la fréquence détectée n'est pas numérique
def test_resolve_sampling_rate_falls_back_on_non_numeric_metadata(
    tmp_path, monkeypatch
):
    """Vérifie que les métadonnées invalides conservent la fréquence par défaut."""

    # Crée le dossier sujet pour simuler l'EDF présent
    subject_dir = tmp_path / "S002"
    # Assure la présence du dossier pour le chemin attendu
    subject_dir.mkdir()
    # Crée un fichier EDF factice pour passer la garde d'existence
    (subject_dir / "S002R01.edf").write_text("dummy")

    # Prépare un Raw factice pour valider l'appel à close()
    class DummyRaw:
        # Initialise un indicateur pour vérifier la fermeture
        def __init__(self):
            # Mémorise l'état de fermeture pour l'assertion finale
            self.closed = False

        # Simule la fermeture pour respecter le contrat MNE
        def close(self):
            # Met à jour l'état pour confirmer l'appel à close()
            self.closed = True

    # Instancie le Raw factice pour le retour du loader
    dummy_raw = DummyRaw()

    # Force la lecture EDF à retourner une fréquence non numérique
    monkeypatch.setattr(
        "scripts.train.preprocessing.load_physionet_raw",
        lambda *_args, **_kwargs: (dummy_raw, {"sampling_rate": {"bad": "value"}}),
    )

    # Appelle la résolution pour couvrir le fallback non numérique
    result = resolve_sampling_rate("S002", "R01", tmp_path, 50.0)
    # Vérifie que la fréquence demandée est conservée
    assert result == 50.0
    # Vérifie que le Raw est bien fermé malgré la valeur invalide
    assert dummy_raw.closed is True


# Vérifie la construction de la pipeline de scoring de fenêtre
def test_build_window_search_pipeline_builds_expected_config(monkeypatch):
    """Vérifie la configuration utilisée pour la pipeline de scoring."""

    # Prépare un conteneur pour capturer la configuration fournie
    captured = {}

    # Remplace build_pipeline pour inspecter la configuration
    def fake_build_pipeline(config):
        # Conserve la configuration reçue pour les assertions
        captured["config"] = config
        # Retourne un objet sentinelle pour éviter un fit réel
        return "pipeline"

    # Injecte le double dans le module ciblé
    monkeypatch.setattr("scripts.train.build_pipeline", fake_build_pipeline)

    # Construit la pipeline de scoring pour une fréquence donnée
    result = _build_window_search_pipeline(100.0)
    # Vérifie que le pipeline retourné est bien celui du double
    assert result == "pipeline"
    # Vérifie que la config capturée utilise le classifieur LDA
    assert captured["config"].classifier == "lda"


# Vérifie que le score de fenêtre retourne None sans échantillons
def test_score_epoch_window_returns_none_on_empty_data():
    """Vérifie le fallback sur un jeu de données vide."""

    # Prépare un tableau vide d'epochs
    X = np.zeros((0, 2, 2))
    # Prépare un tableau vide de labels
    y = np.array([])
    # Vérifie que le score est None quand aucun échantillon n'existe
    assert _score_epoch_window(X, y, 50.0) is None


# Vérifie que le score de fenêtre retourne None si la CV est impossible
def test_score_epoch_window_returns_none_on_insufficient_counts():
    """Vérifie le fallback lorsque la CV est impossible."""

    # Prépare des epochs minimaux
    X = np.zeros((3, 2, 2))
    # Prépare des labels avec effectif minimal insuffisant
    y = np.array([0, 0, 1])
    # Vérifie que le score est None si la CV est impossible
    assert _score_epoch_window(X, y, 50.0) is None


# Vérifie la construction des epochs et des labels pour une fenêtre
def test_build_epochs_for_window_maps_labels(monkeypatch):
    """Contrôle la conversion des labels en entiers pour une fenêtre."""

    # Construit des données d'epochs fictives
    epochs_data = np.zeros((2, 3, 4))

    # Définit une classe d'epochs factice pour le test
    class DummyEpochs:
        # Stocke les données d'epochs simulées
        def __init__(self, data):
            self._data = data

        # Retourne les données d'epochs simulées
        def get_data(self, copy=True):
            return self._data

    # Instancie les epochs factices
    dummy_epochs = DummyEpochs(epochs_data)

    # Force la création d'epochs à retourner notre dummy
    monkeypatch.setattr(
        "scripts.train.preprocessing.create_epochs_from_raw",
        lambda *_args, **_kwargs: dummy_epochs,
    )
    # Force le QC à renvoyer les labels inversés
    monkeypatch.setattr(
        "scripts.train.preprocessing.summarize_epoch_quality",
        lambda *_args, **_kwargs: (dummy_epochs, {}, ["B", "A"]),
    )

    # Définit un Raw factice avec la fréquence requise
    class DummyRaw:
        info = {"sfreq": 100.0}

    # Construit le contexte minimal pour la fenêtre
    context = EpochWindowContext(
        filtered_raw=DummyRaw(),
        events=np.array([]),
        event_id={},
        motor_labels=["B", "A"],
        subject="S001",
        run="R01",
    )
    # Exécute la construction des epochs pour la fenêtre
    built_epochs, labels = _build_epochs_for_window(context, (0.5, 2.5))
    # Vérifie que les données sont conservées
    assert built_epochs.shape == epochs_data.shape
    # Vérifie que le mapping des labels est appliqué
    assert labels.tolist() == [1, 0]


# Vérifie le fallback QC quand une classe est supprimée
def test_build_epochs_for_window_falls_back_on_missing_labels(monkeypatch):
    """Vérifie le fallback lorsque le QC supprime une classe."""

    # Construit des données d'epochs fictives
    epochs_data = np.zeros((2, 2, 2))

    # Définit une classe d'epochs factice pour le test
    class DummyEpochs:
        # Stocke les données d'epochs simulées
        def __init__(self, data):
            self._data = data

        # Retourne les données d'epochs simulées
        def get_data(self, copy=True):
            return self._data

    # Instancie les epochs factices
    dummy_epochs = DummyEpochs(epochs_data)

    # Force la création d'epochs à retourner notre dummy
    monkeypatch.setattr(
        "scripts.train.preprocessing.create_epochs_from_raw",
        lambda *_args, **_kwargs: dummy_epochs,
    )
    # Force le QC à lever une erreur de labels manquants
    monkeypatch.setattr(
        "scripts.train.preprocessing.summarize_epoch_quality",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Missing labels")),
    )

    # Définit un Raw factice avec la fréquence requise
    class DummyRaw:
        info = {"sfreq": 100.0}

    # Construit le contexte minimal pour la fenêtre
    context = EpochWindowContext(
        filtered_raw=DummyRaw(),
        events=np.array([]),
        event_id={},
        motor_labels=["A", "B"],
        subject="S001",
        run="R01",
    )
    # Exécute la construction des epochs pour la fenêtre
    built_epochs, labels = _build_epochs_for_window(context, (0.5, 2.5))
    # Vérifie que les données sont conservées
    assert built_epochs.shape == epochs_data.shape
    # Vérifie que le fallback conserve les labels moteurs
    assert labels.tolist() == [0, 1]


# Vérifie le fallback quand le rejet rend l'effectif trop faible
def test_build_epochs_for_window_relaxes_rejection_on_low_counts(monkeypatch):
    """Vérifie que les labels initiaux sont conservés si l'effectif baisse."""

    # Construit des données d'epochs fictives avec plusieurs essais
    epochs_data = np.zeros((4, 2, 2))

    # Définit une classe d'epochs factice pour le test
    class DummyEpochs:
        # Stocke les données d'epochs simulées
        def __init__(self, data):
            self._data = data

        # Retourne les données d'epochs simulées
        def get_data(self, copy=True):
            return self._data

    # Instancie les epochs factices
    dummy_epochs = DummyEpochs(epochs_data)

    # Force la création d'epochs à retourner notre dummy
    monkeypatch.setattr(
        "scripts.train.preprocessing.create_epochs_from_raw",
        lambda *_args, **_kwargs: dummy_epochs,
    )
    # Force le QC à renvoyer une seule classe pour déclencher le fallback
    monkeypatch.setattr(
        "scripts.train.preprocessing.summarize_epoch_quality",
        lambda *_args, **_kwargs: (dummy_epochs, {}, ["A", "A"]),
    )

    # Définit un Raw factice avec la fréquence requise
    class DummyRaw:
        info = {"sfreq": 100.0}

    # Construit le contexte minimal pour la fenêtre
    context = EpochWindowContext(
        filtered_raw=DummyRaw(),
        events=np.array([]),
        event_id={},
        motor_labels=["A", "B", "A", "B"],
        subject="S001",
        run="R01",
    )
    # Exécute la construction des epochs pour la fenêtre
    built_epochs, labels = _build_epochs_for_window(context, (0.5, 2.5))
    # Vérifie que les données sont conservées
    assert built_epochs.shape == epochs_data.shape
    # Vérifie que les labels initiaux sont conservés
    assert labels.tolist() == [0, 1, 0, 1]


# Vérifie la sélection de la meilleure fenêtre via le score
def test_select_best_epoch_window_picks_best_score(monkeypatch):
    """Vérifie la sélection de fenêtre par score cross-val."""

    # Prépare des scores pour chaque fenêtre dans l'ordre
    scores = iter([0.4, 0.8, 0.6])

    # Stub l'epoching pour retourner des matrices constantes
    monkeypatch.setattr(
        "scripts.train._build_epochs_for_window",
        lambda *_args, **_kwargs: (np.zeros((2, 2, 2)), np.array([0, 1])),
    )
    # Stub le scoring pour retourner des scores déterministes
    monkeypatch.setattr(
        "scripts.train._score_epoch_window",
        lambda *_args, **_kwargs: next(scores),
    )

    # Définit un Raw factice avec la fréquence requise
    class DummyRaw:
        info = {"sfreq": 100.0}

    # Construit le contexte minimal pour la sélection
    context = EpochWindowContext(
        filtered_raw=DummyRaw(),
        events=np.array([]),
        event_id={},
        motor_labels=[],
        subject="S001",
        run="R01",
    )
    # Exécute la sélection de fenêtre
    window, epochs, labels = _select_best_epoch_window(context)
    # Vérifie que la fenêtre choisie correspond au meilleur score
    assert window == (1.0, 3.0)
    # Vérifie que les epochs sont retournées
    assert epochs.shape == (2, 2, 2)
    # Vérifie que les labels sont retournés
    assert labels.tolist() == [0, 1]


# Vérifie la lecture/écriture de la fenêtre d'epochs
def test_epoch_window_metadata_roundtrip(tmp_path):
    """Vérifie que la fenêtre est persistée puis relue."""

    # Définit un sujet et un run factices
    subject = "S123"
    run = "R03"
    # Définit une fenêtre factice
    window = (0.5, 2.5)
    # Écrit la fenêtre dans le dossier temporaire
    _write_epoch_window_metadata(subject, run, tmp_path, window)
    # Vérifie que le fichier attendu existe
    assert _resolve_epoch_window_path(subject, run, tmp_path).exists()
    # Relit la fenêtre persistée
    loaded_window = _read_epoch_window_metadata(subject, run, tmp_path)
    # Vérifie que la fenêtre lue correspond à la fenêtre écrite
    assert loaded_window == window


# Vérifie le comportement quand aucun fichier de fenêtre n'existe
def test_epoch_window_metadata_returns_none_when_missing(tmp_path):
    """Vérifie que l'absence de fichier retourne None."""

    # Définit un sujet et un run factices
    subject = "S999"
    run = "R01"
    # Vérifie que la lecture retourne None sans fichier
    assert _read_epoch_window_metadata(subject, run, tmp_path) is None


# Vérifie le comportement du scoring quand il n'y a aucun échantillon
def test_score_epoch_window_returns_none_for_empty_data() -> None:
    """Vérifie que le scoring retourne None pour un dataset vide."""

    # Définit des données et labels vides
    X = np.zeros((0,))
    y = np.array([], dtype=int)
    # Vérifie que le scoring retourne None sur entrée vide
    assert _score_epoch_window(X, y, 100.0) is None


# Vérifie le comportement du scoring quand la CV est impossible
def test_score_epoch_window_returns_none_for_too_few_samples() -> None:
    """Vérifie que le scoring retourne None si la CV est impossible."""

    # Définit des données factices avec deux échantillons
    X = np.zeros((2, 2, 2))
    # Définit des labels avec un échantillon par classe
    y = np.array([0, 1])
    # Vérifie que la CV est ignorée faute de splits suffisants
    assert _score_epoch_window(X, y, 100.0) is None

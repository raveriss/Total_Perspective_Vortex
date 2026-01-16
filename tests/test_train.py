# Importe NumPy pour fabriquer des labels synthétiques
import numpy as np

# Offre les classifieurs scikit-learn pour le contrôle de la grille
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Importe la config et les helpers de training pour les tester
from scripts.train import (
    EpochWindowContext,
    PipelineConfig,
    _adapt_pipeline_config_for_samples,
    _build_epochs_for_window,
    _build_grid_search_grid,
    _read_epoch_window_metadata,
    _resolve_epoch_window_path,
    _score_epoch_window,
    _select_best_epoch_window,
    _write_epoch_window_metadata,
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

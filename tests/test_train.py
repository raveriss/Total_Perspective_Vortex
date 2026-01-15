# Importe NumPy pour fabriquer des labels synthétiques
import numpy as np

# Offre les classifieurs scikit-learn pour le contrôle de la grille
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Importe la config et les helpers de training pour les tester
from scripts.train import (
    PipelineConfig,
    _adapt_pipeline_config_for_samples,
    _build_grid_search_grid,
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

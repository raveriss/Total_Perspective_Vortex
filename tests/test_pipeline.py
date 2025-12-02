"""Tests du pipeline TPV pour la sérialisation et la validation statistique."""

# Garantit l'accès à numpy pour générer des données synthétiques
import numpy as np

# Vérifie la compatibilité scikit-learn pour les scores de CV
from sklearn.model_selection import cross_val_score

# Utilise pytest pour les assertions et la gestion des fixtures temporaires
import pytest

# Importe la construction et la persistance du pipeline TPV
from tpv.pipeline import build_pipeline, load_pipeline, save_pipeline


# Vérifie que le pipeline peut être sauvegardé puis rechargé sans perte
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_pickling_roundtrip(tmp_path):
    """Teste que pickle conserve le comportement prédictif du pipeline."""

    # Génère des données EEG synthétiques avec deux canaux et cinquante points
    X = np.random.randn(12, 2, 50)
    # Génère des labels binaires indépendants des données
    y = np.random.randint(0, 2, size=12)
    # Construit un pipeline complet avec scaler et réducteur PCA
    pipeline = build_pipeline(
        preprocessors=[],
        sfreq=100.0,
        scaler="standard",
        classifier="logistic",
        dim_method="pca",
    )
    # Entraîne le pipeline sur les données synthétiques
    pipeline.fit(X, y)
    # Détermine le chemin de sauvegarde dans le répertoire temporaire
    model_path = tmp_path / "pipeline.joblib"
    # Sauvegarde le pipeline entraîné sur disque
    save_pipeline(pipeline, str(model_path))
    # Recharge le pipeline depuis le fichier sérialisé
    restored = load_pipeline(str(model_path))
    # Compare les prédictions entre l'original et la version rechargée
    np.testing.assert_allclose(pipeline.predict(X), restored.predict(X))


# Vérifie que le pipeline s'intègre à cross_val_score sans erreur
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_cross_val_score_runs():
    """Teste la compatibilité du pipeline avec cross_val_score."""

    # Crée un dataset synthétique suffisamment grand pour la validation croisée
    X = np.random.randn(30, 2, 80)
    # Génère des labels aléatoires pour simuler deux classes
    y = np.random.randint(0, 2, size=30)
    # Construit un pipeline sans scaler pour tester la configuration minimale
    pipeline = build_pipeline(
        preprocessors=[],
        sfreq=100.0,
        classifier="lda",
        dim_method="pca",
    )
    # Exécute une validation croisée à trois plis pour vérifier l'intégration
    scores = cross_val_score(pipeline, X, y, cv=3)
    # Valide que trois scores sont produits sans lever d'exception
    assert len(scores) == 3


# Vérifie l'absence de fuite de labels sur des données sans signal
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_no_label_leakage():
    """Vérifie que le pipeline ne dépasse pas la performance au hasard."""

    # Crée un dataset plus large pour stabiliser l'estimation des scores
    X = np.random.randn(60, 2, 80)
    # Génère des labels aléatoires pour simuler un problème sans structure
    y = np.random.randint(0, 2, size=60)
    # Construit un pipeline avec scaler robuste pour varier la configuration
    pipeline = build_pipeline(
        preprocessors=[],
        sfreq=120.0,
        scaler="robust",
        classifier="svm",
        dim_method="pca",
    )
    # Évalue la performance moyenne sur quatre plis de validation croisée
    scores = cross_val_score(pipeline, X, y, cv=4)
    # Vérifie que la moyenne reste proche du hasard pour exclure une fuite
    assert np.mean(scores) < 0.7

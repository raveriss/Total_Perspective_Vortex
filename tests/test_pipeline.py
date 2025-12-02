"""Tests du pipeline TPV pour la sérialisation et la validation statistique."""

# Garantit l'accès à numpy pour générer des données synthétiques
import numpy as np

# Utilise pytest pour les assertions et la gestion des fixtures temporaires
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Vérifie la compatibilité scikit-learn pour les scores de CV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

# Importe la construction et la persistance du pipeline TPV
from tpv.pipeline import (
    PipelineConfig,
    _build_classifier,
    _build_scaler,
    build_pipeline,
    load_pipeline,
    save_pipeline,
)

# Fixe le nombre de plis pour harmoniser les tests de validation croisée
CROSS_VALIDATION_SPLITS = 3

# Fixe le nombre de plis dédié au test de fuite de labels
LEAKAGE_SPLITS = 4

# Fixe la borne d'exactitude pour détecter des fuites de labels
LEAKAGE_THRESHOLD = 0.7

# Fixe le nombre maximal d'itérations utilisé par la régression logistique
LOGISTIC_MAX_ITER = 1000

# Fixe le nombre de composantes CSP utilisé dans les tests de pipeline
CSP_COMPONENTS = 5


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
        config=PipelineConfig(
            sfreq=100.0,
            scaler="standard",
            classifier="logistic",
            dim_method="pca",
        ),
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
        config=PipelineConfig(sfreq=100.0, classifier="lda", dim_method="pca"),
    )
    # Exécute une validation croisée à trois plis pour vérifier l'intégration
    scores = cross_val_score(pipeline, X, y, cv=CROSS_VALIDATION_SPLITS)
    # Valide que trois scores sont produits sans lever d'exception
    assert len(scores) == CROSS_VALIDATION_SPLITS


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
        config=PipelineConfig(
            sfreq=120.0,
            scaler="robust",
            classifier="svm",
            dim_method="pca",
        ),
    )
    # Évalue la performance moyenne sur quatre plis de validation croisée
    scores = cross_val_score(pipeline, X, y, cv=LEAKAGE_SPLITS)
    # Vérifie que la moyenne reste proche du hasard pour exclure une fuite
    assert np.mean(scores) < LEAKAGE_THRESHOLD


def test_build_scaler_rejects_invalid_value():
    """Garantit une erreur explicite lorsque le scaler est inconnu."""

    with pytest.raises(
        ValueError, match="scaler must be 'standard', 'robust', or None"
    ):
        build_pipeline(
            preprocessors=[],
            config=PipelineConfig(
                sfreq=100.0,
                scaler="invalid",
                classifier="lda",
                dim_method="pca",
            ),
        )


def test_build_classifier_rejects_invalid_value():
    """Garantit une erreur explicite lorsque le classifieur est inconnu."""

    with pytest.raises(
        ValueError, match="classifier must be 'lda', 'logistic', or 'svm'"
    ):
        build_pipeline(
            preprocessors=[],
            config=PipelineConfig(
                sfreq=100.0,
                scaler=None,
                classifier="invalid",
                dim_method="pca",
            ),
        )


def test_build_scaler_supports_valid_variants():
    assert isinstance(_build_scaler("standard"), StandardScaler)
    assert isinstance(_build_scaler("robust"), RobustScaler)
    assert _build_scaler(None) is None


def test_build_scaler_invalid_message_is_explicit():
    with pytest.raises(
        ValueError, match="^scaler must be 'standard', 'robust', or None$"
    ):
        _build_scaler("unknown")


def test_build_scaler_accepts_uppercase_values():
    assert isinstance(_build_scaler("STANDARD"), StandardScaler)
    assert isinstance(_build_scaler("ROBUST"), RobustScaler)


def test_build_classifier_supports_valid_variants():
    assert isinstance(_build_classifier("lda"), LinearDiscriminantAnalysis)
    assert isinstance(_build_classifier("logistic"), LogisticRegression)
    assert isinstance(_build_classifier("svm"), LinearSVC)


def test_build_classifier_sets_expected_parameters():
    classifier = _build_classifier("logistic")

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == LOGISTIC_MAX_ITER


def test_build_classifier_accepts_uppercase_values():
    assert isinstance(_build_classifier("LDA"), LinearDiscriminantAnalysis)
    assert isinstance(_build_classifier("LOGISTIC"), LogisticRegression)
    assert isinstance(_build_classifier("SVM"), LinearSVC)


def test_build_classifier_invalid_value_message():
    with pytest.raises(
        ValueError, match="^classifier must be 'lda', 'logistic', or 'svm'$"
    ):
        _build_classifier("unknown")


def test_build_pipeline_includes_preprocessors_and_scaler():
    preprocessors = [("dummy", object())]

    pipeline = build_pipeline(
        preprocessors=preprocessors,
        config=PipelineConfig(
            sfreq=200.0,
            scaler="standard",
            classifier="lda",
            dim_method="pca",
        ),
    )

    assert list(pipeline.named_steps) == [
        "dummy",
        "features",
        "scaler",
        "dimensionality",
        "classifier",
    ]


def test_build_pipeline_omits_scaler_when_none():
    pipeline = build_pipeline(
        preprocessors=[],
        config=PipelineConfig(
            sfreq=128.0, scaler=None, classifier="lda", dim_method="csp"
        ),
    )

    assert "scaler" not in pipeline.named_steps
    assert pipeline.named_steps["features"].normalize is True


def test_build_pipeline_applies_normalization_flag():
    pipeline = build_pipeline(
        preprocessors=[],
        config=PipelineConfig(
            sfreq=256.0,
            scaler=None,
            classifier="svm",
            dim_method="pca",
            normalize_features=False,
        ),
    )

    assert pipeline.named_steps["features"].normalize is False


def test_build_pipeline_uses_configured_strategies():
    pipeline = build_pipeline(
        preprocessors=[],
        config=PipelineConfig(
            sfreq=256.0,
            scaler="robust",
            classifier="logistic",
            dim_method="csp",
            n_components=CSP_COMPONENTS,
            feature_strategy="wavelet",
        ),
    )

    assert pipeline.named_steps["features"].feature_strategy == "wavelet"
    assert pipeline.named_steps["dimensionality"].method == "csp"
    assert pipeline.named_steps["dimensionality"].n_components == CSP_COMPONENTS

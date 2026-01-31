# Importe NumPy pour générer des échantillons synthétiques
import numpy as np

# Importe pytest pour vérifier les erreurs attendues
import pytest

# Offre les classifieurs pour valider le mapping des options
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Offre cross_val_score pour vérifier l'intégration scikit-learn
from sklearn.model_selection import cross_val_score

# Offre les scalers pour valider la configuration du pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

# Récupère les API du pipeline à tester
from tpv.pipeline import (
    LOGISTIC_MAX_ITER,
    PipelineConfig,
    _build_classifier,
    _build_scaler,
    build_pipeline,
    load_pipeline,
    save_pipeline,
)

# Fixe le nombre de plis pour harmoniser les validations croisées
CROSS_VALIDATION_SPLITS = 3

# Fixe le nombre de plis dédié au test de fuite de labels
LEAKAGE_SPLITS = 4

# Fixe le seuil maximal toléré pour exclure une fuite de labels
LEAKAGE_THRESHOLD = 0.7

# Fixe le nombre de composantes CSP utilisé dans les tests de pipeline
CSP_COMPONENTS = 5


# Ignore les avertissements libsvm générés par scikit-learn
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_pickling_roundtrip(tmp_path):
    """Teste que pickle conserve le comportement prédictif."""

    # Génère des données EEG synthétiques pour l'entraînement
    X = np.random.randn(12, 2, 50)
    # Génère des labels équilibrés pour garantir deux classes
    y = np.array([0, 1] * 6)
    # Construit un pipeline complet avec scaler et PCA
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=100.0, scaler="standard", classifier="logistic", dim_method="pca"
        )
    )
    # Ajuste le pipeline sur les données synthétiques
    pipeline.fit(X, y)
    # Prépare le chemin du modèle sérialisé dans le répertoire temporaire
    model_path = tmp_path / "pipeline.joblib"
    # Sauvegarde le pipeline entraîné sur disque
    save_pipeline(pipeline, str(model_path))
    # Recharge le pipeline depuis le fichier joblib
    restored = load_pipeline(str(model_path))
    # Vérifie que les prédictions sont stables après rechargement
    np.testing.assert_allclose(pipeline.predict(X), restored.predict(X))


# Ignore les avertissements libsvm générés par scikit-learn
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_cross_val_score_runs():
    """Teste la compatibilité du pipeline avec cross_val_score."""

    # Crée un dataset synthétique pour trois plis de validation
    X = np.random.randn(30, 2, 80)
    # Crée des labels aléatoires pour simuler deux classes
    y = np.random.randint(0, 2, size=30)
    # Construit un pipeline minimal sans scaler
    pipeline = build_pipeline(
        PipelineConfig(sfreq=100.0, classifier="lda", dim_method="pca")
    )
    # Exécute la validation croisée et récupère les scores
    scores = cross_val_score(pipeline, X, y, cv=CROSS_VALIDATION_SPLITS)
    # Vérifie que chaque pli produit un score
    assert len(scores) == CROSS_VALIDATION_SPLITS


# Ignore les avertissements libsvm générés par scikit-learn
@pytest.mark.filterwarnings("ignore:.*libsvm.*")
def test_pipeline_no_label_leakage():
    """Vérifie que la performance reste proche du hasard."""

    # Fixe un générateur déterministe pour stabiliser le test
    rng = np.random.default_rng(0)
    # Crée un dataset plus large pour stabiliser les scores
    X = rng.standard_normal((60, 2, 80))
    # Crée des labels aléatoires pour simuler deux classes
    y = rng.integers(0, 2, size=60)
    # Construit un pipeline avec scaler robuste
    pipeline = build_pipeline(
        PipelineConfig(sfreq=120.0, scaler="robust", classifier="svm", dim_method="pca")
    )
    # Calcule les scores moyens sur quatre plis
    scores = cross_val_score(pipeline, X, y, cv=LEAKAGE_SPLITS)
    # Ajoute une marge pour éviter un faux positif dû à l'arrondi flottant
    leakage_margin = 1e-12
    # Vérifie que la moyenne reste sous le seuil de fuite avec tolérance
    assert np.mean(scores) <= LEAKAGE_THRESHOLD + leakage_margin


# Vérifie que le pipeline produit des formes cohérentes
def test_pipeline_respects_input_and_output_shapes():
    """Contrôle la forme des features et des prédictions."""

    # Crée un petit lot de données EEG synthétiques
    X = np.random.randn(10, 3, 40)
    # Crée des labels aléatoires pour l'entraînement
    y = np.random.randint(0, 2, size=10)
    # Construit un pipeline PCA avec trois composantes
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=64.0,
            scaler=None,
            classifier="lda",
            dim_method="pca",
            n_components=3,
        )
    )
    # Ajuste le pipeline pour préparer les étapes internes
    pipeline.fit(X, y)
    # Récupère les features extraites pour vérifier leur dimension
    features = pipeline.named_steps["features"].transform(X)
    # Calcule le nombre attendu de colonnes tabulaires
    expected_features = X.shape[1] * len(pipeline.named_steps["features"].band_labels)
    # Vérifie que la matrice tabulaire respecte la forme attendue
    assert features.shape == (X.shape[0], expected_features)
    # Applique la réduction de dimension pour vérifier la projection
    reduced = pipeline.named_steps["dimensionality"].transform(features)
    # Vérifie que la projection réduit bien à trois composantes
    assert reduced.shape == (X.shape[0], 3)
    # Vérifie que les prédictions respectent le nombre d'échantillons
    predictions = pipeline.predict(X)
    assert predictions.shape == (X.shape[0],)


# Vérifie qu'une valeur invalide de scaler lève une erreur explicite
def test_build_scaler_rejects_invalid_value():
    """Garantit une erreur claire pour un scaler inconnu."""

    # Construit une configuration invalide pour le scaler
    config = PipelineConfig(
        sfreq=100.0, scaler="invalid", classifier="lda", dim_method="pca"
    )
    # Vérifie que la construction du pipeline échoue explicitement
    with pytest.raises(
        ValueError, match="scaler must be 'standard', 'robust', or None"
    ):
        build_pipeline(config)


# Vérifie qu'une valeur invalide de classifieur lève une erreur explicite
def test_build_classifier_rejects_invalid_value():
    """Garantit une erreur claire pour un classifieur inconnu."""

    # Construit une configuration invalide pour le classifieur
    config = PipelineConfig(
        sfreq=100.0, scaler=None, classifier="invalid", dim_method="pca"
    )
    # Vérifie que la construction du pipeline échoue explicitement
    with pytest.raises(
        ValueError,
        match="classifier must be 'lda', 'logistic', 'svm', or 'centroid'",
    ):
        build_pipeline(config)


# Vérifie que le scaler retourne les bonnes classes scikit-learn
def test_build_scaler_supports_valid_variants():
    """Confirme le mapping scaler → instance scikit-learn."""

    # Vérifie l'instanciation d'un StandardScaler
    assert isinstance(_build_scaler("standard"), StandardScaler)
    # Vérifie l'instanciation d'un RobustScaler
    assert isinstance(_build_scaler("robust"), RobustScaler)
    # Vérifie que None retourne l'absence de scaler
    assert _build_scaler(None) is None


# Vérifie que le message d'erreur du scaler est explicite
def test_build_scaler_invalid_message_is_explicit():
    """Confirme le message d'erreur du scaler inconnu."""

    # Vérifie que la valeur inconnue produit un message précis
    with pytest.raises(
        ValueError, match="^scaler must be 'standard', 'robust', or None$"
    ):
        _build_scaler("unknown")


# Vérifie que les scalers acceptent des valeurs en majuscules
def test_build_scaler_accepts_uppercase_values():
    """Confirme l'insensibilité à la casse du scaler."""

    # Vérifie que STANDARD est accepté
    assert isinstance(_build_scaler("STANDARD"), StandardScaler)
    # Vérifie que ROBUST est accepté
    assert isinstance(_build_scaler("ROBUST"), RobustScaler)


# Vérifie que les classifieurs sont correctement instanciés
def test_build_classifier_supports_valid_variants():
    """Confirme le mapping classifieur → instance scikit-learn."""

    # Vérifie l'instanciation de LDA
    assert isinstance(_build_classifier("lda"), LinearDiscriminantAnalysis)
    # Vérifie l'instanciation de la régression logistique
    assert isinstance(_build_classifier("logistic"), LogisticRegression)
    # Vérifie l'instanciation du SVM linéaire
    assert isinstance(_build_classifier("svm"), LinearSVC)


# Vérifie que les paramètres du classifieur sont correctement réglés
def test_build_classifier_sets_expected_parameters():
    """Contrôle les hyperparamètres appliqués au classifieur."""

    # Construit une régression logistique pour inspection
    classifier = _build_classifier("logistic")
    # Vérifie que la classe instanciée est correcte
    assert isinstance(classifier, LogisticRegression)
    # Vérifie que le nombre d'itérations correspond à la constante
    assert classifier.max_iter == LOGISTIC_MAX_ITER


# Vérifie que les classifieurs acceptent les valeurs en majuscules
def test_build_classifier_accepts_uppercase_values():
    """Confirme l'insensibilité à la casse du classifieur."""

    # Vérifie l'instanciation de LDA en majuscules
    assert isinstance(_build_classifier("LDA"), LinearDiscriminantAnalysis)
    # Vérifie l'instanciation de LogisticRegression en majuscules
    assert isinstance(_build_classifier("LOGISTIC"), LogisticRegression)
    # Vérifie l'instanciation de LinearSVC en majuscules
    assert isinstance(_build_classifier("SVM"), LinearSVC)


# Vérifie que le message d'erreur du classifieur est explicite
def test_build_classifier_invalid_value_message():
    """Confirme le message d'erreur pour un classifieur inconnu."""

    # Vérifie que la valeur inconnue produit un message précis
    with pytest.raises(
        ValueError,
        match="^classifier must be 'lda', 'logistic', 'svm', or 'centroid'$",
    ):
        _build_classifier("unknown")


# Vérifie que les préprocesseurs utilisateur sont inclus dans la pipeline
def test_build_pipeline_includes_preprocessors_and_scaler():
    """S'assure que l'ordre des étapes respecte les préprocesseurs."""

    # Construit une liste de préprocesseurs fictifs
    preprocessors = [("dummy", object())]
    # Construit le pipeline avec scaler standard
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=200.0, scaler="standard", classifier="lda", dim_method="pca"
        ),
        preprocessors=preprocessors,
    )
    # Vérifie que toutes les étapes sont présentes et ordonnées
    assert list(pipeline.named_steps) == [
        "dummy",
        "features",
        "scaler",
        "dimensionality",
        "classifier",
    ]


# Vérifie que le scaler est omis lorsque non demandé
def test_build_pipeline_omits_scaler_when_none():
    """S'assure que la normalisation interne reste active sans scaler."""

    # Construit le pipeline sans scaler explicite
    pipeline = build_pipeline(
        PipelineConfig(sfreq=128.0, scaler=None, classifier="lda", dim_method="pca")
    )
    # Vérifie l'absence de scaler dans les étapes
    assert "scaler" not in pipeline.named_steps
    # Vérifie que la normalisation des features reste activée
    assert pipeline.named_steps["features"].normalize is True


# Vérifie que le drapeau de normalisation est correctement appliqué
def test_build_pipeline_applies_normalization_flag():
    """S'assure que l'option normalize_features est propagée."""

    # Construit le pipeline avec normalisation désactivée
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=256.0,
            scaler=None,
            classifier="svm",
            dim_method="pca",
            normalize_features=False,
        )
    )
    # Vérifie que la normalisation des features est bien désactivée
    assert pipeline.named_steps["features"].normalize is False


# Vérifie que toutes les stratégies configurables sont appliquées
def test_build_pipeline_uses_configured_strategies():
    """Contrôle le passage des paramètres à chaque étape."""

    # Construit le pipeline avec stratégies personnalisées
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=256.0,
            scaler="robust",
            classifier="logistic",
            dim_method="pca",
            n_components=CSP_COMPONENTS,
            feature_strategy="wavelet",
        )
    )
    # Vérifie la stratégie d'extraction de features
    assert pipeline.named_steps["features"].feature_strategy == "wavelet"
    # Vérifie la méthode du réducteur de dimension
    assert pipeline.named_steps["dimensionality"].method == "pca"
    # Vérifie le nombre de composantes configuré
    assert pipeline.named_steps["dimensionality"].n_components == CSP_COMPONENTS


# Vérifie que CSP saute l'extracteur de features et garde la régularisation
def test_build_pipeline_csp_skips_features_and_sets_regularization():
    """Confirme que CSP utilise la régularisation et évite les features."""

    # Construit un pipeline CSP avec régularisation explicite
    pipeline = build_pipeline(
        PipelineConfig(
            sfreq=128.0,
            scaler=None,
            classifier="lda",
            dim_method="csp",
            n_components=CSP_COMPONENTS,
            csp_regularization=0.25,
        )
    )
    # Vérifie que l'extracteur de features est absent en mode CSP
    assert "features" not in pipeline.named_steps
    # Vérifie que le réducteur de dimension est bien CSP
    assert pipeline.named_steps["dimensionality"].method == "csp"
    # Vérifie que la régularisation CSP est correctement propagée
    assert pipeline.named_steps["dimensionality"].regularization == 0.25

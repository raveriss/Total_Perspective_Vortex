# Importe NumPy pour générer des données synthétiques stables
import numpy as np

# Importe pytest pour vérifier les erreurs attendues
import pytest

# Importe le réducteur dimensionnel de TPV à vérifier
from tpv.dimensionality import TPVDimReducer

# Définit un seuil de domination pour la variance expliquée
DOMINANT_VARIANCE_THRESHOLD = 0.9


def test_csp_returns_log_variances_and_orthogonality() -> None:
    """CSP doit générer des filtres orthogonaux et séparer l'énergie."""

    # Prépare un générateur déterministe pour les données
    rng = np.random.default_rng(0)
    # Fixe le nombre d'essais par classe
    trials_per_class = 10
    # Fixe le nombre de canaux simulés
    channels = 3
    # Fixe le nombre d'échantillons temporels
    time_points = 64
    # Génère des essais faiblement énergétiques pour la classe A
    class_a = rng.standard_normal((trials_per_class, channels, time_points)) * 0.5
    # Génère des essais fortement énergétiques pour la classe B
    class_b = rng.standard_normal((trials_per_class, channels, time_points)) * 2.0
    # Concatène les essais pour former le jeu complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires associés
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Instancie le réducteur CSP avec régularisation légère
    reducer = TPVDimReducer(method="csp", n_components=channels, regularization=1e-3)
    # Apprend la matrice de projection sur les essais
    reducer.fit(trials, labels)
    # Transforme les essais pour obtenir des log-variances par composante
    projected = reducer.transform(trials)
    # Vérifie la forme des features renvoyées
    assert projected.shape == (trials_per_class * 2, channels)
    # Calcule la moyenne des features par classe pour vérifier la séparation
    mean_a = projected[:trials_per_class].mean(axis=0)
    # Calcule la moyenne pour la seconde classe
    mean_b = projected[trials_per_class:].mean(axis=0)
    # Vérifie que la classe énergique produit des variances plus élevées
    assert np.all(mean_b > mean_a)
    # Reconstruit la matrice composite pour vérifier l'orthogonalité
    composite = reducer._regularize_matrix(
        reducer._average_covariance(class_a) + reducer._average_covariance(class_b)
    )
    # Calcule le produit attendu proche de l'identité
    identity_candidate = reducer.w_matrix.T @ composite @ reducer.w_matrix
    # Vérifie l'orthogonalité dans l'espace régularisé
    assert np.allclose(identity_candidate, np.eye(channels), atol=1e-2)


def test_pca_explained_variance_and_projection_shape() -> None:
    """PCA doit ordonner les composantes par variance et rester orthonormé."""

    # Prépare un générateur déterministe pour les observations
    rng = np.random.default_rng(1)
    # Fixe le nombre d'échantillons
    samples = 120
    # Génère une composante principale dominante
    dominant = rng.standard_normal(samples) * 4.0
    # Génère deux composantes de bruit réduites
    noise1 = rng.standard_normal(samples) * 0.2
    noise2 = rng.standard_normal(samples) * 0.1
    # Assemble les observations en matrice (samples x features)
    observations = np.column_stack([dominant + noise1, noise1, noise2])
    # Instancie le PCA en gardant deux composantes
    reducer = TPVDimReducer(method="pca", n_components=2, regularization=1e-4)
    # Apprend la projection sur les observations
    reducer.fit(observations)
    # Transforme les observations pour obtenir les scores
    transformed = reducer.transform(observations)
    # Vérifie la forme de la projection obtenue
    assert transformed.shape == (samples, 2)
    # Vérifie que les vecteurs propres sont orthonormés
    assert np.allclose(reducer.w_matrix.T @ reducer.w_matrix, np.eye(2), atol=1e-6)
    # Calcule le ratio de variance expliquée
    variance_ratio = reducer.eigenvalues_ / reducer.eigenvalues_.sum()
    # Vérifie que la première composante capte la majorité de la variance
    assert variance_ratio[0] > DOMINANT_VARIANCE_THRESHOLD
    # Vérifie que la seconde composante reste strictement positive
    assert variance_ratio[1] > 0


def test_pca_regularization_stops_nan_on_duplicate_features() -> None:
    """La régularisation doit stabiliser une covariance singulière."""

    # Prépare une caractéristique unique répliquée deux fois
    base = np.linspace(0, 1, num=50)
    # Construit une matrice avec deux colonnes identiques
    duplicated = np.column_stack([base, base])
    # Instancie le PCA avec régularisation pour éviter les nan
    reducer = TPVDimReducer(method="pca", regularization=1e-3)
    # Apprend la projection malgré la singularité de la covariance
    reducer.fit(duplicated)
    # Transforme les données pour vérifier la stabilité numérique
    transformed = reducer.transform(duplicated)
    # Vérifie l'absence de valeurs non finies
    assert np.isfinite(transformed).all()


def test_save_and_load_roundtrip(tmp_path) -> None:
    """La sérialisation joblib doit préserver la projection apprise."""

    # Prépare des observations aléatoires pour le PCA
    rng = np.random.default_rng(5)
    # Génère une matrice d'observations tabulaires
    observations = rng.standard_normal((40, 3))
    # Instancie et entraîne le réducteur PCA
    reducer = TPVDimReducer(method="pca", n_components=2)
    # Apprend la projection sur les observations fournies
    reducer.fit(observations)
    # Transforme les observations pour servir de référence
    reference = reducer.transform(observations)
    # Définit le chemin de sauvegarde dans le répertoire temporaire
    target = tmp_path / "dim_reducer.joblib"
    # Sauvegarde le modèle entraîné
    reducer.save(target)
    # Crée une nouvelle instance pour charger le modèle
    restored = TPVDimReducer(method="pca")
    # Charge les paramètres depuis le fichier joblib
    restored.load(target)
    # Applique la transformation avec le modèle restauré
    loaded_projection = restored.transform(observations)
    # Vérifie que la projection rechargée correspond à la référence
    assert np.allclose(loaded_projection, reference)


def test_validation_guards_raise_errors_for_invalid_calls() -> None:
    """Les garde-fous doivent empêcher les appels incohérents."""

    # Prépare une matrice tabulaire pour tester les validations PCA
    tabular = np.ones((4, 3))
    # Prépare un tenseur tridimensionnel pour les validations CSP
    trials = np.ones((2, 2, 5))
    # Instancie PCA avec une méthode incorrecte pour vérifier la validation
    invalid_method = TPVDimReducer(method="unknown")
    # Vérifie que la méthode inconnue est rejetée
    with pytest.raises(ValueError):
        invalid_method.fit(tabular)
    # Instancie PCA pour déclencher l'erreur de dimension attendue
    pca = TPVDimReducer(method="pca")
    # Vérifie que PCA refuse des données non tabulaires
    with pytest.raises(ValueError):
        pca.fit(trials)
    # Instancie CSP pour tester l'absence de labels
    csp_missing_labels = TPVDimReducer(method="csp")
    # Vérifie que CSP réclame explicitement les labels
    with pytest.raises(ValueError):
        csp_missing_labels.fit(trials)
    # Instancie CSP pour tester la dimension incorrecte
    csp_wrong_shape = TPVDimReducer(method="csp")
    # Vérifie que CSP refuse une entrée qui n'est pas 3D
    with pytest.raises(ValueError):
        csp_wrong_shape.fit(tabular, np.array([0, 1, 0, 1]))
    # Instancie CSP pour tester le nombre de classes invalide
    csp_many_classes = TPVDimReducer(method="csp")
    # Construit des labels comprenant trois classes distinctes
    too_many_labels = np.array([0, 1, 2])
    # Vérifie que CSP exige exactement deux classes
    with pytest.raises(ValueError):
        csp_many_classes.fit(trials[:3], too_many_labels)
    # Instancie CSP pour vérifier la protection sur la sauvegarde
    csp_unsaved = TPVDimReducer(method="csp")
    # Vérifie que la sauvegarde échoue si fit n'a pas été appelé
    with pytest.raises(ValueError):
        csp_unsaved.save("/tmp/unused.joblib")
    # Instancie CSP pour vérifier la protection sur la transformation
    csp_unfitted = TPVDimReducer(method="csp")
    # Vérifie que transform refuse d'être appelé avant fit
    with pytest.raises(ValueError):
        csp_unfitted.transform(trials)


def test_csp_handles_default_component_count_and_empty_trials_guard() -> None:
    """CSP doit couvrir les branches par défaut et refuser les essais vides."""

    # Prépare un générateur déterministe pour reproduire les résultats
    rng = np.random.default_rng(7)
    # Fixe le nombre de canaux pour aligner l'attente des composantes
    channels = 2
    # Génère des essais pour la classe A
    class_a = rng.standard_normal((5, channels, 16))
    # Génère des essais pour la classe B
    class_b = rng.standard_normal((5, channels, 16))
    # Concatène les essais pour former le jeu complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires associés
    labels = np.array([0] * 5 + [1] * 5)
    # Instancie le CSP sans spécifier n_components pour couvrir la branche par défaut
    reducer = TPVDimReducer(method="csp")
    # Apprend la matrice de projection en mode par défaut
    reducer.fit(trials, labels)
    # Vérifie que la matrice de projection conserve toutes les composantes
    assert reducer.w_matrix.shape[1] == channels
    # Transforme les essais pour préparer un appel avec une dimension invalide
    _ = reducer.transform(trials)
    # Vérifie que transform refuse une dimension inattendue
    with pytest.raises(ValueError):
        reducer.transform(np.ones(5))
    # Instancie un nouvel objet pour tester la covariance vide
    empty_guard = TPVDimReducer(method="csp")
    # Vérifie que la moyenne de covariance échoue sur un tableau vide
    with pytest.raises(ValueError):
        empty_guard._average_covariance(np.empty((0, channels, 4)))

# Importe numpy pour créer des matrices jouets et mesurer l'orthogonalité
import numpy as np

# Importe pytest pour valider les erreurs et les comparaisons numériques
import pytest

# Importe le réducteur dimensionnel de TPV à vérifier
from tpv.dimensionality import TPVDimReducer


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
    assert variance_ratio[0] > 0.9
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

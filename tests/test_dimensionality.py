"""Tests unitaires pour la réduction de dimension TPV."""

# Importe numpy pour générer des données synthétiques
# Utilise Path pour manipuler les chemins de manière sûre
from pathlib import Path

import numpy as np

# Importe le réducteur pour valider son comportement
from tpv.dimensionality import TPVDimReducer

# Fige l'écart de variance minimal attendu entre les classes
VARIANCE_GAP_THRESHOLD = 0.5


# Vérifie que la PCA produit une base orthonormée et reconstructible
def test_pca_projection_is_orthonormal_and_reconstructs():
    # Crée une matrice de données centrée artificiellement
    X = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    # Instancie le réducteur en mode PCA
    reducer = TPVDimReducer(method="pca")
    # Apprend la matrice de projection à partir de X
    reducer.fit(X)
    # Calcule l'orthogonalité en vérifiant W^T W = I
    identity = reducer.w_matrix.T @ reducer.w_matrix
    # Vérifie que la matrice est proche de l'identité
    np.testing.assert_allclose(identity, np.eye(identity.shape[0]), atol=1e-6)
    # Projette les données puis reconstruit dans l'espace initial
    projected = reducer.transform(X)
    # Recompose en appliquant l'inverse orthonormée
    reconstructed = projected @ reducer.w_matrix.T + reducer.mean_
    # Vérifie que la reconstruction approche les données originales
    np.testing.assert_allclose(reconstructed, X, atol=1e-6)


# Vérifie que le CSP sépare des classes aux covariances distinctes
def test_csp_separates_covariances():
    # Crée un générateur déterministe pour stabiliser le test
    rng = np.random.default_rng(42)
    # Définit un nombre de canaux et d'échantillons temporels
    n_channels = 2
    # Fixe la durée en échantillons pour chaque essai
    n_times = 50
    # Génère des essais pour la classe 0 avec variance dominante sur le canal 0
    class0 = rng.standard_normal((20, n_channels, n_times))
    # Amplifie le premier canal pour augmenter sa variance relative
    class0[:, 0, :] *= 2.0
    # Génère des essais pour la classe 1 avec variance dominante sur le canal 1
    class1 = rng.standard_normal((20, n_channels, n_times))
    # Amplifie le second canal pour créer un contraste de variance
    class1[:, 1, :] *= 2.0
    # Concatène les essais en un seul tableau
    X = np.concatenate([class0, class1], axis=0)
    # Crée les étiquettes correspondantes
    y = np.array([0] * class0.shape[0] + [1] * class1.shape[0])
    # Instancie le réducteur en mode CSP avec deux composantes
    reducer = TPVDimReducer(method="csp", n_components=2)
    # Apprend la matrice de projection CSP
    reducer.fit(X, y)
    # Projette les essais complets pour préserver la dynamique temporelle
    projected0 = reducer.transform(class0)
    # Projette la classe 1 pour comparaison
    projected1 = reducer.transform(class1)
    # Calcule la variance spatio-temporelle sur la première composante
    var0 = np.var(projected0[:, 0, :])
    # Calcule la variance sur la même composante pour l'autre classe
    var1 = np.var(projected1[:, 0, :])
    # Vérifie que les variances sont nettement différentes
    assert abs(var0 - var1) > VARIANCE_GAP_THRESHOLD


# Vérifie que la matrice peut être sauvegardée puis rechargée
def test_save_and_load_projection(tmp_path: Path):
    # Crée des données simples pour la PCA
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    # Initialise le réducteur
    reducer = TPVDimReducer(method="pca")
    # Apprend la projection
    reducer.fit(X)
    # Prépare un chemin temporaire pour la sauvegarde
    target = tmp_path / "projection.joblib"
    # Sauvegarde la matrice apprise
    reducer.save(target)
    # Crée un nouveau réducteur vide
    restored = TPVDimReducer(method="pca")
    # Charge la matrice précédemment sauvegardée
    restored.load(target)
    # Vérifie que les matrices sont identiques
    np.testing.assert_allclose(restored.w_matrix, reducer.w_matrix)
    # Vérifie que la moyenne est préservée
    np.testing.assert_allclose(restored.mean_, reducer.mean_)

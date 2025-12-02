"""Réduction de dimension pour TPV."""

# Garantit l'accès aux types manipulables par joblib et pathlib
import os

# Offre une persistance simple de la matrice de projection
import joblib

# Garantit que numpy est disponible pour le calcul matriciel
import numpy as np

# Garantit l'accès aux décompositions hermitiennes généralisées
from scipy import linalg

# Assure l'intégration avec les API scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin

# Fige le nombre de classes attendu pour le CSP pour lever les ambiguïtés
EXPECTED_CSP_CLASSES = 2
# Fige la dimension tabulaire standard pour différencier l'entrée
TABLE_DIMENSION = 2
# Fige la dimension trial x channel x time attendue pour le CSP
TRIAL_DIMENSION = 3


class TPVDimReducer(BaseEstimator, TransformerMixin):
    """Réducteur de dimension via PCA ou CSP."""

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def __init__(self, method: str = "pca", n_components: int | None = None):
        # Conserve la méthode demandée par l'utilisateur
        self.method = method
        # Conserve le nombre de composantes souhaité
        self.n_components = n_components
        # Initialise la matrice de projection à None avant apprentissage
        self.w_matrix: np.ndarray
        # Initialise la moyenne pour la centration éventuelle
        self.mean_: np.ndarray
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None  # type: ignore[assignment]
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None  # type: ignore[assignment]

    # Apprend la matrice de projection à partir des données et des labels
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance empirique des features
            covariance = np.cov(centered, rowvar=False)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Transforme les données en appliquant la matrice de projection apprise
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is not None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = X - self.mean_
        # Applique la projection aux données tabulaires classiques
        if X.ndim == TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            return np.asarray(np.einsum("ij,ajt->ait", self.w_matrix.T, X))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Enregistre la matrice de projection pour réutilisation future
    def save(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump({"w_matrix": self.w_matrix, "mean": self.mean_}, str(path))

    # Charge la matrice de projection depuis un fichier joblib
    def load(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def _average_covariance(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        return np.asarray(cov_sum / trials.shape[0])

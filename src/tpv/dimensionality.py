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
    def __init__(
        self,
        method: str = "csp",
        n_components: int | None = None,
        regularization: float = 0.0,
    ):
        # Conserve la méthode demandée par l'utilisateur
        self.method = method
        # Conserve le nombre de composantes souhaité
        self.n_components = n_components
        # Conserve la régularisation ajoutée aux covariances
        self.regularization = regularization
        # Initialise la matrice de projection à None avant apprentissage
        self.w_matrix: np.ndarray | None
        # Initialise la moyenne pour la centration éventuelle
        self.mean_: np.ndarray | None
        # Prépare le stockage des valeurs propres pour validation et débogage
        self.eigenvalues_: np.ndarray | None
        # Prépare le stockage des valeurs singulières pour l'option SVD
        self.singular_values_: np.ndarray | None
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None
        # Positionne None avant calcul des valeurs singulières
        self.singular_values_ = None

    # Apprend la matrice de projection à partir des données et des labels
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp", "svd"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca', 'csp', or 'svd'")
        # Délègue le calcul aux helpers spécialisés pour réduire la complexité
        if self.method == "pca":
            self._fit_pca(X)
        elif self.method == "svd":
            self._fit_svd(X)
        else:
            self._fit_csp(X, y)
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
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Enregistre la matrice de projection pour réutilisation future
    def save(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "singular_values": self.singular_values_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Charge la matrice de projection depuis un fichier joblib
    def load(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure les valeurs singulières éventuelles
        self.singular_values_ = data.get("singular_values")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

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
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule une covariance régularisée pour les données tabulaires
    def _regularized_covariance(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def _regularize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Applique l'apprentissage PCA sur des données tabulaires
    def _fit_pca(self, X: np.ndarray) -> None:
        """Apprend la projection PCA via la covariance régularisée."""

        # Vérifie que les données sont tabulaires pour PCA
        if X.ndim != TABLE_DIMENSION:
            # Informe que PCA attend des données échantillon x feature
            raise ValueError("PCA expects a 2D array")
        # Centre les données pour une covariance cohérente
        self.mean_ = np.mean(X, axis=0)
        # Calcule les données centrées pour la covariance
        centered = X - self.mean_
        # Calcule la covariance avec régularisation diagonale
        covariance = self._regularized_covariance(centered)
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
            # Tronque aussi la liste des valeurs propres
            eigvals = eigvals[order][: self.n_components]
        else:
            # Conserve toutes les valeurs propres si aucune coupe n'est demandée
            eigvals = eigvals[order]
        # Stocke la matrice de projection apprise
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres associées pour vérification externe
        self.eigenvalues_ = eigvals
        # Réinitialise les valeurs singulières quand PCA est utilisé
        self.singular_values_ = None

    # Applique l'apprentissage SVD sur des données tabulaires
    def _fit_svd(self, X: np.ndarray) -> None:
        """Apprend la projection SVD via la covariance centrée."""

        # Vérifie que les données sont tabulaires pour la SVD
        if X.ndim != TABLE_DIMENSION:
            # Informe que SVD attend des données échantillon x feature
            raise ValueError("SVD expects a 2D array")
        # Centre les données pour stabiliser la covariance
        self.mean_ = np.mean(X, axis=0)
        # Calcule les données centrées pour la SVD
        centered = X - self.mean_
        # Calcule une SVD maison à partir de la covariance centrée
        _u_matrix, singular_values, v_matrix = self._svd_from_covariance(centered)
        # Limite le nombre de composantes si demandé
        if self.n_components is not None:
            # Tronque la base de projection au nombre souhaité
            v_matrix = v_matrix[:, : self.n_components]
            # Tronque les valeurs singulières
            singular_values = singular_values[: self.n_components]
        # Stocke la matrice de projection apprise
        self.w_matrix = v_matrix
        # Stocke les valeurs singulières pour inspection
        self.singular_values_ = singular_values
        # Dérive les valeurs propres de la covariance à partir des singulières
        self.eigenvalues_ = (
            singular_values**2 / (centered.shape[0] - 1)
            if centered.shape[0] > 1
            else singular_values**2
        )

    # Applique l'apprentissage CSP sur des essais EEG 3D
    def _fit_csp(self, X: np.ndarray, y: np.ndarray | None) -> None:
        """Apprend la projection CSP à partir des essais et labels."""

        # Vérifie la présence des étiquettes pour la méthode CSP
        if y is None:
            # Informe que CSP requiert des labels binaires
            raise ValueError("y is required for CSP")
        # Valide la dimension trial x channel x time attendue
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP demande des essais temporels bruts
            raise ValueError("CSP expects a 3D array")
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
        # Ajoute une régularisation pour stabiliser l'inversion implicite
        composite = self._regularize_matrix(composite)
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
            # Tronque également les valeurs propres associées
            eigvals = eigvals[order][: self.n_components]
        else:
            # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
            eigvals = eigvals[order]
        # Stocke la matrice de projection CSP
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres pour inspection éventuelle
        self.eigenvalues_ = eigvals
        # Réinitialise les valeurs singulières pour CSP
        self.singular_values_ = None

    # Reconstruit une SVD via la covariance pour conserver un contrôle local
    def _svd_from_covariance(
        self, centered: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit une SVD maison via la covariance centrée."""

        # Vérifie que les données sont bien 2D pour la SVD
        if centered.ndim != TABLE_DIMENSION:
            # Signale une incohérence de dimension avant le calcul
            raise ValueError("SVD expects a 2D array")
        # Calcule la covariance régularisée pour stabiliser le spectre
        covariance = self._regularized_covariance(centered)
        # Résout la décomposition en valeurs propres de la covariance
        eigvals, eigvecs = np.linalg.eigh(covariance)
        # Trie les valeurs propres en ordre décroissant
        order = np.argsort(eigvals)[::-1]
        # Réordonne les vecteurs propres en conséquence
        eigvecs = eigvecs[:, order]
        # Force les valeurs propres négatives à zéro pour stabilité numérique
        eigvals = np.clip(eigvals[order], 0.0, None)
        # Convertit les valeurs propres en valeurs singulières de X
        singular_values = np.sqrt(eigvals * max(centered.shape[0] - 1, 1))
        # Construit V (vecteurs propres) pour la projection
        v_matrix = eigvecs
        # Calcule U en normalisant les projections sur V
        u_matrix = centered @ v_matrix
        # Évite les divisions par zéro en masquant les valeurs singulières nulles
        nonzero = singular_values > 0
        if np.any(nonzero):
            # Normalise chaque colonne de U par la valeur singulière associée
            u_matrix[:, nonzero] = u_matrix[:, nonzero] / singular_values[nonzero]
        # Retourne les matrices U, S, V pour inspection éventuelle
        return u_matrix, singular_values, v_matrix


# Implémente un transformeur CSP/CSSP pour les signaux EEG
class CSP(BaseEstimator, TransformerMixin):
    """Transformeur CSP/CSSP appliquant W^T X = X_CSP sur X ∈ R^{d×N}."""

    # Déclare le constructeur pour configurer CSP ou CSSP
    def __init__(
        self,
        n_components: int | None = None,
        regularization: float = 0.0,
        method: str = "csp",
        cssp_lag: int = 1,
        return_log_variance: bool = True,
    ):
        # Conserve le nombre de composantes souhaité
        self.n_components = n_components
        # Conserve la régularisation appliquée aux covariances
        self.regularization = regularization
        # Conserve la variante de l'algorithme à appliquer
        self.method = method
        # Conserve le décalage temporel pour CSSP
        self.cssp_lag = cssp_lag
        # Conserve le choix de sortie log-variance pour le classifieur
        self.return_log_variance = return_log_variance
        # Prépare la matrice de projection avant apprentissage
        self.w_matrix: np.ndarray | None
        # Prépare le stockage des valeurs propres pour inspection
        self.eigenvalues_: np.ndarray | None
        # Positionne None avant le calcul des filtres
        self.w_matrix = None
        # Positionne None avant la résolution du problème généralisé
        self.eigenvalues_ = None

    # Apprend les filtres spatiaux CSP/CSSP à partir des essais
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Refuse l'appel sans labels pour éviter un apprentissage invalide
        if y is None:
            # Explique que CSP exige des labels de classe
            raise ValueError("y is required for CSP/CSSP")
        # Valide la dimension trial x channel x time attendue
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP/CSSP attend des essais 3D
            raise ValueError("CSP/CSSP expects a 3D array")
        # Valide la variante CSP/CSSP demandée
        if self.method not in {"csp", "cssp"}:
            # Signale un paramètre de méthode non supporté
            raise ValueError("method must be 'csp' or 'cssp'")
        # Identifie les classes présentes pour vérifier le binaire
        classes = np.unique(y)
        # Valide que le problème est strictement binaire
        if classes.size != EXPECTED_CSP_CLASSES:
            # Empêche le calcul CSP/CSSP hors contrat binaire
            raise ValueError("CSP/CSSP requires exactly two classes")
        # Prépare les essais selon la variante choisie
        trials = X
        # Applique l'augmentation CSSP si demandée
        if self.method == "cssp":
            # Transforme les essais en espace élargi temps-décalé
            trials = self._augment_cssp_trials(trials)
        # Calcule la covariance moyenne de la première classe
        cov_a = self._average_covariance(trials[y == classes[0]])
        # Calcule la covariance moyenne de la seconde classe
        cov_b = self._average_covariance(trials[y == classes[1]])
        # Construit la covariance composite pour le problème généralisé
        composite = cov_a + cov_b
        # Régularise la covariance composite pour stabilité numérique
        composite = self._regularize_matrix(composite)
        # Résout le problème généralisé pour maximiser la séparation
        eigvals, eigvecs = linalg.eigh(cov_a, composite)
        # Trie les composantes par valeurs propres décroissantes
        order = np.argsort(eigvals)[::-1]
        # Réordonne les vecteurs propres selon l'importance
        sorted_vecs = eigvecs[:, order]
        # Coupe le nombre de filtres si demandé
        if self.n_components is not None:
            # Conserve les filtres les plus informatifs
            sorted_vecs = sorted_vecs[:, : self.n_components]
            # Conserve les valeurs propres associées
            eigvals = eigvals[order][: self.n_components]
        else:
            # Conserve toutes les valeurs propres si aucune coupe n'est demandée
            eigvals = eigvals[order]
        # Stocke la matrice de filtres spatiaux
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres pour inspection ultérieure
        self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Applique les filtres CSP/CSSP pour projeter les signaux
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Vérifie la disponibilité des filtres appris
        if self.w_matrix is None:
            # Interdit la projection avant l'apprentissage
            raise ValueError("The CSP/CSSP model must be fitted before transform")
        # Valide la dimension attendue des essais
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP/CSSP attend des essais 3D
            raise ValueError("CSP/CSSP expects a 3D array")
        # Prépare les essais selon la variante choisie
        trials = X
        # Applique l'augmentation CSSP si demandée
        if self.method == "cssp":
            # Transforme les essais en espace élargi temps-décalé
            trials = self._augment_cssp_trials(trials)
        # Projette les essais sur les filtres spatiaux
        projected = np.tensordot(self.w_matrix.T, trials, axes=([1], [1]))
        # Réordonne les axes pour revenir à trial x composante x temps
        reordered = np.moveaxis(projected, 1, 0)
        # Renvoie les signaux projetés si demandé
        if not self.return_log_variance:
            # Retourne directement W^T X pour les étapes suivantes
            return np.asarray(reordered)
        # Calcule la variance par composante pour chaque essai
        variances = np.var(reordered, axis=2)
        # Stabilise la variance via le log pour les classifieurs
        return np.asarray(np.log(variances + np.finfo(float).eps))

    # Calcule la moyenne des covariances pour une classe d'essais
    def _average_covariance(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une moyenne vide
        if trials.size == 0:
            # Signale qu'une classe vide est invalide pour CSP/CSSP
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une matrice d'accumulation stable
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance normalisée
        for trial in trials:
            # Calcule la covariance d'un essai pour capturer l'énergie spatiale
            trial_cov = trial @ trial.T
            # Normalise par la trace pour une échelle comparable
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne des covariances sur la classe
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Retourne la covariance régularisée pour la stabilité
        return self._regularize_matrix(averaged)

    # Ajoute une régularisation diagonale pour stabiliser les covariances
    def _regularize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Copie la matrice pour éviter toute mutation in-place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation si elle est activée
        if self.regularization > 0:
            # Stabilise l'inversion via une identité scalée
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice prête pour la décomposition
        return regularized

    # Construit les essais augmentés pour CSSP via un retard temporel
    def _augment_cssp_trials(self, trials: np.ndarray) -> np.ndarray:
        # Valide un lag strictement positif pour éviter un doublon
        if self.cssp_lag < 1:
            # Signale que le lag CSSP doit être positif
            raise ValueError("cssp_lag must be >= 1")
        # Refuse un lag supérieur à la durée pour garder des segments valides
        if trials.shape[2] <= self.cssp_lag:
            # Signale que le lag est trop grand pour la fenêtre
            raise ValueError("cssp_lag is too large for the trial length")
        # Extrait le signal sans les derniers échantillons
        base = trials[:, :, : -self.cssp_lag]
        # Extrait le signal décalé pour capturer la dynamique temporelle
        delayed = trials[:, :, self.cssp_lag :]
        # Concatène les canaux originaux et décalés pour CSSP
        return np.concatenate([base, delayed], axis=1)

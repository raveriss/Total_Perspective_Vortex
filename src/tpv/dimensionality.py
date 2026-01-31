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
        **ica_params: int | float | None,
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
        # Définit les paramètres ICA attendus et leurs défauts
        default_ica_params: dict[str, int | float | None] = {
            "ica_max_iter": 200,
            "ica_tol": 1e-4,
            "ica_random_state": 0,
            "ica_min_epochs_per_channel": 20,
            "ica_residual_threshold": 0.05,
        }
        # Valide les clés fournies pour éviter des erreurs silencieuses
        for key in ica_params:
            # Refuse les clés inattendues pour garder un contrat explicite
            if key not in default_ica_params:
                # Signale la clé invalide à l'utilisateur
                raise ValueError(f"Unknown ICA parameter: {key}")
        # Fusionne les paramètres par défaut avec les overrides fournis
        merged_ica_params = {**default_ica_params, **ica_params}
        # Extrait max_iter pour valider sa présence
        max_iter = merged_ica_params["ica_max_iter"]
        # Refuse un max_iter nul pour protéger les conversions
        if max_iter is None:
            # Signale l'absence de max_iter attendu
            raise ValueError("ica_max_iter cannot be None")
        # Conserve la limite d'itérations pour la convergence ICA
        self.ica_max_iter = int(max_iter)
        # Extrait la tolérance ICA pour validation
        tol = merged_ica_params["ica_tol"]
        # Refuse une tolérance manquante pour la conversion float
        if tol is None:
            # Signale l'absence de tolérance ICA
            raise ValueError("ica_tol cannot be None")
        # Conserve la tolérance de convergence pour l'ICA
        self.ica_tol = float(tol)
        # Conserve la graine pour stabiliser l'ICA
        self.ica_random_state = (
            int(merged_ica_params["ica_random_state"])
            if merged_ica_params["ica_random_state"] is not None
            else None
        )
        # Extrait le ratio d'échantillons ICA pour validation
        min_epochs = merged_ica_params["ica_min_epochs_per_channel"]
        # Refuse un ratio manquant pour l'ICA
        if min_epochs is None:
            # Signale l'absence de ratio d'échantillons ICA
            raise ValueError("ica_min_epochs_per_channel cannot be None")
        # Conserve le ratio minimum d'échantillons par canal pour l'ICA
        self.ica_min_epochs_per_channel = int(min_epochs)
        # Extrait le seuil de résidu ICA pour validation
        residual_threshold = merged_ica_params["ica_residual_threshold"]
        # Refuse un seuil manquant pour l'ICA
        if residual_threshold is None:
            # Signale l'absence de seuil de résidu ICA
            raise ValueError("ica_residual_threshold cannot be None")
        # Conserve le seuil de résidu relatif accepté pour l'ICA
        self.ica_residual_threshold = float(residual_threshold)
        # Prépare le statut de convergence de l'ICA
        self.ica_converged_: bool | None
        # Prépare le nombre d'itérations effectives de l'ICA
        self.ica_n_iter_: int | None
        # Prépare la norme du résidu de reconstruction ICA
        self.ica_residual_norm_: float | None
        # Prépare la matrice de whitening ICA
        self.ica_whitening_: np.ndarray | None
        # Prépare la matrice de démixage ICA
        self.ica_unmixing_: np.ndarray | None
        # Prépare la matrice de mixage ICA
        self.ica_mixing_: np.ndarray | None
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None
        # Positionne None avant calcul des valeurs singulières
        self.singular_values_ = None
        # Positionne None pour indiquer l'absence de convergence ICA
        self.ica_converged_ = None
        # Positionne None avant le calcul des itérations ICA
        self.ica_n_iter_ = None
        # Positionne None avant le calcul du résidu ICA
        self.ica_residual_norm_ = None
        # Positionne None avant le calcul du whitening ICA
        self.ica_whitening_ = None
        # Positionne None avant le calcul du démixage ICA
        self.ica_unmixing_ = None
        # Positionne None avant le calcul du mixage ICA
        self.ica_mixing_ = None

    # Apprend la matrice de projection à partir des données et des labels
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp", "svd", "ica"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca', 'csp', 'svd', or 'ica'")
        # Délègue le calcul aux helpers spécialisés pour réduire la complexité
        if self.method == "pca":
            self._fit_pca(X)
        elif self.method == "svd":
            self._fit_svd(X)
        elif self.method == "ica":
            self._fit_ica(X)
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
            # Empêche l'ICA d'être appliquée à des essais bruts 3D
            if self.method == "ica":
                # Informe que l'ICA attend des données tabulaires
                raise ValueError("ICA expects a 2D array")
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
                # Sérialise max_iter ICA pour reproduction exacte
                "ica_max_iter": self.ica_max_iter,
                # Sérialise la tolérance ICA pour la convergence
                "ica_tol": self.ica_tol,
                # Sérialise la graine ICA pour reproductibilité
                "ica_random_state": self.ica_random_state,
                # Sérialise le ratio min d'échantillons ICA
                "ica_min_epochs_per_channel": self.ica_min_epochs_per_channel,
                # Sérialise le seuil de résidu ICA
                "ica_residual_threshold": self.ica_residual_threshold,
                # Sérialise l'état de convergence ICA
                "ica_converged": self.ica_converged_,
                # Sérialise le nombre d'itérations ICA
                "ica_n_iter": self.ica_n_iter_,
                # Sérialise le résidu ICA
                "ica_residual_norm": self.ica_residual_norm_,
                # Sérialise le whitening ICA
                "ica_whitening": self.ica_whitening_,
                # Sérialise le démixage ICA
                "ica_unmixing": self.ica_unmixing_,
                # Sérialise le mixage ICA
                "ica_mixing": self.ica_mixing_,
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
        # Restaure la limite d'itérations ICA si disponible
        self.ica_max_iter = data.get("ica_max_iter", self.ica_max_iter)
        # Restaure la tolérance ICA si disponible
        self.ica_tol = data.get("ica_tol", self.ica_tol)
        # Restaure la graine ICA si disponible
        self.ica_random_state = data.get("ica_random_state", self.ica_random_state)
        # Restaure le seuil d'échantillons ICA si disponible
        self.ica_min_epochs_per_channel = data.get(
            "ica_min_epochs_per_channel", self.ica_min_epochs_per_channel
        )
        # Restaure le seuil de résidu ICA si disponible
        self.ica_residual_threshold = data.get(
            "ica_residual_threshold", self.ica_residual_threshold
        )
        # Restaure l'état de convergence ICA si disponible
        self.ica_converged_ = data.get("ica_converged", self.ica_converged_)
        # Restaure le nombre d'itérations ICA si disponible
        self.ica_n_iter_ = data.get("ica_n_iter", self.ica_n_iter_)
        # Restaure le résidu ICA si disponible
        self.ica_residual_norm_ = data.get("ica_residual_norm", self.ica_residual_norm_)
        # Restaure la matrice de whitening ICA si disponible
        self.ica_whitening_ = data.get("ica_whitening", self.ica_whitening_)
        # Restaure la matrice de démixage ICA si disponible
        self.ica_unmixing_ = data.get("ica_unmixing", self.ica_unmixing_)
        # Restaure la matrice de mixage ICA si disponible
        self.ica_mixing_ = data.get("ica_mixing", self.ica_mixing_)

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
        """Apprend la projection PCA via C = Xc^T Xc / (n-1).

        Référence: Jolliffe, I. (2002) Principal Component Analysis.
        """

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

    # Applique l'apprentissage ICA sur des données tabulaires
    def _fit_ica(self, X: np.ndarray) -> None:
        """Apprend une ICA maison telle que W^T X = S.

        Référence: Hyvärinen, A. & Oja, E. (2000) Independent Component Analysis.
        """

        # Vérifie que l'ICA reçoit bien une matrice tabulaire
        if X.ndim != TABLE_DIMENSION:
            # Informe explicitement que l'ICA ne gère pas les essais 3D
            raise ValueError("ICA expects a 2D array")
        # Capture le nombre d'échantillons et de features
        n_samples, n_features = X.shape
        # Valide la configuration ICA avant les calculs coûteux
        self._validate_ica_settings()
        # Valide que le ratio d'échantillons est suffisant
        self._validate_ica_sample_count(n_samples, n_features)
        # Détermine le nombre de composantes ICA final
        n_components = self._resolve_ica_components(n_features)
        # Centre les données pour satisfaire l'hypothèse de moyenne nulle
        self.mean_ = np.mean(X, axis=0)
        # Calcule les données centrées pour l'ICA
        centered = X - self.mean_
        # Calcule le whitening et sa pseudo-inverse
        whitening, dewhitening, eigenvalues = self._ica_whitening(
            centered, n_components
        )
        # Stocke le whitening pour inspection et validation
        self.ica_whitening_ = whitening
        # Stocke le spectre de la covariance whitenée pour diagnostics
        self.eigenvalues_ = eigenvalues
        # Réinitialise les valeurs singulières pour l'ICA
        self.singular_values_ = None
        # Applique le whitening aux données centrées
        whitened = centered @ whitening
        # Exécute l'algorithme FastICA pour trouver les composantes
        weights, n_iter, converged = self._run_fastica(whitened, n_components)
        # Stocke le nombre d'itérations effectives
        self.ica_n_iter_ = n_iter
        # Stocke l'état de convergence ICA
        self.ica_converged_ = converged
        # Refuse de poursuivre si la convergence n'est pas atteinte
        if not converged:
            # Informe explicitement de l'échec de convergence ICA
            raise ValueError("ICA did not converge within max_iter")
        # Stocke la matrice de démixage ICA finale
        self.ica_unmixing_ = weights
        # Calcule la matrice de mixage ICA associée
        self.ica_mixing_ = np.linalg.pinv(weights)
        # Calcule la matrice de projection globale pour transform
        self.w_matrix = whitening @ weights.T
        # Calcule la norme du résidu relatif de reconstruction
        self.ica_residual_norm_ = self._ica_residual(
            centered, whitened, dewhitening, weights
        )
        # Refuse l'ICA si le résidu dépasse le seuil toléré
        if self.ica_residual_norm_ > self.ica_residual_threshold:
            # Informe explicitement d'un résidu ICA trop élevé
            raise ValueError(
                "ICA residual norm "
                f"{self.ica_residual_norm_:.6f} exceeds threshold "
                f"{self.ica_residual_threshold:.6f}"
            )

    # Valide les paramètres ICA fournis
    def _validate_ica_settings(self) -> None:
        """Valide les paramètres ICA avant exécution."""

        # Valide le nombre d'itérations pour éviter une boucle vide
        if self.ica_max_iter < 1:
            # Signale que max_iter doit être positif
            raise ValueError("ICA max_iter must be >= 1")
        # Valide le seuil d'échantillons pour éviter un ratio nul
        if self.ica_min_epochs_per_channel < 1:
            # Signale que le seuil doit être strictement positif
            raise ValueError("ICA min_epochs_per_channel must be >= 1")
        # Valide que le seuil de résidu reste positif
        if self.ica_residual_threshold <= 0:
            # Signale qu'un seuil non positif ne peut pas valider la qualité
            raise ValueError("ICA residual threshold must be > 0")

    # Valide le ratio d'échantillons par canal pour l'ICA
    def _validate_ica_sample_count(self, n_samples: int, n_features: int) -> None:
        """Vérifie que l'ICA dispose d'assez d'échantillons."""

        # Calcule le minimum requis pour un ratio d'échantillons par canal
        min_samples = self.ica_min_epochs_per_channel * n_features
        # Refuse les jeux trop petits pour l'estimation ICA
        if n_samples < min_samples:
            # Décrit le manque de données pour l'ICA
            raise ValueError(
                "ICA requires at least "
                f"{min_samples} samples for {n_features} channels"
            )

    # Détermine le nombre de composantes ICA valides
    def _resolve_ica_components(self, n_features: int) -> int:
        """Détermine n_components pour l'ICA."""

        # Fixe le nombre de composantes ICA si non précisé
        n_components = self.n_components or n_features
        # Vérifie que la demande de composantes reste valide
        if n_components > n_features:
            # Informe que l'ICA ne peut pas produire plus de composantes
            raise ValueError("ICA n_components cannot exceed n_features")
        # Retourne le nombre de composantes validé
        return n_components

    # Exécute l'algorithme FastICA symétrique
    def _run_fastica(
        self, whitened: np.ndarray, n_components: int
    ) -> tuple[np.ndarray, int, bool]:
        """Exécute FastICA sur des données whitenées."""

        # Capture le nombre d'échantillons pour la mise à jour
        n_samples = whitened.shape[0]
        # Initialise un générateur pour des poids ICA reproductibles
        rng = np.random.default_rng(self.ica_random_state)
        # Génère une matrice initiale aléatoire pour l'ICA
        weights = rng.standard_normal((n_components, n_components))
        # Orthogonalise l'initialisation pour stabiliser l'ICA
        weights = self._sym_decorrelation(weights)
        # Lance l'itération FastICA pour trouver les composantes indépendantes
        for iteration in range(self.ica_max_iter):
            # Conserve l'ancienne valeur pour tester la convergence
            weights_old = weights
            # Calcule les projections actuelles dans l'espace whitené
            projected = whitened @ weights.T
            # Applique la non-linéarité tanh pour la maximisation de la non-gaussianité
            non_linear = np.tanh(projected)
            # Calcule la dérivée moyenne pour la mise à jour
            non_linear_derivative = 1.0 - non_linear**2
            # Met à jour les poids via l'équation FastICA symétrique
            weights = (non_linear.T @ whitened) / n_samples - np.diag(
                non_linear_derivative.mean(axis=0)
            ) @ weights
            # Orthogonalise les nouveaux poids pour garantir l'indépendance
            weights = self._sym_decorrelation(weights)
            # Calcule la variation pour le critère de convergence
            deltas = np.abs(np.abs(np.diag(weights @ weights_old.T)) - 1.0)
            # Marque la convergence si la variation est sous la tolérance
            if np.max(deltas) < self.ica_tol:
                # Retourne les poids et l'état de convergence
                return weights, iteration + 1, True
        # Retourne les poids même si la convergence n'est pas atteinte
        return weights, self.ica_max_iter, False

    # Calcule le résidu de reconstruction ICA
    def _ica_residual(
        self,
        centered: np.ndarray,
        whitened: np.ndarray,
        dewhitening: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Calcule le résidu relatif entre original et reconstruction."""

        # Reconstruit les sources ICA pour valider la qualité
        sources = whitened @ weights.T
        # Reconstruit l'espace whitené depuis les sources
        whitened_recon = sources @ np.linalg.pinv(weights).T
        # Reconstruit l'espace original centré via le dewhitening
        centered_recon = whitened_recon @ dewhitening.T
        # Retourne la norme relative du résidu
        return self._relative_residual(centered, centered_recon)

    # Calcule la matrice de whitening pour l'ICA
    def _ica_whitening(
        self, centered: np.ndarray, n_components: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit le whitening ICA via la covariance centrée."""

        # Calcule la covariance échantillon pour le whitening
        covariance = self._regularized_covariance(centered)
        # Extrait le spectre propre de la covariance
        eigvals, eigvecs = np.linalg.eigh(covariance)
        # Trie les valeurs propres en ordre décroissant
        order = np.argsort(eigvals)[::-1]
        # Réordonne les vecteurs propres associés
        eigvecs = eigvecs[:, order]
        # Réordonne les valeurs propres pour cohérence
        eigvals = eigvals[order]
        # Tronque aux composantes demandées
        eigvals = eigvals[:n_components]
        # Tronque les vecteurs associés
        eigvecs = eigvecs[:, :n_components]
        # Évite les divisions par zéro sur les valeurs propres faibles
        eigvals = np.clip(eigvals, np.finfo(float).eps, None)
        # Calcule la matrice de whitening
        whitening = eigvecs / np.sqrt(eigvals)
        # Calcule la matrice de dewhitening pour la reconstruction
        dewhitening = eigvecs * np.sqrt(eigvals)
        # Retourne whitening, dewhitening et le spectre conservé
        return whitening, dewhitening, eigvals

    # Applique une orthogonalisation symétrique pour l'ICA
    def _sym_decorrelation(self, matrix: np.ndarray) -> np.ndarray:
        """Orthonormalise les poids pour l'ICA symétrique."""

        # Calcule la matrice de corrélation des poids
        correlation = matrix @ matrix.T
        # Décompose la corrélation pour obtenir la racine inverse
        eigvals, eigvecs = np.linalg.eigh(correlation)
        # Protège contre des valeurs propres négatives numériques
        eigvals = np.clip(eigvals, np.finfo(float).eps, None)
        # Construit la racine inverse de la corrélation
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        # Applique l'orthogonalisation symétrique
        return np.asarray(inv_sqrt @ matrix)

    # Calcule un résidu relatif pour mesurer la reconstruction
    def _relative_residual(self, original: np.ndarray, recon: np.ndarray) -> float:
        """Calcule ||X - X̂|| / ||X|| pour l'ICA."""

        # Calcule la norme du signal original
        original_norm = np.linalg.norm(original)
        # Garantit une échelle non nulle pour éviter une division par zéro
        if original_norm == 0:
            # Retourne un résidu nul pour un signal nul
            return 0.0
        # Calcule la norme du résidu
        residual_norm = np.linalg.norm(original - recon)
        # Retourne le résidu relatif pour la comparaison au seuil
        return float(residual_norm / original_norm)

    # Applique l'apprentissage CSP sur des essais EEG 3D
    def _fit_csp(self, X: np.ndarray, y: np.ndarray | None) -> None:
        """Apprend la projection CSP via un problème généralisé.

        Référence: Koles et al. (1990) Spatial filtering.
        """

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

"""Construction des pipelines de traitement EEG pour TPV."""

# Garantit l'accès aux types attendus par les signatures publiques
from __future__ import annotations

# Offre un conteneur immuable pour configurer la pipeline
from dataclasses import dataclass

# Maintient la compatibilité avec les types génériques scikit-learn
from typing import Iterable, List, Tuple

# Garantit la persistance pickle via le protocole scikit-learn
from joblib import dump, load

# Fournit les interfaces typées communes aux transformateurs scikit-learn
from sklearn.base import TransformerMixin

# Fournit les classifieurs linéaires et à marge large
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Fournit les pipelines séquentiels pour chaîner les transformateurs
from sklearn.pipeline import Pipeline

# Offre des scalers robustes et standards pour stabiliser les features
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

# Importe le classifieur léger basé sur les centroïdes
from tpv.classifier import CentroidClassifier

# Récupère les transformeurs de réduction de dimension
from tpv.dimensionality import CSP, TPVDimReducer

# Récupère l'extracteur de features de puissance bande
from tpv.features import ExtractFeatures


# Centralise la configuration du pipeline pour limiter les paramètres
@dataclass
class PipelineConfig:
    """Configuration complète pour assembler le pipeline TPV."""

    # Fréquence d'échantillonnage utilisée par l'extracteur de features
    sfreq: float
    # Stratégie de features pour harmoniser extraction et prédiction
    feature_strategy: str = "fft"
    # Active la normalisation pour stabiliser les entrées du classifieur
    normalize_features: bool = True
    # Méthode de réduction de dimension pour compacter les features
    dim_method: str = "pca"
    # Nombre de composantes conservées par le réducteur
    n_components: int | None = None
    # Classifieur final choisi par l'utilisateur
    classifier: str = "lda"
    # Scaler optionnel appliqué après l'extraction des features
    scaler: str | None = None
    # Fixe la régularisation CSP pour stabiliser les covariances
    csp_regularization: float = 0.1


# Fixe le nombre d'itérations de la régression logistique pour la stabilité
LOGISTIC_MAX_ITER = 1000
# Privilégie le solver SVD pour éviter les covariances instables en petit n
LDA_SOLVER = "svd"
# Désactive le shrinkage pour rester compatible avec le solver SVD
LDA_SHRINKAGE = None
# Fixe un nombre de composantes par défaut pour CSP en mode Welch
DEFAULT_WELCH_CSP_COMPONENTS = 4


# Construit une pipeline complète incluant préprocessing, features et classification
def build_pipeline(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Indique si CSP ou CSSP est utilisé pour adapter la pipeline
    uses_csp = config.dim_method in {"csp", "cssp"}
    # Identifie les stratégies de features nécessitant un signal projeté
    uses_signal_features = config.feature_strategy in {"welch", "wavelet"}
    # Prépare le nombre de composantes CSP effectif pour Welch+CSP
    csp_components = config.n_components
    # Applique un défaut seulement pour Welch afin de comparer les pipelines
    if uses_csp and uses_signal_features and csp_components is None:
        # Définit un nombre de composantes stable pour le benchmark Welch+CSP
        csp_components = DEFAULT_WELCH_CSP_COMPONENTS
    # Ajoute l'extracteur de features lorsqu'on n'utilise pas CSP/CSSP
    if not uses_csp:
        # Convertit les signaux bruts en vecteurs tabulaires
        steps.append(
            (
                "features",
                ExtractFeatures(
                    sfreq=config.sfreq,
                    feature_strategy=config.feature_strategy,
                    normalize=config.normalize_features,
                ),
            )
        )
        # Insère un scaler optionnel pour stabiliser la variance des features
        scaler_instance = _build_scaler(config.scaler)
        # Ajoute le scaler uniquement lorsqu'il est explicitement demandé
        if scaler_instance is not None:
            # Sécurise la position du scaler juste après les features tabulaires
            steps.append(("scaler", scaler_instance))
    # Ajoute CSP/CSSP en amont si la réduction spatiale est demandée
    if uses_csp:
        # Choisit la sortie CSP selon la présence de features spectrales
        return_log_variance = not uses_signal_features
        # Ajoute le bloc CSP/CSSP pour filtrer les signaux EEG
        steps.append(
            (
                "spatial_filters",
                CSP(
                    n_components=csp_components,
                    regularization=config.csp_regularization,
                    method=config.dim_method,
                    return_log_variance=return_log_variance,
                ),
            )
        )
        # Ajoute l'extracteur de features après CSP en mode Welch/Wavelet
        if uses_signal_features:
            # Convertit les signaux projetés en vecteurs tabulaires
            steps.append(
                (
                    "features",
                    ExtractFeatures(
                        sfreq=config.sfreq,
                        feature_strategy=config.feature_strategy,
                        normalize=config.normalize_features,
                    ),
                )
            )
            # Insère un scaler optionnel pour stabiliser la variance des features
            scaler_instance = _build_scaler(config.scaler)
            # Ajoute le scaler uniquement lorsqu'il est explicitement demandé
            if scaler_instance is not None:
                # Sécurise la position du scaler après les features projetées
                steps.append(("scaler", scaler_instance))
    else:
        # Ajoute la réduction de dimension pour compacter les représentations
        steps.append(
            (
                "dimensionality",
                TPVDimReducer(
                    method=config.dim_method,
                    n_components=config.n_components,
                    regularization=config.csp_regularization,
                ),
            )
        )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline dédiée aux recherches d'hyperparamètres
def build_search_pipeline(config: PipelineConfig) -> Pipeline:
    """Assemble une pipeline avec des étapes paramétrables pour GridSearch."""

    # Signale l'usage de CSP/CSSP pour adapter les étapes de pipeline
    uses_csp = config.dim_method in {"csp", "cssp"}
    # Identifie les stratégies nécessitant un signal projeté
    uses_signal_features = config.feature_strategy in {"welch", "wavelet"}
    # Prépare les étapes fixes de la pipeline
    if not uses_csp:
        # Construit une pipeline classique avec extracteur et scaler configurable
        steps: List[Tuple[str, object]] = [
            (
                "features",
                ExtractFeatures(
                    sfreq=config.sfreq,
                    feature_strategy=config.feature_strategy,
                    normalize=config.normalize_features,
                ),
            ),
            # Utilise passthrough pour autoriser la sélection de scaler en grid search
            ("scaler", "passthrough"),
            (
                "dimensionality",
                TPVDimReducer(
                    method=config.dim_method,
                    n_components=config.n_components,
                    regularization=config.csp_regularization,
                ),
            ),
            ("classifier", _build_classifier(config.classifier)),
        ]
    else:
        # Fixe un nombre de composantes par défaut pour Welch+CSP en recherche
        csp_components = config.n_components
        # Applique un défaut stable pour Welch afin d'assurer la comparaison
        if uses_signal_features and csp_components is None:
            # Réutilise le même défaut que la pipeline standard
            csp_components = DEFAULT_WELCH_CSP_COMPONENTS
        # Choisit la sortie CSP selon la présence de features spectrales
        return_log_variance = not uses_signal_features
        # Construit la pipeline CSP/CSSP avec éventuelles features
        steps = [
            (
                "spatial_filters",
                CSP(
                    n_components=csp_components,
                    regularization=config.csp_regularization,
                    method=config.dim_method,
                    return_log_variance=return_log_variance,
                ),
            ),
        ]
        # Ajoute les features spectrales après CSP pour Welch/Wavelet
        if uses_signal_features:
            # Ajoute l'extracteur de features pour GridSearch
            steps.append(
                (
                    "features",
                    ExtractFeatures(
                        sfreq=config.sfreq,
                        feature_strategy=config.feature_strategy,
                        normalize=config.normalize_features,
                    ),
                )
            )
            # Permet le scaler en passthrough pour la grid search
            steps.append(("scaler", "passthrough"))
        # Ajoute le classifieur en fin de pipeline
        steps.append(("classifier", _build_classifier(config.classifier)))
    # Retourne la pipeline prête pour GridSearchCV
    return Pipeline(steps)


# Sélectionne le scaler adapté selon la configuration utilisateur
def _build_scaler(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = option.lower()
    # Mappe la demande vers le scaler standard pour une normalisation z-score
    if normalized == "standard":
        # Fournit un StandardScaler sans centrage par défaut
        return StandardScaler()
    # Mappe la demande vers le scaler robuste pour limiter l'influence des outliers
    if normalized == "robust":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def _build_classifier(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA en mode SVD pour éviter les covariances instables
        return LinearDiscriminantAnalysis(
            solver=LDA_SOLVER,
            shrinkage=LDA_SHRINKAGE,
        )
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Augmente max_iter pour limiter les warnings de convergence liblinear
        return LinearSVC(max_iter=5000)
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sérialise un pipeline entraîné pour usage ultérieur
def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(pipeline, path)


# Restaure un pipeline sauvegardé depuis le disque
def load_pipeline(path: str) -> Pipeline:
    """Charge un pipeline précédemment sauvegardé."""

    # Utilise joblib.load pour reconstruire le pipeline complet
    return load(path)

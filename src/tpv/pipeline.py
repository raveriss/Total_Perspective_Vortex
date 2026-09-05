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
    # Transporte la configuration spécifique aux stratégies de features
    feature_strategy_config: dict[str, object] | None = None
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
# Expose les choix CLI depuis une seule source de vérité.
CLASSIFIER_CHOICES = ("lda", "logistic", "svm", "centroid")
SCALER_CHOICES = ("standard", "robust", "none")
FEATURE_STRATEGIES = ("fft", "welch", "wavelet")
DIMENSIONALITY_METHODS = ("pca", "csp", "cssp", "svd")
SPATIAL_METHODS = ("csp", "cssp")
SIGNAL_FEATURE_STRATEGIES = ("welch", "wavelet")


# Signale un ordre spatial puis spectral implicite sans modifier la configuration
def warn_if_spectral_features_follow_spatial_filter(
    feature_strategy: str,
    dim_method: str,
    dim_method_explicit: bool,
) -> None:
    """Explique l'ordre CSP/CSSP → Welch/Wavelet lorsqu'il est implicite."""

    # Une stratégie temporelle ou FFT ne nécessite pas ce diagnostic d'ordre
    if feature_strategy not in SIGNAL_FEATURE_STRATEGIES:
        # Le helper reste sans effet lorsque les features ne sont pas spectrales
        return
    # PCA et SVD n'appliquent pas le filtre spatial concerné par l'avertissement
    if dim_method not in SPATIAL_METHODS:
        # Le helper ne doit pas commenter une combinaison sans CSP ou CSSP
        return
    # Un choix explicite signifie que l'utilisateur connaît déjà cet enchaînement
    if dim_method_explicit:
        # Le mode explicite reste silencieux pour préserver les sorties CLI
        return
    # Ce message historique rend visible l'ordre réel sans changer le pipeline
    print(
        # La formulation reste stable pour les tests et la démonstration existants
        "INFO: dim_method='csp/cssp' appliqué avant l'extraction des features."
    )


# Construit l'extracteur spectral commun aux deux variantes de pipeline.
def _build_feature_extractor(config: PipelineConfig) -> ExtractFeatures:
    """Retourne l'extracteur configuré sans dupliquer ses paramètres."""

    return ExtractFeatures(
        sfreq=config.sfreq,
        feature_strategy=config.feature_strategy,
        normalize=config.normalize_features,
        strategy_config=config.feature_strategy_config,
    )


# Construit le filtre spatial commun aux pipelines standard et de recherche.
def _build_spatial_filter(config: PipelineConfig, uses_signal_features: bool) -> CSP:
    """Retourne CSP/CSSP avec exactement les mêmes valeurs par défaut."""

    csp_components = config.n_components
    if uses_signal_features and csp_components is None:
        csp_components = DEFAULT_WELCH_CSP_COMPONENTS
    return CSP(
        n_components=csp_components,
        regularization=config.csp_regularization,
        method=config.dim_method,
        return_log_variance=not uses_signal_features,
    )


# Ajoute ensemble l'extraction tabulaire et son scaler optionnel.
def _append_feature_steps(
    steps: List[Tuple[str, object]], config: PipelineConfig
) -> None:
    """Ajoute les étapes features puis scaler en conservant leur ordre."""

    steps.append(("features", _build_feature_extractor(config)))
    scaler = _build_scaler(config.scaler)
    if scaler is not None:
        steps.append(("scaler", scaler))


# Construit une pipeline complète incluant préprocessing, features et classification
def build_pipeline(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Indique si CSP ou CSSP est utilisé pour adapter la pipeline
    uses_csp = config.dim_method in SPATIAL_METHODS
    # Identifie les stratégies de features nécessitant un signal projeté
    uses_signal_features = config.feature_strategy in SIGNAL_FEATURE_STRATEGIES
    # Ajoute l'extracteur de features lorsqu'on n'utilise pas CSP/CSSP
    if not uses_csp:
        # Convertit le signal en features puis applique le scaler demandé.
        _append_feature_steps(steps, config)
    # Ajoute CSP/CSSP en amont si la réduction spatiale est demandée
    if uses_csp:
        # Ajoute le bloc CSP/CSSP pour filtrer les signaux EEG
        steps.append(
            (
                "spatial_filters",
                _build_spatial_filter(config, uses_signal_features),
            )
        )
        # Ajoute l'extracteur de features après CSP en mode Welch/Wavelet
        if uses_signal_features:
            # Convertit le signal projeté puis applique le même contrat de scaling.
            _append_feature_steps(steps, config)
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
    uses_csp = config.dim_method in SPATIAL_METHODS
    # Identifie les stratégies nécessitant un signal projeté
    uses_signal_features = config.feature_strategy in SIGNAL_FEATURE_STRATEGIES
    # Prépare les étapes fixes de la pipeline
    if not uses_csp:
        # Construit une pipeline classique avec extracteur et scaler configurable
        steps: List[Tuple[str, object]] = [
            (
                "features",
                _build_feature_extractor(config),
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
        # Construit la pipeline CSP/CSSP avec éventuelles features
        steps = [
            (
                "spatial_filters",
                _build_spatial_filter(config, uses_signal_features),
            ),
        ]
        # Ajoute les features spectrales après CSP pour Welch/Wavelet
        if uses_signal_features:
            # Ajoute l'extracteur de features pour GridSearch
            steps.append(
                (
                    "features",
                    _build_feature_extractor(config),
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

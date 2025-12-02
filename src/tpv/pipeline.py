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

# Récupère le réducteur de dimension CSP ou PCA
from tpv.dimensionality import TPVDimReducer

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


# Construit une pipeline complète incluant préprocessing, features et classification
def build_pipeline(
    preprocessors: Iterable[Tuple[str, object]] | None, config: PipelineConfig
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
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
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(
                method=config.dim_method, n_components=config.n_components
            ),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
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
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=1000)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', or 'svm'")


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

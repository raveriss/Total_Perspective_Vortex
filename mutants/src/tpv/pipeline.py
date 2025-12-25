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

# Récupère le réducteur de dimension CSP ou PCA
from tpv.dimensionality import TPVDimReducer

# Récupère l'extracteur de features de puissance bande
from tpv.features import ExtractFeatures
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg is not None:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


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


# Fixe le nombre d'itérations de la régression logistique pour la stabilité
LOGISTIC_MAX_ITER = 1000


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_orig(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_1(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_2(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(None)
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_3(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors and [])
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_4(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
    steps.append(
        None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_5(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
    steps.append(
        (
            "XXfeaturesXX",
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_6(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
    steps.append(
        (
            "FEATURES",
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_7(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
    steps.append(
        (
            "features",
            ExtractFeatures(
                sfreq=None,
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_8(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
                feature_strategy=None,
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_9(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
                normalize=None,
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_10(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
) -> Pipeline:
    """Assemble un pipeline scikit-learn cohérent pour l'EEG."""

    # Prépare la liste des étapes en partant d'éventuels préprocesseurs
    steps: List[Tuple[str, object]] = list(preprocessors or [])
    # Ajoute l'extracteur de features pour convertir les signaux en tabulaire
    steps.append(
        (
            "features",
            ExtractFeatures(
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_11(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_12(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_13(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
    scaler_instance = None
    # Ajoute le scaler uniquement lorsqu'il est explicitement demandé
    if scaler_instance is not None:
        # Sécurise la position du scaler juste après les features tabulaires
        steps.append(("scaler", scaler_instance))
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_14(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
    scaler_instance = _build_scaler(None)
    # Ajoute le scaler uniquement lorsqu'il est explicitement demandé
    if scaler_instance is not None:
        # Sécurise la position du scaler juste après les features tabulaires
        steps.append(("scaler", scaler_instance))
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_15(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
    if scaler_instance is None:
        # Sécurise la position du scaler juste après les features tabulaires
        steps.append(("scaler", scaler_instance))
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_16(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
        steps.append(None)
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_17(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
        steps.append(("XXscalerXX", scaler_instance))
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_18(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
        steps.append(("SCALER", scaler_instance))
    # Ajoute la réduction de dimension pour compacter les représentations
    steps.append(
        (
            "dimensionality",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_19(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
        None
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_20(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            "XXdimensionalityXX",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_21(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            "DIMENSIONALITY",
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_22(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=None, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_23(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=None),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_24(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_25(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, ),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_26(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = None
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_27(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(None)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_28(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(None)
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_29(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("XXclassifierXX", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_30(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("CLASSIFIER", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(steps)


# Construit une pipeline complète incluant préprocessing, features et classification
def x_build_pipeline__mutmut_31(
    config: PipelineConfig, preprocessors: Iterable[Tuple[str, object]] | None = None
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
            TPVDimReducer(method=config.dim_method, n_components=config.n_components),
        )
    )
    # Construit le classifieur final selon la stratégie choisie
    classifier_instance = _build_classifier(config.classifier)
    # Ajoute le classifieur au pipeline pour la prédiction finale
    steps.append(("classifier", classifier_instance))
    # Assemble et retourne la pipeline scikit-learn séquentielle
    return Pipeline(None)

x_build_pipeline__mutmut_mutants : ClassVar[MutantDict] = {
'x_build_pipeline__mutmut_1': x_build_pipeline__mutmut_1, 
    'x_build_pipeline__mutmut_2': x_build_pipeline__mutmut_2, 
    'x_build_pipeline__mutmut_3': x_build_pipeline__mutmut_3, 
    'x_build_pipeline__mutmut_4': x_build_pipeline__mutmut_4, 
    'x_build_pipeline__mutmut_5': x_build_pipeline__mutmut_5, 
    'x_build_pipeline__mutmut_6': x_build_pipeline__mutmut_6, 
    'x_build_pipeline__mutmut_7': x_build_pipeline__mutmut_7, 
    'x_build_pipeline__mutmut_8': x_build_pipeline__mutmut_8, 
    'x_build_pipeline__mutmut_9': x_build_pipeline__mutmut_9, 
    'x_build_pipeline__mutmut_10': x_build_pipeline__mutmut_10, 
    'x_build_pipeline__mutmut_11': x_build_pipeline__mutmut_11, 
    'x_build_pipeline__mutmut_12': x_build_pipeline__mutmut_12, 
    'x_build_pipeline__mutmut_13': x_build_pipeline__mutmut_13, 
    'x_build_pipeline__mutmut_14': x_build_pipeline__mutmut_14, 
    'x_build_pipeline__mutmut_15': x_build_pipeline__mutmut_15, 
    'x_build_pipeline__mutmut_16': x_build_pipeline__mutmut_16, 
    'x_build_pipeline__mutmut_17': x_build_pipeline__mutmut_17, 
    'x_build_pipeline__mutmut_18': x_build_pipeline__mutmut_18, 
    'x_build_pipeline__mutmut_19': x_build_pipeline__mutmut_19, 
    'x_build_pipeline__mutmut_20': x_build_pipeline__mutmut_20, 
    'x_build_pipeline__mutmut_21': x_build_pipeline__mutmut_21, 
    'x_build_pipeline__mutmut_22': x_build_pipeline__mutmut_22, 
    'x_build_pipeline__mutmut_23': x_build_pipeline__mutmut_23, 
    'x_build_pipeline__mutmut_24': x_build_pipeline__mutmut_24, 
    'x_build_pipeline__mutmut_25': x_build_pipeline__mutmut_25, 
    'x_build_pipeline__mutmut_26': x_build_pipeline__mutmut_26, 
    'x_build_pipeline__mutmut_27': x_build_pipeline__mutmut_27, 
    'x_build_pipeline__mutmut_28': x_build_pipeline__mutmut_28, 
    'x_build_pipeline__mutmut_29': x_build_pipeline__mutmut_29, 
    'x_build_pipeline__mutmut_30': x_build_pipeline__mutmut_30, 
    'x_build_pipeline__mutmut_31': x_build_pipeline__mutmut_31
}

def build_pipeline(*args, **kwargs):
    result = _mutmut_trampoline(x_build_pipeline__mutmut_orig, x_build_pipeline__mutmut_mutants, args, kwargs)
    return result 

build_pipeline.__signature__ = _mutmut_signature(x_build_pipeline__mutmut_orig)
x_build_pipeline__mutmut_orig.__name__ = 'x_build_pipeline'


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_orig(option: str | None) -> TransformerMixin | None:
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


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_1(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is not None:
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


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_2(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = None
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


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_3(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = option.upper()
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


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_4(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = option.lower()
    # Mappe la demande vers le scaler standard pour une normalisation z-score
    if normalized != "standard":
        # Fournit un StandardScaler sans centrage par défaut
        return StandardScaler()
    # Mappe la demande vers le scaler robuste pour limiter l'influence des outliers
    if normalized == "robust":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_5(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = option.lower()
    # Mappe la demande vers le scaler standard pour une normalisation z-score
    if normalized == "XXstandardXX":
        # Fournit un StandardScaler sans centrage par défaut
        return StandardScaler()
    # Mappe la demande vers le scaler robuste pour limiter l'influence des outliers
    if normalized == "robust":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_6(option: str | None) -> TransformerMixin | None:
    """Retourne l'instance de scaler correspondant au paramètre fourni."""

    # Ignore la construction lorsqu'aucun scaler n'est demandé
    if option is None:
        # Retourne None pour laisser le pipeline sans étape de scaling
        return None
    # Normalise le paramètre pour éviter les erreurs de casse
    normalized = option.lower()
    # Mappe la demande vers le scaler standard pour une normalisation z-score
    if normalized == "STANDARD":
        # Fournit un StandardScaler sans centrage par défaut
        return StandardScaler()
    # Mappe la demande vers le scaler robuste pour limiter l'influence des outliers
    if normalized == "robust":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_7(option: str | None) -> TransformerMixin | None:
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
    if normalized != "robust":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_8(option: str | None) -> TransformerMixin | None:
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
    if normalized == "XXrobustXX":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_9(option: str | None) -> TransformerMixin | None:
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
    if normalized == "ROBUST":
        # Fournit un RobustScaler adapté aux distributions asymétriques
        return RobustScaler()
    # Provoque une erreur claire en cas de paramètre non supporté
    raise ValueError("scaler must be 'standard', 'robust', or None")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_10(option: str | None) -> TransformerMixin | None:
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
    raise ValueError(None)


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_11(option: str | None) -> TransformerMixin | None:
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
    raise ValueError("XXscaler must be 'standard', 'robust', or NoneXX")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_12(option: str | None) -> TransformerMixin | None:
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
    raise ValueError("scaler must be 'standard', 'robust', or none")


# Sélectionne le scaler adapté selon la configuration utilisateur
def x__build_scaler__mutmut_13(option: str | None) -> TransformerMixin | None:
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
    raise ValueError("SCALER MUST BE 'STANDARD', 'ROBUST', OR NONE")

x__build_scaler__mutmut_mutants : ClassVar[MutantDict] = {
'x__build_scaler__mutmut_1': x__build_scaler__mutmut_1, 
    'x__build_scaler__mutmut_2': x__build_scaler__mutmut_2, 
    'x__build_scaler__mutmut_3': x__build_scaler__mutmut_3, 
    'x__build_scaler__mutmut_4': x__build_scaler__mutmut_4, 
    'x__build_scaler__mutmut_5': x__build_scaler__mutmut_5, 
    'x__build_scaler__mutmut_6': x__build_scaler__mutmut_6, 
    'x__build_scaler__mutmut_7': x__build_scaler__mutmut_7, 
    'x__build_scaler__mutmut_8': x__build_scaler__mutmut_8, 
    'x__build_scaler__mutmut_9': x__build_scaler__mutmut_9, 
    'x__build_scaler__mutmut_10': x__build_scaler__mutmut_10, 
    'x__build_scaler__mutmut_11': x__build_scaler__mutmut_11, 
    'x__build_scaler__mutmut_12': x__build_scaler__mutmut_12, 
    'x__build_scaler__mutmut_13': x__build_scaler__mutmut_13
}

def _build_scaler(*args, **kwargs):
    result = _mutmut_trampoline(x__build_scaler__mutmut_orig, x__build_scaler__mutmut_mutants, args, kwargs)
    return result 

_build_scaler.__signature__ = _mutmut_signature(x__build_scaler__mutmut_orig)
x__build_scaler__mutmut_orig.__name__ = 'x__build_scaler'


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_orig(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_1(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = None
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_2(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.upper()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_3(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized != "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_4(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "XXldaXX":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_5(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "LDA":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_6(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized != "logistic":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_7(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "XXlogisticXX":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_8(option: str) -> object:
    """Retourne un classifieur entraînable compatible scikit-learn."""

    # Normalise la valeur pour autoriser plusieurs casses utilisateur
    normalized = option.lower()
    # Retourne une analyse discriminante linéaire pour la simplicité
    if normalized == "lda":
        # Utilise LDA avec solution automatique de covariance
        return LinearDiscriminantAnalysis()
    # Retourne une régression logistique pour des décisions probabilistes
    if normalized == "LOGISTIC":
        # Configure la régularisation l2 avec solver lbfgs stable
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_9(option: str) -> object:
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
        return LogisticRegression(max_iter=None)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_10(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized != "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_11(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "XXsvmXX":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_12(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "SVM":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_13(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized != "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_14(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "XXcentroidXX":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_15(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "CENTROID":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("classifier must be 'lda', 'logistic', 'svm', or 'centroid'")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_16(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError(None)


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_17(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("XXclassifier must be 'lda', 'logistic', 'svm', or 'centroid'XX")


# Sélectionne le classifieur selon la chaîne fournie par l'utilisateur
def x__build_classifier__mutmut_18(option: str) -> object:
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
        return LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    # Retourne un SVM linéaire pour des marges maximales
    if normalized == "svm":
        # Utilise LinearSVC pour des données tabulaires haute dimension
        return LinearSVC()
    # Retourne le classifieur léger personnalisé pour des prototypes rapides
    if normalized == "centroid":
        # Utilise un classifieur basé sur les centroïdes pour limiter la variance
        return CentroidClassifier()
    # Provoque une erreur explicite lorsque le classifieur n'est pas reconnu
    raise ValueError("CLASSIFIER MUST BE 'LDA', 'LOGISTIC', 'SVM', OR 'CENTROID'")

x__build_classifier__mutmut_mutants : ClassVar[MutantDict] = {
'x__build_classifier__mutmut_1': x__build_classifier__mutmut_1, 
    'x__build_classifier__mutmut_2': x__build_classifier__mutmut_2, 
    'x__build_classifier__mutmut_3': x__build_classifier__mutmut_3, 
    'x__build_classifier__mutmut_4': x__build_classifier__mutmut_4, 
    'x__build_classifier__mutmut_5': x__build_classifier__mutmut_5, 
    'x__build_classifier__mutmut_6': x__build_classifier__mutmut_6, 
    'x__build_classifier__mutmut_7': x__build_classifier__mutmut_7, 
    'x__build_classifier__mutmut_8': x__build_classifier__mutmut_8, 
    'x__build_classifier__mutmut_9': x__build_classifier__mutmut_9, 
    'x__build_classifier__mutmut_10': x__build_classifier__mutmut_10, 
    'x__build_classifier__mutmut_11': x__build_classifier__mutmut_11, 
    'x__build_classifier__mutmut_12': x__build_classifier__mutmut_12, 
    'x__build_classifier__mutmut_13': x__build_classifier__mutmut_13, 
    'x__build_classifier__mutmut_14': x__build_classifier__mutmut_14, 
    'x__build_classifier__mutmut_15': x__build_classifier__mutmut_15, 
    'x__build_classifier__mutmut_16': x__build_classifier__mutmut_16, 
    'x__build_classifier__mutmut_17': x__build_classifier__mutmut_17, 
    'x__build_classifier__mutmut_18': x__build_classifier__mutmut_18
}

def _build_classifier(*args, **kwargs):
    result = _mutmut_trampoline(x__build_classifier__mutmut_orig, x__build_classifier__mutmut_mutants, args, kwargs)
    return result 

_build_classifier.__signature__ = _mutmut_signature(x__build_classifier__mutmut_orig)
x__build_classifier__mutmut_orig.__name__ = 'x__build_classifier'


# Sérialise un pipeline entraîné pour usage ultérieur
def x_save_pipeline__mutmut_orig(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(pipeline, path)


# Sérialise un pipeline entraîné pour usage ultérieur
def x_save_pipeline__mutmut_1(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(None, path)


# Sérialise un pipeline entraîné pour usage ultérieur
def x_save_pipeline__mutmut_2(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(pipeline, None)


# Sérialise un pipeline entraîné pour usage ultérieur
def x_save_pipeline__mutmut_3(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(path)


# Sérialise un pipeline entraîné pour usage ultérieur
def x_save_pipeline__mutmut_4(pipeline: Pipeline, path: str) -> None:
    """Sauvegarde le pipeline sur disque via joblib."""

    # Utilise joblib.dump pour persister l'objet complet
    dump(pipeline, )

x_save_pipeline__mutmut_mutants : ClassVar[MutantDict] = {
'x_save_pipeline__mutmut_1': x_save_pipeline__mutmut_1, 
    'x_save_pipeline__mutmut_2': x_save_pipeline__mutmut_2, 
    'x_save_pipeline__mutmut_3': x_save_pipeline__mutmut_3, 
    'x_save_pipeline__mutmut_4': x_save_pipeline__mutmut_4
}

def save_pipeline(*args, **kwargs):
    result = _mutmut_trampoline(x_save_pipeline__mutmut_orig, x_save_pipeline__mutmut_mutants, args, kwargs)
    return result 

save_pipeline.__signature__ = _mutmut_signature(x_save_pipeline__mutmut_orig)
x_save_pipeline__mutmut_orig.__name__ = 'x_save_pipeline'


# Restaure un pipeline sauvegardé depuis le disque
def x_load_pipeline__mutmut_orig(path: str) -> Pipeline:
    """Charge un pipeline précédemment sauvegardé."""

    # Utilise joblib.load pour reconstruire le pipeline complet
    return load(path)


# Restaure un pipeline sauvegardé depuis le disque
def x_load_pipeline__mutmut_1(path: str) -> Pipeline:
    """Charge un pipeline précédemment sauvegardé."""

    # Utilise joblib.load pour reconstruire le pipeline complet
    return load(None)

x_load_pipeline__mutmut_mutants : ClassVar[MutantDict] = {
'x_load_pipeline__mutmut_1': x_load_pipeline__mutmut_1
}

def load_pipeline(*args, **kwargs):
    result = _mutmut_trampoline(x_load_pipeline__mutmut_orig, x_load_pipeline__mutmut_mutants, args, kwargs)
    return result 

load_pipeline.__signature__ = _mutmut_signature(x_load_pipeline__mutmut_orig)
x_load_pipeline__mutmut_orig.__name__ = 'x_load_pipeline'

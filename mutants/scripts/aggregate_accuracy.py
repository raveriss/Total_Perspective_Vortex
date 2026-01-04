# Importe argparse pour fournir une CLI dédiée au reporting
import argparse

# Garantit l'accès aux chemins indépendants de la plateforme
from pathlib import Path

# Offre des statistiques numériques stables pour les agrégations
import numpy as np

# Réutilise l'évaluation existante pour relire les artefacts sauvegardés
from scripts import predict as predict_cli

# Définit le répertoire par défaut où chercher les jeux de données
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut où lire les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
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


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_orig() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_1() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = None
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_2() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=None,
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_3() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "XXAgrège les accuracies par run, sujet et global à partir des XX" "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_4() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_5() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "AGRÈGE LES ACCURACIES PAR RUN, SUJET ET GLOBAL À PARTIR DES " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_6() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "XXartefactsXX"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_7() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "ARTEFACTS"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_8() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        None,
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_9() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=None,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_10() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_11() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=None,
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_12() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_13() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_14() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_15() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_16() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "XX--data-dirXX",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_17() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--DATA-DIR",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_18() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="XXRépertoire racine contenant les matrices numpy utilisées pour le scoringXX",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_19() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_20() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="RÉPERTOIRE RACINE CONTENANT LES MATRICES NUMPY UTILISÉES POUR LE SCORING",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_21() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        None,
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_22() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=None,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_23() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_24() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=None,
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_25() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_26() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_27() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_28() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_29() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "XX--artifacts-dirXX",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_30() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--ARTIFACTS-DIR",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_31() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="XXRépertoire racine où sont stockés les modèles et matrices WXX",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_32() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="répertoire racine où sont stockés les modèles et matrices w",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer le reporting en CLI
def x_build_parser__mutmut_33() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="RÉPERTOIRE RACINE OÙ SONT STOCKÉS LES MODÈLES ET MATRICES W",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser

x_build_parser__mutmut_mutants : ClassVar[MutantDict] = {
'x_build_parser__mutmut_1': x_build_parser__mutmut_1, 
    'x_build_parser__mutmut_2': x_build_parser__mutmut_2, 
    'x_build_parser__mutmut_3': x_build_parser__mutmut_3, 
    'x_build_parser__mutmut_4': x_build_parser__mutmut_4, 
    'x_build_parser__mutmut_5': x_build_parser__mutmut_5, 
    'x_build_parser__mutmut_6': x_build_parser__mutmut_6, 
    'x_build_parser__mutmut_7': x_build_parser__mutmut_7, 
    'x_build_parser__mutmut_8': x_build_parser__mutmut_8, 
    'x_build_parser__mutmut_9': x_build_parser__mutmut_9, 
    'x_build_parser__mutmut_10': x_build_parser__mutmut_10, 
    'x_build_parser__mutmut_11': x_build_parser__mutmut_11, 
    'x_build_parser__mutmut_12': x_build_parser__mutmut_12, 
    'x_build_parser__mutmut_13': x_build_parser__mutmut_13, 
    'x_build_parser__mutmut_14': x_build_parser__mutmut_14, 
    'x_build_parser__mutmut_15': x_build_parser__mutmut_15, 
    'x_build_parser__mutmut_16': x_build_parser__mutmut_16, 
    'x_build_parser__mutmut_17': x_build_parser__mutmut_17, 
    'x_build_parser__mutmut_18': x_build_parser__mutmut_18, 
    'x_build_parser__mutmut_19': x_build_parser__mutmut_19, 
    'x_build_parser__mutmut_20': x_build_parser__mutmut_20, 
    'x_build_parser__mutmut_21': x_build_parser__mutmut_21, 
    'x_build_parser__mutmut_22': x_build_parser__mutmut_22, 
    'x_build_parser__mutmut_23': x_build_parser__mutmut_23, 
    'x_build_parser__mutmut_24': x_build_parser__mutmut_24, 
    'x_build_parser__mutmut_25': x_build_parser__mutmut_25, 
    'x_build_parser__mutmut_26': x_build_parser__mutmut_26, 
    'x_build_parser__mutmut_27': x_build_parser__mutmut_27, 
    'x_build_parser__mutmut_28': x_build_parser__mutmut_28, 
    'x_build_parser__mutmut_29': x_build_parser__mutmut_29, 
    'x_build_parser__mutmut_30': x_build_parser__mutmut_30, 
    'x_build_parser__mutmut_31': x_build_parser__mutmut_31, 
    'x_build_parser__mutmut_32': x_build_parser__mutmut_32, 
    'x_build_parser__mutmut_33': x_build_parser__mutmut_33
}

def build_parser(*args, **kwargs):
    result = _mutmut_trampoline(x_build_parser__mutmut_orig, x_build_parser__mutmut_mutants, args, kwargs)
    return result 

build_parser.__signature__ = _mutmut_signature(x_build_parser__mutmut_orig)
x_build_parser__mutmut_orig.__name__ = 'x_build_parser'


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_orig(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_1(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = None
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_2(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_3(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(None):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_4(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_5(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            break
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_6(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(None):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_7(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = None
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_8(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir * "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_9(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "XXmodel.joblibXX"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_10(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "MODEL.JOBLIB"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Inventorie les runs disponibles en inspectant les artefacts présents
def x__discover_runs__mutmut_11(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append(None)
    # Retourne l'ensemble des couples détectés
    return runs

x__discover_runs__mutmut_mutants : ClassVar[MutantDict] = {
'x__discover_runs__mutmut_1': x__discover_runs__mutmut_1, 
    'x__discover_runs__mutmut_2': x__discover_runs__mutmut_2, 
    'x__discover_runs__mutmut_3': x__discover_runs__mutmut_3, 
    'x__discover_runs__mutmut_4': x__discover_runs__mutmut_4, 
    'x__discover_runs__mutmut_5': x__discover_runs__mutmut_5, 
    'x__discover_runs__mutmut_6': x__discover_runs__mutmut_6, 
    'x__discover_runs__mutmut_7': x__discover_runs__mutmut_7, 
    'x__discover_runs__mutmut_8': x__discover_runs__mutmut_8, 
    'x__discover_runs__mutmut_9': x__discover_runs__mutmut_9, 
    'x__discover_runs__mutmut_10': x__discover_runs__mutmut_10, 
    'x__discover_runs__mutmut_11': x__discover_runs__mutmut_11
}

def _discover_runs(*args, **kwargs):
    result = _mutmut_trampoline(x__discover_runs__mutmut_orig, x__discover_runs__mutmut_mutants, args, kwargs)
    return result 

_discover_runs.__signature__ = _mutmut_signature(x__discover_runs__mutmut_orig)
x__discover_runs__mutmut_orig.__name__ = 'x__discover_runs'


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_orig(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_1(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = None
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_2(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(None)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_3(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = None
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_4(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = None
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_5(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = None
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_6(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = None
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_7(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(None, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_8(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, None, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_9(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, None, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_10(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, None)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_11(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_12(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_13(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_14(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, )
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_15(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = None
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_16(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = None
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_17(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["XXaccuracyXX"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_18(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["ACCURACY"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_19(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_20(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = None
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_21(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(None)
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_22(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["XXaccuracyXX"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_23(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["ACCURACY"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_24(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(None)
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_25(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["XXaccuracyXX"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_26(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["ACCURACY"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_27(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = None
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_28(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(None) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_29(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(None)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_30(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = None
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_31(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(None) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_32(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(None)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_33(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 1.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_34(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"XXby_runXX": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_35(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"BY_RUN": by_run, "by_subject": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_36(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "XXby_subjectXX": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_37(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "BY_SUBJECT": by_subject, "global": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_38(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "XXglobalXX": global_accuracy}


# Calcule les accuracies en réutilisant les artefacts existants
def x_aggregate_accuracies__mutmut_39(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "GLOBAL": global_accuracy}

x_aggregate_accuracies__mutmut_mutants : ClassVar[MutantDict] = {
'x_aggregate_accuracies__mutmut_1': x_aggregate_accuracies__mutmut_1, 
    'x_aggregate_accuracies__mutmut_2': x_aggregate_accuracies__mutmut_2, 
    'x_aggregate_accuracies__mutmut_3': x_aggregate_accuracies__mutmut_3, 
    'x_aggregate_accuracies__mutmut_4': x_aggregate_accuracies__mutmut_4, 
    'x_aggregate_accuracies__mutmut_5': x_aggregate_accuracies__mutmut_5, 
    'x_aggregate_accuracies__mutmut_6': x_aggregate_accuracies__mutmut_6, 
    'x_aggregate_accuracies__mutmut_7': x_aggregate_accuracies__mutmut_7, 
    'x_aggregate_accuracies__mutmut_8': x_aggregate_accuracies__mutmut_8, 
    'x_aggregate_accuracies__mutmut_9': x_aggregate_accuracies__mutmut_9, 
    'x_aggregate_accuracies__mutmut_10': x_aggregate_accuracies__mutmut_10, 
    'x_aggregate_accuracies__mutmut_11': x_aggregate_accuracies__mutmut_11, 
    'x_aggregate_accuracies__mutmut_12': x_aggregate_accuracies__mutmut_12, 
    'x_aggregate_accuracies__mutmut_13': x_aggregate_accuracies__mutmut_13, 
    'x_aggregate_accuracies__mutmut_14': x_aggregate_accuracies__mutmut_14, 
    'x_aggregate_accuracies__mutmut_15': x_aggregate_accuracies__mutmut_15, 
    'x_aggregate_accuracies__mutmut_16': x_aggregate_accuracies__mutmut_16, 
    'x_aggregate_accuracies__mutmut_17': x_aggregate_accuracies__mutmut_17, 
    'x_aggregate_accuracies__mutmut_18': x_aggregate_accuracies__mutmut_18, 
    'x_aggregate_accuracies__mutmut_19': x_aggregate_accuracies__mutmut_19, 
    'x_aggregate_accuracies__mutmut_20': x_aggregate_accuracies__mutmut_20, 
    'x_aggregate_accuracies__mutmut_21': x_aggregate_accuracies__mutmut_21, 
    'x_aggregate_accuracies__mutmut_22': x_aggregate_accuracies__mutmut_22, 
    'x_aggregate_accuracies__mutmut_23': x_aggregate_accuracies__mutmut_23, 
    'x_aggregate_accuracies__mutmut_24': x_aggregate_accuracies__mutmut_24, 
    'x_aggregate_accuracies__mutmut_25': x_aggregate_accuracies__mutmut_25, 
    'x_aggregate_accuracies__mutmut_26': x_aggregate_accuracies__mutmut_26, 
    'x_aggregate_accuracies__mutmut_27': x_aggregate_accuracies__mutmut_27, 
    'x_aggregate_accuracies__mutmut_28': x_aggregate_accuracies__mutmut_28, 
    'x_aggregate_accuracies__mutmut_29': x_aggregate_accuracies__mutmut_29, 
    'x_aggregate_accuracies__mutmut_30': x_aggregate_accuracies__mutmut_30, 
    'x_aggregate_accuracies__mutmut_31': x_aggregate_accuracies__mutmut_31, 
    'x_aggregate_accuracies__mutmut_32': x_aggregate_accuracies__mutmut_32, 
    'x_aggregate_accuracies__mutmut_33': x_aggregate_accuracies__mutmut_33, 
    'x_aggregate_accuracies__mutmut_34': x_aggregate_accuracies__mutmut_34, 
    'x_aggregate_accuracies__mutmut_35': x_aggregate_accuracies__mutmut_35, 
    'x_aggregate_accuracies__mutmut_36': x_aggregate_accuracies__mutmut_36, 
    'x_aggregate_accuracies__mutmut_37': x_aggregate_accuracies__mutmut_37, 
    'x_aggregate_accuracies__mutmut_38': x_aggregate_accuracies__mutmut_38, 
    'x_aggregate_accuracies__mutmut_39': x_aggregate_accuracies__mutmut_39
}

def aggregate_accuracies(*args, **kwargs):
    result = _mutmut_trampoline(x_aggregate_accuracies__mutmut_orig, x_aggregate_accuracies__mutmut_mutants, args, kwargs)
    return result 

aggregate_accuracies.__signature__ = _mutmut_signature(x_aggregate_accuracies__mutmut_orig)
x_aggregate_accuracies__mutmut_orig.__name__ = 'x_aggregate_accuracies'


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_orig(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_1(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = None
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_2(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["XXRun\tSubject\tAccuracyXX"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_3(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["run\tsubject\taccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_4(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["RUN\tSUBJECT\tACCURACY"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_5(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(None):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_6(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["XXby_runXX"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_7(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["BY_RUN"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_8(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = None
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_9(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split(None)
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_10(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("XX/XX")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_11(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(None)
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_12(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['XXby_runXX'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_13(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['BY_RUN'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_14(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append(None)
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_15(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("XXSubject\tMean AccuracyXX")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_16(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("subject\tmean accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_17(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("SUBJECT\tMEAN ACCURACY")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_18(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(None):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_19(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["XXby_subjectXX"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_20(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["BY_SUBJECT"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_21(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(None)
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_22(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['XXby_subjectXX'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_23(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['BY_SUBJECT'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_24(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(None)
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_25(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['XXglobalXX']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_26(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['GLOBAL']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_27(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(None)


# Formate un tableau texte des accuracies calculées
def x_format_accuracy_table__mutmut_28(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "XX\nXX".join(lines)

x_format_accuracy_table__mutmut_mutants : ClassVar[MutantDict] = {
'x_format_accuracy_table__mutmut_1': x_format_accuracy_table__mutmut_1, 
    'x_format_accuracy_table__mutmut_2': x_format_accuracy_table__mutmut_2, 
    'x_format_accuracy_table__mutmut_3': x_format_accuracy_table__mutmut_3, 
    'x_format_accuracy_table__mutmut_4': x_format_accuracy_table__mutmut_4, 
    'x_format_accuracy_table__mutmut_5': x_format_accuracy_table__mutmut_5, 
    'x_format_accuracy_table__mutmut_6': x_format_accuracy_table__mutmut_6, 
    'x_format_accuracy_table__mutmut_7': x_format_accuracy_table__mutmut_7, 
    'x_format_accuracy_table__mutmut_8': x_format_accuracy_table__mutmut_8, 
    'x_format_accuracy_table__mutmut_9': x_format_accuracy_table__mutmut_9, 
    'x_format_accuracy_table__mutmut_10': x_format_accuracy_table__mutmut_10, 
    'x_format_accuracy_table__mutmut_11': x_format_accuracy_table__mutmut_11, 
    'x_format_accuracy_table__mutmut_12': x_format_accuracy_table__mutmut_12, 
    'x_format_accuracy_table__mutmut_13': x_format_accuracy_table__mutmut_13, 
    'x_format_accuracy_table__mutmut_14': x_format_accuracy_table__mutmut_14, 
    'x_format_accuracy_table__mutmut_15': x_format_accuracy_table__mutmut_15, 
    'x_format_accuracy_table__mutmut_16': x_format_accuracy_table__mutmut_16, 
    'x_format_accuracy_table__mutmut_17': x_format_accuracy_table__mutmut_17, 
    'x_format_accuracy_table__mutmut_18': x_format_accuracy_table__mutmut_18, 
    'x_format_accuracy_table__mutmut_19': x_format_accuracy_table__mutmut_19, 
    'x_format_accuracy_table__mutmut_20': x_format_accuracy_table__mutmut_20, 
    'x_format_accuracy_table__mutmut_21': x_format_accuracy_table__mutmut_21, 
    'x_format_accuracy_table__mutmut_22': x_format_accuracy_table__mutmut_22, 
    'x_format_accuracy_table__mutmut_23': x_format_accuracy_table__mutmut_23, 
    'x_format_accuracy_table__mutmut_24': x_format_accuracy_table__mutmut_24, 
    'x_format_accuracy_table__mutmut_25': x_format_accuracy_table__mutmut_25, 
    'x_format_accuracy_table__mutmut_26': x_format_accuracy_table__mutmut_26, 
    'x_format_accuracy_table__mutmut_27': x_format_accuracy_table__mutmut_27, 
    'x_format_accuracy_table__mutmut_28': x_format_accuracy_table__mutmut_28
}

def format_accuracy_table(*args, **kwargs):
    result = _mutmut_trampoline(x_format_accuracy_table__mutmut_orig, x_format_accuracy_table__mutmut_mutants, args, kwargs)
    return result 

format_accuracy_table.__signature__ = _mutmut_signature(x_format_accuracy_table__mutmut_orig)
x_format_accuracy_table__mutmut_orig.__name__ = 'x_format_accuracy_table'


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_orig(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_1(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = None
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_2(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = None
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_3(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(None)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_4(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = None
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_5(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(None, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_6(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, None)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_7(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_8(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, )
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_9(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = None
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_10(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(None)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_11(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(None)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_12(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 1

x_main__mutmut_mutants : ClassVar[MutantDict] = {
'x_main__mutmut_1': x_main__mutmut_1, 
    'x_main__mutmut_2': x_main__mutmut_2, 
    'x_main__mutmut_3': x_main__mutmut_3, 
    'x_main__mutmut_4': x_main__mutmut_4, 
    'x_main__mutmut_5': x_main__mutmut_5, 
    'x_main__mutmut_6': x_main__mutmut_6, 
    'x_main__mutmut_7': x_main__mutmut_7, 
    'x_main__mutmut_8': x_main__mutmut_8, 
    'x_main__mutmut_9': x_main__mutmut_9, 
    'x_main__mutmut_10': x_main__mutmut_10, 
    'x_main__mutmut_11': x_main__mutmut_11, 
    'x_main__mutmut_12': x_main__mutmut_12
}

def main(*args, **kwargs):
    result = _mutmut_trampoline(x_main__mutmut_orig, x_main__mutmut_mutants, args, kwargs)
    return result 

main.__signature__ = _mutmut_signature(x_main__mutmut_orig)
x_main__mutmut_orig.__name__ = 'x_main'


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())

# Offre un parsing robuste des arguments CLI
# Garantit le parsing des options CLI de manière déclarative
import argparse

# Fournit l'écriture CSV structurée pour le reporting
import csv

# Fournit une sérialisation JSON native pour la CI
import json

# Garantit des chemins indépendants du système
from pathlib import Path

# Assure des statistiques stables pour les moyennes
import numpy as np

# Réutilise l'évaluation existante pour relire les artefacts sauvegardés
from scripts import predict as predict_cli

# Fige le seuil d'acceptation minimal des runs
MINIMUM_ACCURACY = 0.60
# Fige la cible ambitieuse pour les livrables WBS
TARGET_ACCURACY = 0.75
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


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_orig() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_3() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "XXAgrège les accuracies par run, sujet et global à partir des XX"
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_4() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_5() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "AGRÈGE LES ACCURACIES PAR RUN, SUJET ET GLOBAL À PARTIR DES "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_6() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_7() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_8() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_9() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_10() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_11() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_12() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_13() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_14() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_15() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_16() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_17() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_18() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_19() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_20() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_21() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_22() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_23() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_24() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_25() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_26() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_27() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_28() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_29() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_30() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_31() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_32() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        None,
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_33() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=None,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_34() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=None,
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_35() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_36() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_37() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_38() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_39() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "XX--csv-outputXX",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_40() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--CSV-OUTPUT",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_41() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="XXChemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)XX",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_42() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="chemin de sortie du rapport csv (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_43() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CHEMIN DE SORTIE DU RAPPORT CSV (TYPE,SUBJECT,RUN,ACCURACY,THRESHOLDS)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_44() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        None,
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_45() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=None,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_46() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=None,
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_47() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_48() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_49() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_50() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_51() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "XX--json-outputXX",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_52() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--JSON-OUTPUT",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_53() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="XXChemin de sortie du rapport JSON aligné CIXX",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_54() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="chemin de sortie du rapport json aligné ci",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Construit un argument parser pour exposer l'agrégation en CLI
def x_build_parser__mutmut_55() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
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
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="CHEMIN DE SORTIE DU RAPPORT JSON ALIGNÉ CI",
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
    'x_build_parser__mutmut_33': x_build_parser__mutmut_33, 
    'x_build_parser__mutmut_34': x_build_parser__mutmut_34, 
    'x_build_parser__mutmut_35': x_build_parser__mutmut_35, 
    'x_build_parser__mutmut_36': x_build_parser__mutmut_36, 
    'x_build_parser__mutmut_37': x_build_parser__mutmut_37, 
    'x_build_parser__mutmut_38': x_build_parser__mutmut_38, 
    'x_build_parser__mutmut_39': x_build_parser__mutmut_39, 
    'x_build_parser__mutmut_40': x_build_parser__mutmut_40, 
    'x_build_parser__mutmut_41': x_build_parser__mutmut_41, 
    'x_build_parser__mutmut_42': x_build_parser__mutmut_42, 
    'x_build_parser__mutmut_43': x_build_parser__mutmut_43, 
    'x_build_parser__mutmut_44': x_build_parser__mutmut_44, 
    'x_build_parser__mutmut_45': x_build_parser__mutmut_45, 
    'x_build_parser__mutmut_46': x_build_parser__mutmut_46, 
    'x_build_parser__mutmut_47': x_build_parser__mutmut_47, 
    'x_build_parser__mutmut_48': x_build_parser__mutmut_48, 
    'x_build_parser__mutmut_49': x_build_parser__mutmut_49, 
    'x_build_parser__mutmut_50': x_build_parser__mutmut_50, 
    'x_build_parser__mutmut_51': x_build_parser__mutmut_51, 
    'x_build_parser__mutmut_52': x_build_parser__mutmut_52, 
    'x_build_parser__mutmut_53': x_build_parser__mutmut_53, 
    'x_build_parser__mutmut_54': x_build_parser__mutmut_54, 
    'x_build_parser__mutmut_55': x_build_parser__mutmut_55
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


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_orig(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_1(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = None
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_2(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(None, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_3(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, None, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_4(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, None, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_5(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, None)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_6(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_7(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_8(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_9(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, )
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_10(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "XXsubjectXX": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_11(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "SUBJECT": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_12(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "XXrunXX": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_13(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "RUN": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_14(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "XXaccuracyXX": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_15(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "ACCURACY": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_16(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["XXaccuracyXX"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_17(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["ACCURACY"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_18(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "XXmeets_minimumXX": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_19(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "MEETS_MINIMUM": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_20(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["XXaccuracyXX"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_21(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["ACCURACY"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_22(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] > MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_23(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "XXmeets_targetXX": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_24(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "MEETS_TARGET": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_25(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["XXaccuracyXX"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_26(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["ACCURACY"] >= TARGET_ACCURACY,
    }


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def x__score_run__mutmut_27(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] > TARGET_ACCURACY,
    }

x__score_run__mutmut_mutants : ClassVar[MutantDict] = {
'x__score_run__mutmut_1': x__score_run__mutmut_1, 
    'x__score_run__mutmut_2': x__score_run__mutmut_2, 
    'x__score_run__mutmut_3': x__score_run__mutmut_3, 
    'x__score_run__mutmut_4': x__score_run__mutmut_4, 
    'x__score_run__mutmut_5': x__score_run__mutmut_5, 
    'x__score_run__mutmut_6': x__score_run__mutmut_6, 
    'x__score_run__mutmut_7': x__score_run__mutmut_7, 
    'x__score_run__mutmut_8': x__score_run__mutmut_8, 
    'x__score_run__mutmut_9': x__score_run__mutmut_9, 
    'x__score_run__mutmut_10': x__score_run__mutmut_10, 
    'x__score_run__mutmut_11': x__score_run__mutmut_11, 
    'x__score_run__mutmut_12': x__score_run__mutmut_12, 
    'x__score_run__mutmut_13': x__score_run__mutmut_13, 
    'x__score_run__mutmut_14': x__score_run__mutmut_14, 
    'x__score_run__mutmut_15': x__score_run__mutmut_15, 
    'x__score_run__mutmut_16': x__score_run__mutmut_16, 
    'x__score_run__mutmut_17': x__score_run__mutmut_17, 
    'x__score_run__mutmut_18': x__score_run__mutmut_18, 
    'x__score_run__mutmut_19': x__score_run__mutmut_19, 
    'x__score_run__mutmut_20': x__score_run__mutmut_20, 
    'x__score_run__mutmut_21': x__score_run__mutmut_21, 
    'x__score_run__mutmut_22': x__score_run__mutmut_22, 
    'x__score_run__mutmut_23': x__score_run__mutmut_23, 
    'x__score_run__mutmut_24': x__score_run__mutmut_24, 
    'x__score_run__mutmut_25': x__score_run__mutmut_25, 
    'x__score_run__mutmut_26': x__score_run__mutmut_26, 
    'x__score_run__mutmut_27': x__score_run__mutmut_27
}

def _score_run(*args, **kwargs):
    result = _mutmut_trampoline(x__score_run__mutmut_orig, x__score_run__mutmut_mutants, args, kwargs)
    return result 

_score_run.__signature__ = _mutmut_signature(x__score_run__mutmut_orig)
x__score_run__mutmut_orig.__name__ = 'x__score_run'


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_orig(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_1(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = None
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_2(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(None)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_3(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = None
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_4(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = None
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_5(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = None
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_6(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(None, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_7(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, None, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_8(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, None, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_9(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, None)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_10(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_11(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_12(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_13(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, )
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_14(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(None)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_15(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_16(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = None
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_17(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(None)
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_18(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["XXaccuracyXX"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_19(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["ACCURACY"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_20(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = None
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_21(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "XXsubjectXX": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_22(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "SUBJECT": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_23(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "XXaccuracyXX": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_24(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "ACCURACY": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_25(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(None),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_26(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(None)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_27(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "XXmeets_minimumXX": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_28(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "MEETS_MINIMUM": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_29(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(None) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_30(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(None)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_31(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) > MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_32(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "XXmeets_targetXX": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_33(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "MEETS_TARGET": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_34(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(None) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_35(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(None)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_36(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) > TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_37(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = None
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_38(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["XXaccuracyXX"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_39(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["ACCURACY"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_40(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = None
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_41(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(None) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_42(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(None)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_43(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 1.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_44(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = None
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_45(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "XXaccuracyXX": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_46(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "ACCURACY": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_47(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "XXmeets_minimumXX": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_48(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "MEETS_MINIMUM": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_49(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy > MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_50(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "XXmeets_targetXX": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_51(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "MEETS_TARGET": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_52(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy > TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_53(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "XXrunsXX": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_54(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "RUNS": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_55(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "XXsubjectsXX": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_56(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "SUBJECTS": subject_entries,
        "global": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_57(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "XXglobalXX": global_entry,
    }


# Calcule les accuracies agrégées par run, sujet et global
def x_aggregate_scores__mutmut_58(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "GLOBAL": global_entry,
    }

x_aggregate_scores__mutmut_mutants : ClassVar[MutantDict] = {
'x_aggregate_scores__mutmut_1': x_aggregate_scores__mutmut_1, 
    'x_aggregate_scores__mutmut_2': x_aggregate_scores__mutmut_2, 
    'x_aggregate_scores__mutmut_3': x_aggregate_scores__mutmut_3, 
    'x_aggregate_scores__mutmut_4': x_aggregate_scores__mutmut_4, 
    'x_aggregate_scores__mutmut_5': x_aggregate_scores__mutmut_5, 
    'x_aggregate_scores__mutmut_6': x_aggregate_scores__mutmut_6, 
    'x_aggregate_scores__mutmut_7': x_aggregate_scores__mutmut_7, 
    'x_aggregate_scores__mutmut_8': x_aggregate_scores__mutmut_8, 
    'x_aggregate_scores__mutmut_9': x_aggregate_scores__mutmut_9, 
    'x_aggregate_scores__mutmut_10': x_aggregate_scores__mutmut_10, 
    'x_aggregate_scores__mutmut_11': x_aggregate_scores__mutmut_11, 
    'x_aggregate_scores__mutmut_12': x_aggregate_scores__mutmut_12, 
    'x_aggregate_scores__mutmut_13': x_aggregate_scores__mutmut_13, 
    'x_aggregate_scores__mutmut_14': x_aggregate_scores__mutmut_14, 
    'x_aggregate_scores__mutmut_15': x_aggregate_scores__mutmut_15, 
    'x_aggregate_scores__mutmut_16': x_aggregate_scores__mutmut_16, 
    'x_aggregate_scores__mutmut_17': x_aggregate_scores__mutmut_17, 
    'x_aggregate_scores__mutmut_18': x_aggregate_scores__mutmut_18, 
    'x_aggregate_scores__mutmut_19': x_aggregate_scores__mutmut_19, 
    'x_aggregate_scores__mutmut_20': x_aggregate_scores__mutmut_20, 
    'x_aggregate_scores__mutmut_21': x_aggregate_scores__mutmut_21, 
    'x_aggregate_scores__mutmut_22': x_aggregate_scores__mutmut_22, 
    'x_aggregate_scores__mutmut_23': x_aggregate_scores__mutmut_23, 
    'x_aggregate_scores__mutmut_24': x_aggregate_scores__mutmut_24, 
    'x_aggregate_scores__mutmut_25': x_aggregate_scores__mutmut_25, 
    'x_aggregate_scores__mutmut_26': x_aggregate_scores__mutmut_26, 
    'x_aggregate_scores__mutmut_27': x_aggregate_scores__mutmut_27, 
    'x_aggregate_scores__mutmut_28': x_aggregate_scores__mutmut_28, 
    'x_aggregate_scores__mutmut_29': x_aggregate_scores__mutmut_29, 
    'x_aggregate_scores__mutmut_30': x_aggregate_scores__mutmut_30, 
    'x_aggregate_scores__mutmut_31': x_aggregate_scores__mutmut_31, 
    'x_aggregate_scores__mutmut_32': x_aggregate_scores__mutmut_32, 
    'x_aggregate_scores__mutmut_33': x_aggregate_scores__mutmut_33, 
    'x_aggregate_scores__mutmut_34': x_aggregate_scores__mutmut_34, 
    'x_aggregate_scores__mutmut_35': x_aggregate_scores__mutmut_35, 
    'x_aggregate_scores__mutmut_36': x_aggregate_scores__mutmut_36, 
    'x_aggregate_scores__mutmut_37': x_aggregate_scores__mutmut_37, 
    'x_aggregate_scores__mutmut_38': x_aggregate_scores__mutmut_38, 
    'x_aggregate_scores__mutmut_39': x_aggregate_scores__mutmut_39, 
    'x_aggregate_scores__mutmut_40': x_aggregate_scores__mutmut_40, 
    'x_aggregate_scores__mutmut_41': x_aggregate_scores__mutmut_41, 
    'x_aggregate_scores__mutmut_42': x_aggregate_scores__mutmut_42, 
    'x_aggregate_scores__mutmut_43': x_aggregate_scores__mutmut_43, 
    'x_aggregate_scores__mutmut_44': x_aggregate_scores__mutmut_44, 
    'x_aggregate_scores__mutmut_45': x_aggregate_scores__mutmut_45, 
    'x_aggregate_scores__mutmut_46': x_aggregate_scores__mutmut_46, 
    'x_aggregate_scores__mutmut_47': x_aggregate_scores__mutmut_47, 
    'x_aggregate_scores__mutmut_48': x_aggregate_scores__mutmut_48, 
    'x_aggregate_scores__mutmut_49': x_aggregate_scores__mutmut_49, 
    'x_aggregate_scores__mutmut_50': x_aggregate_scores__mutmut_50, 
    'x_aggregate_scores__mutmut_51': x_aggregate_scores__mutmut_51, 
    'x_aggregate_scores__mutmut_52': x_aggregate_scores__mutmut_52, 
    'x_aggregate_scores__mutmut_53': x_aggregate_scores__mutmut_53, 
    'x_aggregate_scores__mutmut_54': x_aggregate_scores__mutmut_54, 
    'x_aggregate_scores__mutmut_55': x_aggregate_scores__mutmut_55, 
    'x_aggregate_scores__mutmut_56': x_aggregate_scores__mutmut_56, 
    'x_aggregate_scores__mutmut_57': x_aggregate_scores__mutmut_57, 
    'x_aggregate_scores__mutmut_58': x_aggregate_scores__mutmut_58
}

def aggregate_scores(*args, **kwargs):
    result = _mutmut_trampoline(x_aggregate_scores__mutmut_orig, x_aggregate_scores__mutmut_mutants, args, kwargs)
    return result 

aggregate_scores.__signature__ = _mutmut_signature(x_aggregate_scores__mutmut_orig)
x_aggregate_scores__mutmut_orig.__name__ = 'x_aggregate_scores'


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_orig(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_1(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=None, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_2(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=None)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_3(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_4(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, )
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_5(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=False, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_6(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=False)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_7(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open(None, encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_8(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding=None, newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_9(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline=None) as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_10(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open(encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_11(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_12(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", ) as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_13(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("XXwXX", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_14(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("W", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_15(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="XXutf-8XX", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_16(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="UTF-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_17(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="XXXX") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_18(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = None
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_19(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = None
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_20(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(None, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_21(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=None)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_22(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_23(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, )
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_24(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["XXrunsXX"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_25(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["RUNS"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_26(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                None
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_27(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "XXtypeXX": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_28(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "TYPE": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_29(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "XXrunXX",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_30(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "RUN",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_31(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "XXsubjectXX": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_32(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "SUBJECT": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_33(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["XXsubjectXX"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_34(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["SUBJECT"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_35(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "XXrunXX": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_36(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "RUN": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_37(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["XXrunXX"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_38(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["RUN"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_39(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "XXaccuracyXX": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_40(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "ACCURACY": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_41(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['XXaccuracyXX']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_42(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['ACCURACY']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_43(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "XXmeets_minimumXX": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_44(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "MEETS_MINIMUM": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_45(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["XXmeets_minimumXX"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_46(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["MEETS_MINIMUM"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_47(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "XXmeets_targetXX": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_48(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "MEETS_TARGET": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_49(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["XXmeets_targetXX"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_50(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["MEETS_TARGET"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_51(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["XXsubjectsXX"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_52(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["SUBJECTS"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_53(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                None
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_54(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "XXtypeXX": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_55(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "TYPE": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_56(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "XXsubjectXX",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_57(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "SUBJECT",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_58(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "XXsubjectXX": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_59(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "SUBJECT": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_60(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["XXsubjectXX"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_61(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["SUBJECT"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_62(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "XXrunXX": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_63(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "RUN": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_64(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "XXXX",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_65(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "XXaccuracyXX": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_66(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "ACCURACY": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_67(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['XXaccuracyXX']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_68(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['ACCURACY']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_69(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "XXmeets_minimumXX": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_70(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "MEETS_MINIMUM": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_71(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["XXmeets_minimumXX"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_72(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["MEETS_MINIMUM"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_73(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "XXmeets_targetXX": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_74(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "MEETS_TARGET": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_75(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["XXmeets_targetXX"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_76(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["MEETS_TARGET"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_77(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            None
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_78(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "XXtypeXX": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_79(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "TYPE": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_80(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "XXglobalXX",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_81(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "GLOBAL",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_82(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "XXsubjectXX": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_83(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "SUBJECT": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_84(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "XXXX",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_85(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "XXrunXX": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_86(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "RUN": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_87(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "XXXX",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_88(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "XXaccuracyXX": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_89(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "ACCURACY": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_90(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['XXglobalXX']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_91(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['GLOBAL']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_92(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['XXaccuracyXX']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_93(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['ACCURACY']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_94(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "XXmeets_minimumXX": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_95(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "MEETS_MINIMUM": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_96(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["XXglobalXX"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_97(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["GLOBAL"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_98(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["XXmeets_minimumXX"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_99(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["MEETS_MINIMUM"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_100(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "XXmeets_targetXX": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_101(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "MEETS_TARGET": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_102(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["XXglobalXX"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_103(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["GLOBAL"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_104(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["XXmeets_targetXX"],
            }
        )


# Sérialise le rapport agrégé au format CSV
def x_write_csv__mutmut_105(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["MEETS_TARGET"],
            }
        )

x_write_csv__mutmut_mutants : ClassVar[MutantDict] = {
'x_write_csv__mutmut_1': x_write_csv__mutmut_1, 
    'x_write_csv__mutmut_2': x_write_csv__mutmut_2, 
    'x_write_csv__mutmut_3': x_write_csv__mutmut_3, 
    'x_write_csv__mutmut_4': x_write_csv__mutmut_4, 
    'x_write_csv__mutmut_5': x_write_csv__mutmut_5, 
    'x_write_csv__mutmut_6': x_write_csv__mutmut_6, 
    'x_write_csv__mutmut_7': x_write_csv__mutmut_7, 
    'x_write_csv__mutmut_8': x_write_csv__mutmut_8, 
    'x_write_csv__mutmut_9': x_write_csv__mutmut_9, 
    'x_write_csv__mutmut_10': x_write_csv__mutmut_10, 
    'x_write_csv__mutmut_11': x_write_csv__mutmut_11, 
    'x_write_csv__mutmut_12': x_write_csv__mutmut_12, 
    'x_write_csv__mutmut_13': x_write_csv__mutmut_13, 
    'x_write_csv__mutmut_14': x_write_csv__mutmut_14, 
    'x_write_csv__mutmut_15': x_write_csv__mutmut_15, 
    'x_write_csv__mutmut_16': x_write_csv__mutmut_16, 
    'x_write_csv__mutmut_17': x_write_csv__mutmut_17, 
    'x_write_csv__mutmut_18': x_write_csv__mutmut_18, 
    'x_write_csv__mutmut_19': x_write_csv__mutmut_19, 
    'x_write_csv__mutmut_20': x_write_csv__mutmut_20, 
    'x_write_csv__mutmut_21': x_write_csv__mutmut_21, 
    'x_write_csv__mutmut_22': x_write_csv__mutmut_22, 
    'x_write_csv__mutmut_23': x_write_csv__mutmut_23, 
    'x_write_csv__mutmut_24': x_write_csv__mutmut_24, 
    'x_write_csv__mutmut_25': x_write_csv__mutmut_25, 
    'x_write_csv__mutmut_26': x_write_csv__mutmut_26, 
    'x_write_csv__mutmut_27': x_write_csv__mutmut_27, 
    'x_write_csv__mutmut_28': x_write_csv__mutmut_28, 
    'x_write_csv__mutmut_29': x_write_csv__mutmut_29, 
    'x_write_csv__mutmut_30': x_write_csv__mutmut_30, 
    'x_write_csv__mutmut_31': x_write_csv__mutmut_31, 
    'x_write_csv__mutmut_32': x_write_csv__mutmut_32, 
    'x_write_csv__mutmut_33': x_write_csv__mutmut_33, 
    'x_write_csv__mutmut_34': x_write_csv__mutmut_34, 
    'x_write_csv__mutmut_35': x_write_csv__mutmut_35, 
    'x_write_csv__mutmut_36': x_write_csv__mutmut_36, 
    'x_write_csv__mutmut_37': x_write_csv__mutmut_37, 
    'x_write_csv__mutmut_38': x_write_csv__mutmut_38, 
    'x_write_csv__mutmut_39': x_write_csv__mutmut_39, 
    'x_write_csv__mutmut_40': x_write_csv__mutmut_40, 
    'x_write_csv__mutmut_41': x_write_csv__mutmut_41, 
    'x_write_csv__mutmut_42': x_write_csv__mutmut_42, 
    'x_write_csv__mutmut_43': x_write_csv__mutmut_43, 
    'x_write_csv__mutmut_44': x_write_csv__mutmut_44, 
    'x_write_csv__mutmut_45': x_write_csv__mutmut_45, 
    'x_write_csv__mutmut_46': x_write_csv__mutmut_46, 
    'x_write_csv__mutmut_47': x_write_csv__mutmut_47, 
    'x_write_csv__mutmut_48': x_write_csv__mutmut_48, 
    'x_write_csv__mutmut_49': x_write_csv__mutmut_49, 
    'x_write_csv__mutmut_50': x_write_csv__mutmut_50, 
    'x_write_csv__mutmut_51': x_write_csv__mutmut_51, 
    'x_write_csv__mutmut_52': x_write_csv__mutmut_52, 
    'x_write_csv__mutmut_53': x_write_csv__mutmut_53, 
    'x_write_csv__mutmut_54': x_write_csv__mutmut_54, 
    'x_write_csv__mutmut_55': x_write_csv__mutmut_55, 
    'x_write_csv__mutmut_56': x_write_csv__mutmut_56, 
    'x_write_csv__mutmut_57': x_write_csv__mutmut_57, 
    'x_write_csv__mutmut_58': x_write_csv__mutmut_58, 
    'x_write_csv__mutmut_59': x_write_csv__mutmut_59, 
    'x_write_csv__mutmut_60': x_write_csv__mutmut_60, 
    'x_write_csv__mutmut_61': x_write_csv__mutmut_61, 
    'x_write_csv__mutmut_62': x_write_csv__mutmut_62, 
    'x_write_csv__mutmut_63': x_write_csv__mutmut_63, 
    'x_write_csv__mutmut_64': x_write_csv__mutmut_64, 
    'x_write_csv__mutmut_65': x_write_csv__mutmut_65, 
    'x_write_csv__mutmut_66': x_write_csv__mutmut_66, 
    'x_write_csv__mutmut_67': x_write_csv__mutmut_67, 
    'x_write_csv__mutmut_68': x_write_csv__mutmut_68, 
    'x_write_csv__mutmut_69': x_write_csv__mutmut_69, 
    'x_write_csv__mutmut_70': x_write_csv__mutmut_70, 
    'x_write_csv__mutmut_71': x_write_csv__mutmut_71, 
    'x_write_csv__mutmut_72': x_write_csv__mutmut_72, 
    'x_write_csv__mutmut_73': x_write_csv__mutmut_73, 
    'x_write_csv__mutmut_74': x_write_csv__mutmut_74, 
    'x_write_csv__mutmut_75': x_write_csv__mutmut_75, 
    'x_write_csv__mutmut_76': x_write_csv__mutmut_76, 
    'x_write_csv__mutmut_77': x_write_csv__mutmut_77, 
    'x_write_csv__mutmut_78': x_write_csv__mutmut_78, 
    'x_write_csv__mutmut_79': x_write_csv__mutmut_79, 
    'x_write_csv__mutmut_80': x_write_csv__mutmut_80, 
    'x_write_csv__mutmut_81': x_write_csv__mutmut_81, 
    'x_write_csv__mutmut_82': x_write_csv__mutmut_82, 
    'x_write_csv__mutmut_83': x_write_csv__mutmut_83, 
    'x_write_csv__mutmut_84': x_write_csv__mutmut_84, 
    'x_write_csv__mutmut_85': x_write_csv__mutmut_85, 
    'x_write_csv__mutmut_86': x_write_csv__mutmut_86, 
    'x_write_csv__mutmut_87': x_write_csv__mutmut_87, 
    'x_write_csv__mutmut_88': x_write_csv__mutmut_88, 
    'x_write_csv__mutmut_89': x_write_csv__mutmut_89, 
    'x_write_csv__mutmut_90': x_write_csv__mutmut_90, 
    'x_write_csv__mutmut_91': x_write_csv__mutmut_91, 
    'x_write_csv__mutmut_92': x_write_csv__mutmut_92, 
    'x_write_csv__mutmut_93': x_write_csv__mutmut_93, 
    'x_write_csv__mutmut_94': x_write_csv__mutmut_94, 
    'x_write_csv__mutmut_95': x_write_csv__mutmut_95, 
    'x_write_csv__mutmut_96': x_write_csv__mutmut_96, 
    'x_write_csv__mutmut_97': x_write_csv__mutmut_97, 
    'x_write_csv__mutmut_98': x_write_csv__mutmut_98, 
    'x_write_csv__mutmut_99': x_write_csv__mutmut_99, 
    'x_write_csv__mutmut_100': x_write_csv__mutmut_100, 
    'x_write_csv__mutmut_101': x_write_csv__mutmut_101, 
    'x_write_csv__mutmut_102': x_write_csv__mutmut_102, 
    'x_write_csv__mutmut_103': x_write_csv__mutmut_103, 
    'x_write_csv__mutmut_104': x_write_csv__mutmut_104, 
    'x_write_csv__mutmut_105': x_write_csv__mutmut_105
}

def write_csv(*args, **kwargs):
    result = _mutmut_trampoline(x_write_csv__mutmut_orig, x_write_csv__mutmut_mutants, args, kwargs)
    return result 

write_csv.__signature__ = _mutmut_signature(x_write_csv__mutmut_orig)
x_write_csv__mutmut_orig.__name__ = 'x_write_csv'


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_orig(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_1(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=None, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_2(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=None)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_3(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_4(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, )
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_5(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=False, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_6(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=False)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_7(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open(None, encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_8(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding=None) as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_9(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open(encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_10(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", ) as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_11(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("XXwXX", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_12(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("W", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_13(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="XXutf-8XX") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_14(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="UTF-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_15(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(None, handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_16(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, None, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_17(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=None)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_18(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(handle, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_19(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, indent=2)


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_20(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, )


# Sérialise le rapport agrégé au format JSON
def x_write_json__mutmut_21(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=3)

x_write_json__mutmut_mutants : ClassVar[MutantDict] = {
'x_write_json__mutmut_1': x_write_json__mutmut_1, 
    'x_write_json__mutmut_2': x_write_json__mutmut_2, 
    'x_write_json__mutmut_3': x_write_json__mutmut_3, 
    'x_write_json__mutmut_4': x_write_json__mutmut_4, 
    'x_write_json__mutmut_5': x_write_json__mutmut_5, 
    'x_write_json__mutmut_6': x_write_json__mutmut_6, 
    'x_write_json__mutmut_7': x_write_json__mutmut_7, 
    'x_write_json__mutmut_8': x_write_json__mutmut_8, 
    'x_write_json__mutmut_9': x_write_json__mutmut_9, 
    'x_write_json__mutmut_10': x_write_json__mutmut_10, 
    'x_write_json__mutmut_11': x_write_json__mutmut_11, 
    'x_write_json__mutmut_12': x_write_json__mutmut_12, 
    'x_write_json__mutmut_13': x_write_json__mutmut_13, 
    'x_write_json__mutmut_14': x_write_json__mutmut_14, 
    'x_write_json__mutmut_15': x_write_json__mutmut_15, 
    'x_write_json__mutmut_16': x_write_json__mutmut_16, 
    'x_write_json__mutmut_17': x_write_json__mutmut_17, 
    'x_write_json__mutmut_18': x_write_json__mutmut_18, 
    'x_write_json__mutmut_19': x_write_json__mutmut_19, 
    'x_write_json__mutmut_20': x_write_json__mutmut_20, 
    'x_write_json__mutmut_21': x_write_json__mutmut_21
}

def write_json(*args, **kwargs):
    result = _mutmut_trampoline(x_write_json__mutmut_orig, x_write_json__mutmut_mutants, args, kwargs)
    return result 

write_json.__signature__ = _mutmut_signature(x_write_json__mutmut_orig)
x_write_json__mutmut_orig.__name__ = 'x_write_json'


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_orig(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_1(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = None
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_2(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = None
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_3(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(None)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_4(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = None
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_5(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(None, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_6(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, None)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_7(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_8(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, )
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_9(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_10(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(None, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_11(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, None)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_12(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_13(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, )
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_14(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_15(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(None, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_16(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, None)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_17(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_18(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, )
    # Retourne 0 pour signaler un succès standard
    return 0


# Point d'entrée principal pour l'usage en ligne de commande
def x_main__mutmut_19(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
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
    'x_main__mutmut_12': x_main__mutmut_12, 
    'x_main__mutmut_13': x_main__mutmut_13, 
    'x_main__mutmut_14': x_main__mutmut_14, 
    'x_main__mutmut_15': x_main__mutmut_15, 
    'x_main__mutmut_16': x_main__mutmut_16, 
    'x_main__mutmut_17': x_main__mutmut_17, 
    'x_main__mutmut_18': x_main__mutmut_18, 
    'x_main__mutmut_19': x_main__mutmut_19
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

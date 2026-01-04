"""Visualisation WBS 3.3 : brut vs filtré Physionet."""

# Force un backend sans affichage pour les environnements CI et serveurs
import matplotlib

# Active Agg pour générer les PNG sans dépendre d'un écran
matplotlib.use("Agg")

# Importe argparse pour exposer les options CLI sujet/run/canaux
import argparse

# Importe json pour sérialiser la configuration accompagnant les figures
import json

# Importe dataclass pour encapsuler les paramètres de visualisation
from dataclasses import dataclass

# Importe pathlib pour gérer les chemins dataset et de sortie
from pathlib import Path

# Importe typing pour typer explicitement les séquences et tuples
from typing import Sequence, Tuple

# Importe pyplot pour tracer les figures comparatives
import matplotlib.pyplot as plt

# Importe BaseRaw pour typer précisément les enregistrements EEG
from mne.io import BaseRaw

# Importe le filtrage validé pour rester aligné avec preprocessing
from tpv.preprocessing import apply_bandpass_filter, load_physionet_raw
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


# Regroupe les paramètres de visualisation pour limiter les arguments
@dataclass
class VisualizationConfig:
    """Conteneur des options CLI pour la visualisation brut/filtré."""

    # Définit la sélection facultative de canaux à tracer
    channels: Sequence[str] | None
    # Définit le répertoire de sortie des figures générées
    output_dir: Path
    # Choisit la méthode de filtrage alignée avec preprocessing
    filter_method: str
    # Spécifie la bande fréquentielle appliquée au signal
    freq_band: Tuple[float, float]
    # Spécifie la durée de padding pour réduire les artefacts de bord
    pad_duration: float
    # Définit un titre optionnel pour annoter la figure
    title: str | None


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_orig() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_1() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = None
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_2() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=None
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_3() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "XXCharge un run Physionet, applique le filtre 8-40 Hz, XX"
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_4() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "charge un run physionet, applique le filtre 8-40 hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_5() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "CHARGE UN RUN PHYSIONET, APPLIQUE LE FILTRE 8-40 HZ, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_6() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument(None, help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_7() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help=None)
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_8() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument(help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_9() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", )
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_10() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("XXsubjectXX", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_11() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("SUBJECT", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_12() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="XXIdentifiant du sujet ex: S001XX")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_13() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="identifiant du sujet ex: s001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_14() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="IDENTIFIANT DU SUJET EX: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_15() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument(None, help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_16() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help=None)
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_17() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument(help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_18() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", )
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_19() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("XXrunXX", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_20() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("RUN", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_21() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="XXIdentifiant du run ex: R01XX")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_22() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="identifiant du run ex: r01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_23() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="IDENTIFIANT DU RUN EX: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_24() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        None,
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_25() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default=None,
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_26() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help=None,
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_27() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_28() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_29() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_30() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "XX--data-rootXX",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_31() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--DATA-ROOT",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_32() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="XXdataXX",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_33() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="DATA",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_34() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="XXRacine locale des données PhysionetXX",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_35() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="racine locale des données physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_36() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="RACINE LOCALE DES DONNÉES PHYSIONET",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_37() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        None, default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_38() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default=None, help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_39() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help=None
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_40() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_41() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_42() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_43() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "XX--output-dirXX", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_44() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--OUTPUT-DIR", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_45() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="XXdocs/vizXX", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_46() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="DOCS/VIZ", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_47() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="XXRépertoire de sauvegarde PNGXX"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_48() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="répertoire de sauvegarde png"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_49() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="RÉPERTOIRE DE SAUVEGARDE PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_50() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        None,
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_51() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs=None,
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_52() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help=None,
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_53() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_54() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_55() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_56() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "XX--channelsXX",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_57() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--CHANNELS",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_58() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="XX*XX",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_59() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="XXSous-ensemble de canaux à tracer (ex: C3 C4 Cz)XX",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_60() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="sous-ensemble de canaux à tracer (ex: c3 c4 cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_61() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="SOUS-ENSEMBLE DE CANAUX À TRACER (EX: C3 C4 CZ)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_62() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        None,
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_63() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=None,
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_64() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default=None,
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_65() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help=None,
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_66() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_67() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_68() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_69() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_70() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "XX--filter-methodXX",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_71() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--FILTER-METHOD",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_72() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["XXfirXX", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_73() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["FIR", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_74() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "XXiirXX"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_75() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "IIR"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_76() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="XXfirXX",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_77() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="FIR",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_78() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="XXFamille de filtre à appliquerXX",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_79() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_80() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="FAMILLE DE FILTRE À APPLIQUER",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_81() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        None,
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_82() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=None,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_83() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=None,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_84() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_85() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=None,
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_86() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help=None,
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_87() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_88() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_89() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_90() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_91() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_92() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_93() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "XX--freq-bandXX",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_94() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--FREQ-BAND",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_95() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=3,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_96() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(9.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_97() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 41.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_98() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("XXLOWXX", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_99() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("low", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_100() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "XXHIGHXX"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_101() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "high"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_102() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="XXBande passe-bas/passe-haut en HzXX",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_103() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="bande passe-bas/passe-haut en hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_104() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="BANDE PASSE-BAS/PASSE-HAUT EN HZ",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_105() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        None,
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_106() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=None,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_107() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=None,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_108() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help=None,
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_109() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_110() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_111() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_112() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_113() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "XX--pad-durationXX",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_114() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--PAD-DURATION",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_115() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=1.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_116() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="XXDurée de padding réfléchissant en secondesXX",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_117() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_118() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="DURÉE DE PADDING RÉFLÉCHISSANT EN SECONDES",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_119() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        None,
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_120() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help=None,
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_121() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_122() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_123() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_124() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "XX--titleXX",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_125() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--TITLE",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_126() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="XXTitre personnalisé pour la figureXX",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_127() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Centralise la construction du parseur pour harmoniser l'interface CLI
def x_build_parser__mutmut_128() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument("subject", help="Identifiant du sujet ex: S001")
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument("run", help="Identifiant du run ex: R01")
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir", default="docs/viz", help="Répertoire de sauvegarde PNG"
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="TITRE PERSONNALISÉ POUR LA FIGURE",
    )
    # Retourne le parseur configuré pour réutilisation en test
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
    'x_build_parser__mutmut_55': x_build_parser__mutmut_55, 
    'x_build_parser__mutmut_56': x_build_parser__mutmut_56, 
    'x_build_parser__mutmut_57': x_build_parser__mutmut_57, 
    'x_build_parser__mutmut_58': x_build_parser__mutmut_58, 
    'x_build_parser__mutmut_59': x_build_parser__mutmut_59, 
    'x_build_parser__mutmut_60': x_build_parser__mutmut_60, 
    'x_build_parser__mutmut_61': x_build_parser__mutmut_61, 
    'x_build_parser__mutmut_62': x_build_parser__mutmut_62, 
    'x_build_parser__mutmut_63': x_build_parser__mutmut_63, 
    'x_build_parser__mutmut_64': x_build_parser__mutmut_64, 
    'x_build_parser__mutmut_65': x_build_parser__mutmut_65, 
    'x_build_parser__mutmut_66': x_build_parser__mutmut_66, 
    'x_build_parser__mutmut_67': x_build_parser__mutmut_67, 
    'x_build_parser__mutmut_68': x_build_parser__mutmut_68, 
    'x_build_parser__mutmut_69': x_build_parser__mutmut_69, 
    'x_build_parser__mutmut_70': x_build_parser__mutmut_70, 
    'x_build_parser__mutmut_71': x_build_parser__mutmut_71, 
    'x_build_parser__mutmut_72': x_build_parser__mutmut_72, 
    'x_build_parser__mutmut_73': x_build_parser__mutmut_73, 
    'x_build_parser__mutmut_74': x_build_parser__mutmut_74, 
    'x_build_parser__mutmut_75': x_build_parser__mutmut_75, 
    'x_build_parser__mutmut_76': x_build_parser__mutmut_76, 
    'x_build_parser__mutmut_77': x_build_parser__mutmut_77, 
    'x_build_parser__mutmut_78': x_build_parser__mutmut_78, 
    'x_build_parser__mutmut_79': x_build_parser__mutmut_79, 
    'x_build_parser__mutmut_80': x_build_parser__mutmut_80, 
    'x_build_parser__mutmut_81': x_build_parser__mutmut_81, 
    'x_build_parser__mutmut_82': x_build_parser__mutmut_82, 
    'x_build_parser__mutmut_83': x_build_parser__mutmut_83, 
    'x_build_parser__mutmut_84': x_build_parser__mutmut_84, 
    'x_build_parser__mutmut_85': x_build_parser__mutmut_85, 
    'x_build_parser__mutmut_86': x_build_parser__mutmut_86, 
    'x_build_parser__mutmut_87': x_build_parser__mutmut_87, 
    'x_build_parser__mutmut_88': x_build_parser__mutmut_88, 
    'x_build_parser__mutmut_89': x_build_parser__mutmut_89, 
    'x_build_parser__mutmut_90': x_build_parser__mutmut_90, 
    'x_build_parser__mutmut_91': x_build_parser__mutmut_91, 
    'x_build_parser__mutmut_92': x_build_parser__mutmut_92, 
    'x_build_parser__mutmut_93': x_build_parser__mutmut_93, 
    'x_build_parser__mutmut_94': x_build_parser__mutmut_94, 
    'x_build_parser__mutmut_95': x_build_parser__mutmut_95, 
    'x_build_parser__mutmut_96': x_build_parser__mutmut_96, 
    'x_build_parser__mutmut_97': x_build_parser__mutmut_97, 
    'x_build_parser__mutmut_98': x_build_parser__mutmut_98, 
    'x_build_parser__mutmut_99': x_build_parser__mutmut_99, 
    'x_build_parser__mutmut_100': x_build_parser__mutmut_100, 
    'x_build_parser__mutmut_101': x_build_parser__mutmut_101, 
    'x_build_parser__mutmut_102': x_build_parser__mutmut_102, 
    'x_build_parser__mutmut_103': x_build_parser__mutmut_103, 
    'x_build_parser__mutmut_104': x_build_parser__mutmut_104, 
    'x_build_parser__mutmut_105': x_build_parser__mutmut_105, 
    'x_build_parser__mutmut_106': x_build_parser__mutmut_106, 
    'x_build_parser__mutmut_107': x_build_parser__mutmut_107, 
    'x_build_parser__mutmut_108': x_build_parser__mutmut_108, 
    'x_build_parser__mutmut_109': x_build_parser__mutmut_109, 
    'x_build_parser__mutmut_110': x_build_parser__mutmut_110, 
    'x_build_parser__mutmut_111': x_build_parser__mutmut_111, 
    'x_build_parser__mutmut_112': x_build_parser__mutmut_112, 
    'x_build_parser__mutmut_113': x_build_parser__mutmut_113, 
    'x_build_parser__mutmut_114': x_build_parser__mutmut_114, 
    'x_build_parser__mutmut_115': x_build_parser__mutmut_115, 
    'x_build_parser__mutmut_116': x_build_parser__mutmut_116, 
    'x_build_parser__mutmut_117': x_build_parser__mutmut_117, 
    'x_build_parser__mutmut_118': x_build_parser__mutmut_118, 
    'x_build_parser__mutmut_119': x_build_parser__mutmut_119, 
    'x_build_parser__mutmut_120': x_build_parser__mutmut_120, 
    'x_build_parser__mutmut_121': x_build_parser__mutmut_121, 
    'x_build_parser__mutmut_122': x_build_parser__mutmut_122, 
    'x_build_parser__mutmut_123': x_build_parser__mutmut_123, 
    'x_build_parser__mutmut_124': x_build_parser__mutmut_124, 
    'x_build_parser__mutmut_125': x_build_parser__mutmut_125, 
    'x_build_parser__mutmut_126': x_build_parser__mutmut_126, 
    'x_build_parser__mutmut_127': x_build_parser__mutmut_127, 
    'x_build_parser__mutmut_128': x_build_parser__mutmut_128
}

def build_parser(*args, **kwargs):
    result = _mutmut_trampoline(x_build_parser__mutmut_orig, x_build_parser__mutmut_mutants, args, kwargs)
    return result 

build_parser.__signature__ = _mutmut_signature(x_build_parser__mutmut_orig)
x_build_parser__mutmut_orig.__name__ = 'x_build_parser'


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_orig(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    # Compose le chemin EDF en respectant la convention README
    recording_path = normalized_root / subject / f"{run}.edf"
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_1(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = None
    # Compose le chemin EDF en respectant la convention README
    recording_path = normalized_root / subject / f"{run}.edf"
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_2(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(None).expanduser().resolve()
    # Compose le chemin EDF en respectant la convention README
    recording_path = normalized_root / subject / f"{run}.edf"
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_3(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    # Compose le chemin EDF en respectant la convention README
    recording_path = None
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_4(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    # Compose le chemin EDF en respectant la convention README
    recording_path = normalized_root / subject * f"{run}.edf"
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path


# Construit le chemin EDF attendu pour un sujet et un run donnés
def x_build_recording_path__mutmut_5(data_root: Path, subject: str, run: str) -> Path:
    """Retourne le chemin EDF data/<subject>/<run>.edf."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    # Compose le chemin EDF en respectant la convention README
    recording_path = normalized_root * subject / f"{run}.edf"
    # Retourne le chemin, même s'il n'existe pas encore pour simplifier les mocks
    return recording_path

x_build_recording_path__mutmut_mutants : ClassVar[MutantDict] = {
'x_build_recording_path__mutmut_1': x_build_recording_path__mutmut_1, 
    'x_build_recording_path__mutmut_2': x_build_recording_path__mutmut_2, 
    'x_build_recording_path__mutmut_3': x_build_recording_path__mutmut_3, 
    'x_build_recording_path__mutmut_4': x_build_recording_path__mutmut_4, 
    'x_build_recording_path__mutmut_5': x_build_recording_path__mutmut_5
}

def build_recording_path(*args, **kwargs):
    result = _mutmut_trampoline(x_build_recording_path__mutmut_orig, x_build_recording_path__mutmut_mutants, args, kwargs)
    return result 

build_recording_path.__signature__ = _mutmut_signature(x_build_recording_path__mutmut_orig)
x_build_recording_path__mutmut_orig.__name__ = 'x_build_recording_path'


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_orig(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_1(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_2(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(None)
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_3(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = None
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_4(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(None)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_5(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        None
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_6(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "XXsubjectXX": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_7(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "SUBJECT": recording_path.parent.name,
            "run": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_8(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "XXrunXX": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Charge un enregistrement Physionet complet avec métadonnées associées
def x_load_recording__mutmut_9(recording_path: Path) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    metadata.update(
        {
            "subject": recording_path.parent.name,
            "RUN": recording_path.stem,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata

x_load_recording__mutmut_mutants : ClassVar[MutantDict] = {
'x_load_recording__mutmut_1': x_load_recording__mutmut_1, 
    'x_load_recording__mutmut_2': x_load_recording__mutmut_2, 
    'x_load_recording__mutmut_3': x_load_recording__mutmut_3, 
    'x_load_recording__mutmut_4': x_load_recording__mutmut_4, 
    'x_load_recording__mutmut_5': x_load_recording__mutmut_5, 
    'x_load_recording__mutmut_6': x_load_recording__mutmut_6, 
    'x_load_recording__mutmut_7': x_load_recording__mutmut_7, 
    'x_load_recording__mutmut_8': x_load_recording__mutmut_8, 
    'x_load_recording__mutmut_9': x_load_recording__mutmut_9
}

def load_recording(*args, **kwargs):
    result = _mutmut_trampoline(x_load_recording__mutmut_orig, x_load_recording__mutmut_mutants, args, kwargs)
    return result 

load_recording.__signature__ = _mutmut_signature(x_load_recording__mutmut_orig)
x_load_recording__mutmut_orig.__name__ = 'x_load_recording'


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_orig(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_1(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is not None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_2(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = None
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_3(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_4(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(None)
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_5(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(None)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_6(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {'XX, XX'.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_7(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = None
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def x_pick_channels__mutmut_8(raw: BaseRaw, channels: Sequence[str] | None) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(None)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked

x_pick_channels__mutmut_mutants : ClassVar[MutantDict] = {
'x_pick_channels__mutmut_1': x_pick_channels__mutmut_1, 
    'x_pick_channels__mutmut_2': x_pick_channels__mutmut_2, 
    'x_pick_channels__mutmut_3': x_pick_channels__mutmut_3, 
    'x_pick_channels__mutmut_4': x_pick_channels__mutmut_4, 
    'x_pick_channels__mutmut_5': x_pick_channels__mutmut_5, 
    'x_pick_channels__mutmut_6': x_pick_channels__mutmut_6, 
    'x_pick_channels__mutmut_7': x_pick_channels__mutmut_7, 
    'x_pick_channels__mutmut_8': x_pick_channels__mutmut_8
}

def pick_channels(*args, **kwargs):
    result = _mutmut_trampoline(x_pick_channels__mutmut_orig, x_pick_channels__mutmut_mutants, args, kwargs)
    return result 

pick_channels.__signature__ = _mutmut_signature(x_pick_channels__mutmut_orig)
x_pick_channels__mutmut_orig.__name__ = 'x_pick_channels'


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_orig(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_1(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = None
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_2(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        None,
        method=method,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_3(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=None,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_4(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=None,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_5(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=freq_band,
        pad_duration=None,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_6(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        method=method,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_7(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_8(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Applique le filtre passe-bande avec les paramètres CLI
def x_filter_recording__mutmut_9(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=freq_band,
        )
    # Retourne le Raw filtré pour la visualisation
    return filtered

x_filter_recording__mutmut_mutants : ClassVar[MutantDict] = {
'x_filter_recording__mutmut_1': x_filter_recording__mutmut_1, 
    'x_filter_recording__mutmut_2': x_filter_recording__mutmut_2, 
    'x_filter_recording__mutmut_3': x_filter_recording__mutmut_3, 
    'x_filter_recording__mutmut_4': x_filter_recording__mutmut_4, 
    'x_filter_recording__mutmut_5': x_filter_recording__mutmut_5, 
    'x_filter_recording__mutmut_6': x_filter_recording__mutmut_6, 
    'x_filter_recording__mutmut_7': x_filter_recording__mutmut_7, 
    'x_filter_recording__mutmut_8': x_filter_recording__mutmut_8, 
    'x_filter_recording__mutmut_9': x_filter_recording__mutmut_9
}

def filter_recording(*args, **kwargs):
    result = _mutmut_trampoline(x_filter_recording__mutmut_orig, x_filter_recording__mutmut_mutants, args, kwargs)
    return result 

filter_recording.__signature__ = _mutmut_signature(x_filter_recording__mutmut_orig)
x_filter_recording__mutmut_orig.__name__ = 'x_filter_recording'


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_orig(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_1(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=None, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_2(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=None)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_3(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_4(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, )
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_5(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=False, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_6(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=False)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_7(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = None
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_8(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = None
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_9(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = None
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_10(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = None
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_11(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(None, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_12(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, None, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_13(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=None, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_14(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=None)
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_15(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_16(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_17(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_18(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, )
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_19(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_20(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_21(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_22(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_23(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_24(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(None)
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_25(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title and f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_26(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['XXsubjectXX']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_27(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['SUBJECT']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_28(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['XXrunXX']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_29(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['RUN']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_30(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(None):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_31(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(None, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_32(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, None, label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_33(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=None)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_34(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_35(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_36(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], )
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_37(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[1].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_38(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc=None)
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_39(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_40(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="XXupper rightXX")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_41(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="UPPER RIGHT")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_42(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel(None)
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_43(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[1].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_44(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("XXAmplitude (a.u.)XX")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_45(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_46(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("AMPLITUDE (A.U.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_47(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title(None)
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_48(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[1].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_49(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("XXSignal brutXX")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_50(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_51(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("SIGNAL BRUT")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_52(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(None):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_53(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(None, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_54(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, None, label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_55(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=None)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_56(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_57(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_58(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], )
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_59(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[2].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_60(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc=None)
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_61(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[2].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_62(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="XXupper rightXX")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_63(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="UPPER RIGHT")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_64(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel(None)
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_65(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[2].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_66(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("XXAmplitude filtrée (a.u.)XX")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_67(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_68(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("AMPLITUDE FILTRÉE (A.U.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_69(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel(None)
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_70(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[2].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_71(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("XXTemps (s)XX")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_72(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_73(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("TEMPS (S)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_74(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title(None)
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_75(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[2].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_76(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("XXSignal filtré 8-40 HzXX")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_77(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("signal filtré 8-40 hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_78(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("SIGNAL FILTRÉ 8-40 HZ")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_79(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(None)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_80(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(None)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_81(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        None,
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_82(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding=None,
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_83(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_84(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_85(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(None).write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_86(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix("XX.jsonXX").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_87(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".JSON").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_88(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps(None, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_89(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=None),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_90(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps(indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_91(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, ),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_92(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"XXmetadataXX": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_93(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"METADATA": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_94(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=3),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_95(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="XXutf-8XX",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def x_plot_raw_vs_filtered__mutmut_96(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")
    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        axes[0].plot(times, raw_data[idx], label=channel)
    # Ajoute une légende compacte au panneau brut
    axes[0].legend(loc="upper right")
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute une légende pour vérifier que les canaux concordent
    axes[1].legend(loc="upper right")
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")
    # Réduit les marges pour minimiser l'espace vide dans les PNG
    fig.tight_layout()
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="UTF-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path

x_plot_raw_vs_filtered__mutmut_mutants : ClassVar[MutantDict] = {
'x_plot_raw_vs_filtered__mutmut_1': x_plot_raw_vs_filtered__mutmut_1, 
    'x_plot_raw_vs_filtered__mutmut_2': x_plot_raw_vs_filtered__mutmut_2, 
    'x_plot_raw_vs_filtered__mutmut_3': x_plot_raw_vs_filtered__mutmut_3, 
    'x_plot_raw_vs_filtered__mutmut_4': x_plot_raw_vs_filtered__mutmut_4, 
    'x_plot_raw_vs_filtered__mutmut_5': x_plot_raw_vs_filtered__mutmut_5, 
    'x_plot_raw_vs_filtered__mutmut_6': x_plot_raw_vs_filtered__mutmut_6, 
    'x_plot_raw_vs_filtered__mutmut_7': x_plot_raw_vs_filtered__mutmut_7, 
    'x_plot_raw_vs_filtered__mutmut_8': x_plot_raw_vs_filtered__mutmut_8, 
    'x_plot_raw_vs_filtered__mutmut_9': x_plot_raw_vs_filtered__mutmut_9, 
    'x_plot_raw_vs_filtered__mutmut_10': x_plot_raw_vs_filtered__mutmut_10, 
    'x_plot_raw_vs_filtered__mutmut_11': x_plot_raw_vs_filtered__mutmut_11, 
    'x_plot_raw_vs_filtered__mutmut_12': x_plot_raw_vs_filtered__mutmut_12, 
    'x_plot_raw_vs_filtered__mutmut_13': x_plot_raw_vs_filtered__mutmut_13, 
    'x_plot_raw_vs_filtered__mutmut_14': x_plot_raw_vs_filtered__mutmut_14, 
    'x_plot_raw_vs_filtered__mutmut_15': x_plot_raw_vs_filtered__mutmut_15, 
    'x_plot_raw_vs_filtered__mutmut_16': x_plot_raw_vs_filtered__mutmut_16, 
    'x_plot_raw_vs_filtered__mutmut_17': x_plot_raw_vs_filtered__mutmut_17, 
    'x_plot_raw_vs_filtered__mutmut_18': x_plot_raw_vs_filtered__mutmut_18, 
    'x_plot_raw_vs_filtered__mutmut_19': x_plot_raw_vs_filtered__mutmut_19, 
    'x_plot_raw_vs_filtered__mutmut_20': x_plot_raw_vs_filtered__mutmut_20, 
    'x_plot_raw_vs_filtered__mutmut_21': x_plot_raw_vs_filtered__mutmut_21, 
    'x_plot_raw_vs_filtered__mutmut_22': x_plot_raw_vs_filtered__mutmut_22, 
    'x_plot_raw_vs_filtered__mutmut_23': x_plot_raw_vs_filtered__mutmut_23, 
    'x_plot_raw_vs_filtered__mutmut_24': x_plot_raw_vs_filtered__mutmut_24, 
    'x_plot_raw_vs_filtered__mutmut_25': x_plot_raw_vs_filtered__mutmut_25, 
    'x_plot_raw_vs_filtered__mutmut_26': x_plot_raw_vs_filtered__mutmut_26, 
    'x_plot_raw_vs_filtered__mutmut_27': x_plot_raw_vs_filtered__mutmut_27, 
    'x_plot_raw_vs_filtered__mutmut_28': x_plot_raw_vs_filtered__mutmut_28, 
    'x_plot_raw_vs_filtered__mutmut_29': x_plot_raw_vs_filtered__mutmut_29, 
    'x_plot_raw_vs_filtered__mutmut_30': x_plot_raw_vs_filtered__mutmut_30, 
    'x_plot_raw_vs_filtered__mutmut_31': x_plot_raw_vs_filtered__mutmut_31, 
    'x_plot_raw_vs_filtered__mutmut_32': x_plot_raw_vs_filtered__mutmut_32, 
    'x_plot_raw_vs_filtered__mutmut_33': x_plot_raw_vs_filtered__mutmut_33, 
    'x_plot_raw_vs_filtered__mutmut_34': x_plot_raw_vs_filtered__mutmut_34, 
    'x_plot_raw_vs_filtered__mutmut_35': x_plot_raw_vs_filtered__mutmut_35, 
    'x_plot_raw_vs_filtered__mutmut_36': x_plot_raw_vs_filtered__mutmut_36, 
    'x_plot_raw_vs_filtered__mutmut_37': x_plot_raw_vs_filtered__mutmut_37, 
    'x_plot_raw_vs_filtered__mutmut_38': x_plot_raw_vs_filtered__mutmut_38, 
    'x_plot_raw_vs_filtered__mutmut_39': x_plot_raw_vs_filtered__mutmut_39, 
    'x_plot_raw_vs_filtered__mutmut_40': x_plot_raw_vs_filtered__mutmut_40, 
    'x_plot_raw_vs_filtered__mutmut_41': x_plot_raw_vs_filtered__mutmut_41, 
    'x_plot_raw_vs_filtered__mutmut_42': x_plot_raw_vs_filtered__mutmut_42, 
    'x_plot_raw_vs_filtered__mutmut_43': x_plot_raw_vs_filtered__mutmut_43, 
    'x_plot_raw_vs_filtered__mutmut_44': x_plot_raw_vs_filtered__mutmut_44, 
    'x_plot_raw_vs_filtered__mutmut_45': x_plot_raw_vs_filtered__mutmut_45, 
    'x_plot_raw_vs_filtered__mutmut_46': x_plot_raw_vs_filtered__mutmut_46, 
    'x_plot_raw_vs_filtered__mutmut_47': x_plot_raw_vs_filtered__mutmut_47, 
    'x_plot_raw_vs_filtered__mutmut_48': x_plot_raw_vs_filtered__mutmut_48, 
    'x_plot_raw_vs_filtered__mutmut_49': x_plot_raw_vs_filtered__mutmut_49, 
    'x_plot_raw_vs_filtered__mutmut_50': x_plot_raw_vs_filtered__mutmut_50, 
    'x_plot_raw_vs_filtered__mutmut_51': x_plot_raw_vs_filtered__mutmut_51, 
    'x_plot_raw_vs_filtered__mutmut_52': x_plot_raw_vs_filtered__mutmut_52, 
    'x_plot_raw_vs_filtered__mutmut_53': x_plot_raw_vs_filtered__mutmut_53, 
    'x_plot_raw_vs_filtered__mutmut_54': x_plot_raw_vs_filtered__mutmut_54, 
    'x_plot_raw_vs_filtered__mutmut_55': x_plot_raw_vs_filtered__mutmut_55, 
    'x_plot_raw_vs_filtered__mutmut_56': x_plot_raw_vs_filtered__mutmut_56, 
    'x_plot_raw_vs_filtered__mutmut_57': x_plot_raw_vs_filtered__mutmut_57, 
    'x_plot_raw_vs_filtered__mutmut_58': x_plot_raw_vs_filtered__mutmut_58, 
    'x_plot_raw_vs_filtered__mutmut_59': x_plot_raw_vs_filtered__mutmut_59, 
    'x_plot_raw_vs_filtered__mutmut_60': x_plot_raw_vs_filtered__mutmut_60, 
    'x_plot_raw_vs_filtered__mutmut_61': x_plot_raw_vs_filtered__mutmut_61, 
    'x_plot_raw_vs_filtered__mutmut_62': x_plot_raw_vs_filtered__mutmut_62, 
    'x_plot_raw_vs_filtered__mutmut_63': x_plot_raw_vs_filtered__mutmut_63, 
    'x_plot_raw_vs_filtered__mutmut_64': x_plot_raw_vs_filtered__mutmut_64, 
    'x_plot_raw_vs_filtered__mutmut_65': x_plot_raw_vs_filtered__mutmut_65, 
    'x_plot_raw_vs_filtered__mutmut_66': x_plot_raw_vs_filtered__mutmut_66, 
    'x_plot_raw_vs_filtered__mutmut_67': x_plot_raw_vs_filtered__mutmut_67, 
    'x_plot_raw_vs_filtered__mutmut_68': x_plot_raw_vs_filtered__mutmut_68, 
    'x_plot_raw_vs_filtered__mutmut_69': x_plot_raw_vs_filtered__mutmut_69, 
    'x_plot_raw_vs_filtered__mutmut_70': x_plot_raw_vs_filtered__mutmut_70, 
    'x_plot_raw_vs_filtered__mutmut_71': x_plot_raw_vs_filtered__mutmut_71, 
    'x_plot_raw_vs_filtered__mutmut_72': x_plot_raw_vs_filtered__mutmut_72, 
    'x_plot_raw_vs_filtered__mutmut_73': x_plot_raw_vs_filtered__mutmut_73, 
    'x_plot_raw_vs_filtered__mutmut_74': x_plot_raw_vs_filtered__mutmut_74, 
    'x_plot_raw_vs_filtered__mutmut_75': x_plot_raw_vs_filtered__mutmut_75, 
    'x_plot_raw_vs_filtered__mutmut_76': x_plot_raw_vs_filtered__mutmut_76, 
    'x_plot_raw_vs_filtered__mutmut_77': x_plot_raw_vs_filtered__mutmut_77, 
    'x_plot_raw_vs_filtered__mutmut_78': x_plot_raw_vs_filtered__mutmut_78, 
    'x_plot_raw_vs_filtered__mutmut_79': x_plot_raw_vs_filtered__mutmut_79, 
    'x_plot_raw_vs_filtered__mutmut_80': x_plot_raw_vs_filtered__mutmut_80, 
    'x_plot_raw_vs_filtered__mutmut_81': x_plot_raw_vs_filtered__mutmut_81, 
    'x_plot_raw_vs_filtered__mutmut_82': x_plot_raw_vs_filtered__mutmut_82, 
    'x_plot_raw_vs_filtered__mutmut_83': x_plot_raw_vs_filtered__mutmut_83, 
    'x_plot_raw_vs_filtered__mutmut_84': x_plot_raw_vs_filtered__mutmut_84, 
    'x_plot_raw_vs_filtered__mutmut_85': x_plot_raw_vs_filtered__mutmut_85, 
    'x_plot_raw_vs_filtered__mutmut_86': x_plot_raw_vs_filtered__mutmut_86, 
    'x_plot_raw_vs_filtered__mutmut_87': x_plot_raw_vs_filtered__mutmut_87, 
    'x_plot_raw_vs_filtered__mutmut_88': x_plot_raw_vs_filtered__mutmut_88, 
    'x_plot_raw_vs_filtered__mutmut_89': x_plot_raw_vs_filtered__mutmut_89, 
    'x_plot_raw_vs_filtered__mutmut_90': x_plot_raw_vs_filtered__mutmut_90, 
    'x_plot_raw_vs_filtered__mutmut_91': x_plot_raw_vs_filtered__mutmut_91, 
    'x_plot_raw_vs_filtered__mutmut_92': x_plot_raw_vs_filtered__mutmut_92, 
    'x_plot_raw_vs_filtered__mutmut_93': x_plot_raw_vs_filtered__mutmut_93, 
    'x_plot_raw_vs_filtered__mutmut_94': x_plot_raw_vs_filtered__mutmut_94, 
    'x_plot_raw_vs_filtered__mutmut_95': x_plot_raw_vs_filtered__mutmut_95, 
    'x_plot_raw_vs_filtered__mutmut_96': x_plot_raw_vs_filtered__mutmut_96
}

def plot_raw_vs_filtered(*args, **kwargs):
    result = _mutmut_trampoline(x_plot_raw_vs_filtered__mutmut_orig, x_plot_raw_vs_filtered__mutmut_mutants, args, kwargs)
    return result 

plot_raw_vs_filtered.__signature__ = _mutmut_signature(x_plot_raw_vs_filtered__mutmut_orig)
x_plot_raw_vs_filtered__mutmut_orig.__name__ = 'x_plot_raw_vs_filtered'


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_orig(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_1(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = None
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_2(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(None, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_3(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, None, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_4(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, None)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_5(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_6(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_7(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, )
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_8(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = None
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_9(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(None)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_10(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = None
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_11(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(None, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_12(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, None)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_13(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_14(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, )
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_15(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = None
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_16(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        None,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_17(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=None,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_18(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=None,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_19(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=None,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_20(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_21(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_22(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_23(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_24(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = None
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_25(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) * f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_26(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(None) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_27(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        None,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_28(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        None,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_29(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        None,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_30(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        None,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_31(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        None,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_32(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_33(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        output_path,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_34(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        config.title,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_35(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        metadata,
    )


# Orchestration complète du chargement, filtrage et traçage
def x_visualize_run__mutmut_36(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        )

x_visualize_run__mutmut_mutants : ClassVar[MutantDict] = {
'x_visualize_run__mutmut_1': x_visualize_run__mutmut_1, 
    'x_visualize_run__mutmut_2': x_visualize_run__mutmut_2, 
    'x_visualize_run__mutmut_3': x_visualize_run__mutmut_3, 
    'x_visualize_run__mutmut_4': x_visualize_run__mutmut_4, 
    'x_visualize_run__mutmut_5': x_visualize_run__mutmut_5, 
    'x_visualize_run__mutmut_6': x_visualize_run__mutmut_6, 
    'x_visualize_run__mutmut_7': x_visualize_run__mutmut_7, 
    'x_visualize_run__mutmut_8': x_visualize_run__mutmut_8, 
    'x_visualize_run__mutmut_9': x_visualize_run__mutmut_9, 
    'x_visualize_run__mutmut_10': x_visualize_run__mutmut_10, 
    'x_visualize_run__mutmut_11': x_visualize_run__mutmut_11, 
    'x_visualize_run__mutmut_12': x_visualize_run__mutmut_12, 
    'x_visualize_run__mutmut_13': x_visualize_run__mutmut_13, 
    'x_visualize_run__mutmut_14': x_visualize_run__mutmut_14, 
    'x_visualize_run__mutmut_15': x_visualize_run__mutmut_15, 
    'x_visualize_run__mutmut_16': x_visualize_run__mutmut_16, 
    'x_visualize_run__mutmut_17': x_visualize_run__mutmut_17, 
    'x_visualize_run__mutmut_18': x_visualize_run__mutmut_18, 
    'x_visualize_run__mutmut_19': x_visualize_run__mutmut_19, 
    'x_visualize_run__mutmut_20': x_visualize_run__mutmut_20, 
    'x_visualize_run__mutmut_21': x_visualize_run__mutmut_21, 
    'x_visualize_run__mutmut_22': x_visualize_run__mutmut_22, 
    'x_visualize_run__mutmut_23': x_visualize_run__mutmut_23, 
    'x_visualize_run__mutmut_24': x_visualize_run__mutmut_24, 
    'x_visualize_run__mutmut_25': x_visualize_run__mutmut_25, 
    'x_visualize_run__mutmut_26': x_visualize_run__mutmut_26, 
    'x_visualize_run__mutmut_27': x_visualize_run__mutmut_27, 
    'x_visualize_run__mutmut_28': x_visualize_run__mutmut_28, 
    'x_visualize_run__mutmut_29': x_visualize_run__mutmut_29, 
    'x_visualize_run__mutmut_30': x_visualize_run__mutmut_30, 
    'x_visualize_run__mutmut_31': x_visualize_run__mutmut_31, 
    'x_visualize_run__mutmut_32': x_visualize_run__mutmut_32, 
    'x_visualize_run__mutmut_33': x_visualize_run__mutmut_33, 
    'x_visualize_run__mutmut_34': x_visualize_run__mutmut_34, 
    'x_visualize_run__mutmut_35': x_visualize_run__mutmut_35, 
    'x_visualize_run__mutmut_36': x_visualize_run__mutmut_36
}

def visualize_run(*args, **kwargs):
    result = _mutmut_trampoline(x_visualize_run__mutmut_orig, x_visualize_run__mutmut_mutants, args, kwargs)
    return result 

visualize_run.__signature__ = _mutmut_signature(x_visualize_run__mutmut_orig)
x_visualize_run__mutmut_orig.__name__ = 'x_visualize_run'


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_orig() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_1() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = None
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_2() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = None
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_3() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = None
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_4() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=None,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_5() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=None,
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_6() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=None,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_7() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=None,
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_8() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=None,
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_9() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=None,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_10() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_11() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_12() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_13() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_14() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_15() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_16() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(None),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_17() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(None), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_18() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[1]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_19() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(None)),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_20() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[2])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_21() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(None),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_22() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=None,
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_23() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=None,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_24() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=None,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_25() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=None,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_26() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_27() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_28() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_29() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_30() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(None),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_31() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(None)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_32() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(None) from error


# Point d'entrée CLI conforme aux instructions README
def x_main__mutmut_33() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(2) from error

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
    'x_main__mutmut_19': x_main__mutmut_19, 
    'x_main__mutmut_20': x_main__mutmut_20, 
    'x_main__mutmut_21': x_main__mutmut_21, 
    'x_main__mutmut_22': x_main__mutmut_22, 
    'x_main__mutmut_23': x_main__mutmut_23, 
    'x_main__mutmut_24': x_main__mutmut_24, 
    'x_main__mutmut_25': x_main__mutmut_25, 
    'x_main__mutmut_26': x_main__mutmut_26, 
    'x_main__mutmut_27': x_main__mutmut_27, 
    'x_main__mutmut_28': x_main__mutmut_28, 
    'x_main__mutmut_29': x_main__mutmut_29, 
    'x_main__mutmut_30': x_main__mutmut_30, 
    'x_main__mutmut_31': x_main__mutmut_31, 
    'x_main__mutmut_32': x_main__mutmut_32, 
    'x_main__mutmut_33': x_main__mutmut_33
}

def main(*args, **kwargs):
    result = _mutmut_trampoline(x_main__mutmut_orig, x_main__mutmut_mutants, args, kwargs)
    return result 

main.__signature__ = _mutmut_signature(x_main__mutmut_orig)
x_main__mutmut_orig.__name__ = 'x_main'


# Active l'exécution du main uniquement en invocation directe
if __name__ == "__main__":
    main()

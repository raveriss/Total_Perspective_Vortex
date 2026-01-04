#!/usr/bin/env python3
"""CLI pour récupérer le dataset Physionet avec vérification stricte."""

# Rassure les instructions sur l'usage explicite des arguments CLI
import argparse

# Garantit un contrôle de somme cryptographique sur les fichiers copiés
import hashlib

# Sérialise et lit le manifeste JSON attendu
import json

# Permet la duplication locale des fichiers EDF
import shutil

# Facilite la récupération HTTP sans dépendance tierce
import urllib.error

# Parse les URLs pour restreindre les schémas autorisés
import urllib.parse

# Assure l'ouverture sécurisée des URLs distantes
import urllib.request

# Gère les chemins de façon robuste entre systèmes
from pathlib import Path

# Typage explicite pour rassurer mypy sur la structure du manifeste
from typing import Any

# Prépare un formatage cohérent pour les messages d'erreur
ERROR_PREFIX = "ERROR:"
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


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_orig(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_1(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_2(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            None
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_3(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            None
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_4(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open(None, encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_5(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding=None) as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_6(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open(encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_7(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", ) as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_8(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("XXrXX", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_9(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("R", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_10(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="XXutf-8XX") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_11(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="UTF-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_12(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = None
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_13(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(None)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_14(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_15(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            None
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_16(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(None)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_17(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = None
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_18(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get(None, [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_19(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", None)
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_20(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get([])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_21(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", )
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_22(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("XXfilesXX", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_23(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("FILES", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_24(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_25(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            None
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_26(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = None
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_27(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_28(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                None
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(entry)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files


# Encapsule le chargement du manifeste pour centraliser les contrôles
def x_load_manifest__mutmut_29(manifest_path: Path) -> list[dict[str, Any]]:
    """Charge la liste des fichiers attendus depuis un manifeste JSON."""
    # Impose l'existence du manifeste avant toute manipulation
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} manifeste introuvable: {manifest_path}"
        )
    # Évite les répertoires accidentels en vérifiant le type de fichier
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"{ERROR_PREFIX} manifeste non lisible: {manifest_path}"
        )
    # Refuse les manifestes non dictionnaires pour éviter les attributs manquants
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    # Valide le type du manifeste avant toute lecture de champ
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"{ERROR_PREFIX} format de manifeste inattendu: {type(manifest_data)}"
        )
    # Extrait la liste brute en validant le format du manifeste
    raw_files = manifest_data.get("files", [])
    # Bloque les schémas inattendus pour stabiliser la boucle d'import
    if not isinstance(raw_files, list):
        raise ValueError(
            f"{ERROR_PREFIX} le champ 'files' du manifeste n'est pas une liste"
        )
    # Initialiser le conteneur typé pour rassurer l'analyse statique
    typed_files: list[dict[str, Any]] = []
    # Valide chaque entrée pour éviter des structures non dict
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{ERROR_PREFIX} entrée de manifeste non dictionnaire: {entry}"
            )
        # Conserve l'entrée validée dans la liste typée
        typed_files.append(None)
    # Extrait la liste des fichiers attendus pour itération ultérieure
    return typed_files

x_load_manifest__mutmut_mutants : ClassVar[MutantDict] = {
'x_load_manifest__mutmut_1': x_load_manifest__mutmut_1, 
    'x_load_manifest__mutmut_2': x_load_manifest__mutmut_2, 
    'x_load_manifest__mutmut_3': x_load_manifest__mutmut_3, 
    'x_load_manifest__mutmut_4': x_load_manifest__mutmut_4, 
    'x_load_manifest__mutmut_5': x_load_manifest__mutmut_5, 
    'x_load_manifest__mutmut_6': x_load_manifest__mutmut_6, 
    'x_load_manifest__mutmut_7': x_load_manifest__mutmut_7, 
    'x_load_manifest__mutmut_8': x_load_manifest__mutmut_8, 
    'x_load_manifest__mutmut_9': x_load_manifest__mutmut_9, 
    'x_load_manifest__mutmut_10': x_load_manifest__mutmut_10, 
    'x_load_manifest__mutmut_11': x_load_manifest__mutmut_11, 
    'x_load_manifest__mutmut_12': x_load_manifest__mutmut_12, 
    'x_load_manifest__mutmut_13': x_load_manifest__mutmut_13, 
    'x_load_manifest__mutmut_14': x_load_manifest__mutmut_14, 
    'x_load_manifest__mutmut_15': x_load_manifest__mutmut_15, 
    'x_load_manifest__mutmut_16': x_load_manifest__mutmut_16, 
    'x_load_manifest__mutmut_17': x_load_manifest__mutmut_17, 
    'x_load_manifest__mutmut_18': x_load_manifest__mutmut_18, 
    'x_load_manifest__mutmut_19': x_load_manifest__mutmut_19, 
    'x_load_manifest__mutmut_20': x_load_manifest__mutmut_20, 
    'x_load_manifest__mutmut_21': x_load_manifest__mutmut_21, 
    'x_load_manifest__mutmut_22': x_load_manifest__mutmut_22, 
    'x_load_manifest__mutmut_23': x_load_manifest__mutmut_23, 
    'x_load_manifest__mutmut_24': x_load_manifest__mutmut_24, 
    'x_load_manifest__mutmut_25': x_load_manifest__mutmut_25, 
    'x_load_manifest__mutmut_26': x_load_manifest__mutmut_26, 
    'x_load_manifest__mutmut_27': x_load_manifest__mutmut_27, 
    'x_load_manifest__mutmut_28': x_load_manifest__mutmut_28, 
    'x_load_manifest__mutmut_29': x_load_manifest__mutmut_29
}

def load_manifest(*args, **kwargs):
    result = _mutmut_trampoline(x_load_manifest__mutmut_orig, x_load_manifest__mutmut_mutants, args, kwargs)
    return result 

load_manifest.__signature__ = _mutmut_signature(x_load_manifest__mutmut_orig)
x_load_manifest__mutmut_orig.__name__ = 'x_load_manifest'


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_orig(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_1(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = None
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_2(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open(None) as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_3(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("XXrbXX") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_4(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("RB") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_5(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(None, b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_6(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), None):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_7(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_8(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), ):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_9(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: None, b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_10(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(None), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_11(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8193), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_12(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b"XXXX"):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def x_compute_sha256__mutmut_13(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(None)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()

x_compute_sha256__mutmut_mutants : ClassVar[MutantDict] = {
'x_compute_sha256__mutmut_1': x_compute_sha256__mutmut_1, 
    'x_compute_sha256__mutmut_2': x_compute_sha256__mutmut_2, 
    'x_compute_sha256__mutmut_3': x_compute_sha256__mutmut_3, 
    'x_compute_sha256__mutmut_4': x_compute_sha256__mutmut_4, 
    'x_compute_sha256__mutmut_5': x_compute_sha256__mutmut_5, 
    'x_compute_sha256__mutmut_6': x_compute_sha256__mutmut_6, 
    'x_compute_sha256__mutmut_7': x_compute_sha256__mutmut_7, 
    'x_compute_sha256__mutmut_8': x_compute_sha256__mutmut_8, 
    'x_compute_sha256__mutmut_9': x_compute_sha256__mutmut_9, 
    'x_compute_sha256__mutmut_10': x_compute_sha256__mutmut_10, 
    'x_compute_sha256__mutmut_11': x_compute_sha256__mutmut_11, 
    'x_compute_sha256__mutmut_12': x_compute_sha256__mutmut_12, 
    'x_compute_sha256__mutmut_13': x_compute_sha256__mutmut_13
}

def compute_sha256(*args, **kwargs):
    result = _mutmut_trampoline(x_compute_sha256__mutmut_orig, x_compute_sha256__mutmut_mutants, args, kwargs)
    return result 

compute_sha256.__signature__ = _mutmut_signature(x_compute_sha256__mutmut_orig)
x_compute_sha256__mutmut_orig.__name__ = 'x_compute_sha256'


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_orig(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_1(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = None
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_2(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(None)
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_3(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["XXpathXX"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_4(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["PATH"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_5(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = None
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_6(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root * relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_7(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=None, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_8(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=None)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_9(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_10(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, )
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_11(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=False, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_12(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=False)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_13(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = None
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_14(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") and source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_15(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith(None) or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_16(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("XXhttp://XX") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_17(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("HTTP://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_18(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith(None)
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_19(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("XXhttps://XX")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_20(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("HTTPS://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_21(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = None
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_22(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip(None)}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_23(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.lstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_24(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('XX/XX')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_25(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_26(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = None
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_27(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(None)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_28(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_29(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                None
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_30(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(None, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_31(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, None)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_32(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_33(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, )
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_34(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = None
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_35(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(None)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_36(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_37(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("XXhttpXX", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_38(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("HTTP", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_39(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "XXhttpsXX"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_40(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "HTTPS"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_41(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(None)
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_42(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = None
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_43(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(None) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_44(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open(None) as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_45(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("XXwbXX") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_46(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("WB") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_47(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(None, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_48(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, None)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_49(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_50(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, )
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"{ERROR_PREFIX} téléchargement impossible: {error}"
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path


# Choisit automatiquement la méthode (download ou copie locale)
def x_retrieve_file__mutmut_51(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data
    destination_path = destination_root / relative_path
    # Garantit l'existence de l'arborescence cible
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Détermine si la source est distante via un schéma HTTP(S)
    is_remote = source_root.startswith("http://") or source_root.startswith("https://")
    # Calcule le chemin complet côté source pour copie ou téléchargement
    source_location = f"{source_root.rstrip('/')}/{relative_path.as_posix()}"
    # Dirige vers une copie locale si la source n'est pas HTTP(S)
    if not is_remote:
        # Compose le chemin absolu pour la lecture locale
        candidate = Path(source_location)
        # Vérifie l'existence du fichier source avant copie
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ERROR_PREFIX} fichier source absent: {candidate}"
            )
        # Sécurise la copie en écrasant explicitement les doublons
        shutil.copy2(candidate, destination_path)
        # Retourne le chemin final pour vérifications ultérieures
        return destination_path
    # Tente un téléchargement HTTP avec gestion explicite des erreurs
    try:
        # Valide le schéma pour limiter urlopen aux protocoles sûrs
        parsed_url = urllib.parse.urlparse(source_location)
        # Refuse les schémas non HTTP pour limiter les vecteurs d'attaque
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"{ERROR_PREFIX} schéma URL interdit: {parsed_url.scheme}")
        # Instancie un opener standard pour bannir les schémas non validés
        opener = urllib.request.build_opener()
        # Télécharge le flux via l'opener après validation du schéma
        with opener.open(source_location) as response:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(response, target)
    except urllib.error.URLError as error:
        raise ConnectionError(
            None
        ) from error
    # Confirme la disponibilité du fichier téléchargé
    return destination_path

x_retrieve_file__mutmut_mutants : ClassVar[MutantDict] = {
'x_retrieve_file__mutmut_1': x_retrieve_file__mutmut_1, 
    'x_retrieve_file__mutmut_2': x_retrieve_file__mutmut_2, 
    'x_retrieve_file__mutmut_3': x_retrieve_file__mutmut_3, 
    'x_retrieve_file__mutmut_4': x_retrieve_file__mutmut_4, 
    'x_retrieve_file__mutmut_5': x_retrieve_file__mutmut_5, 
    'x_retrieve_file__mutmut_6': x_retrieve_file__mutmut_6, 
    'x_retrieve_file__mutmut_7': x_retrieve_file__mutmut_7, 
    'x_retrieve_file__mutmut_8': x_retrieve_file__mutmut_8, 
    'x_retrieve_file__mutmut_9': x_retrieve_file__mutmut_9, 
    'x_retrieve_file__mutmut_10': x_retrieve_file__mutmut_10, 
    'x_retrieve_file__mutmut_11': x_retrieve_file__mutmut_11, 
    'x_retrieve_file__mutmut_12': x_retrieve_file__mutmut_12, 
    'x_retrieve_file__mutmut_13': x_retrieve_file__mutmut_13, 
    'x_retrieve_file__mutmut_14': x_retrieve_file__mutmut_14, 
    'x_retrieve_file__mutmut_15': x_retrieve_file__mutmut_15, 
    'x_retrieve_file__mutmut_16': x_retrieve_file__mutmut_16, 
    'x_retrieve_file__mutmut_17': x_retrieve_file__mutmut_17, 
    'x_retrieve_file__mutmut_18': x_retrieve_file__mutmut_18, 
    'x_retrieve_file__mutmut_19': x_retrieve_file__mutmut_19, 
    'x_retrieve_file__mutmut_20': x_retrieve_file__mutmut_20, 
    'x_retrieve_file__mutmut_21': x_retrieve_file__mutmut_21, 
    'x_retrieve_file__mutmut_22': x_retrieve_file__mutmut_22, 
    'x_retrieve_file__mutmut_23': x_retrieve_file__mutmut_23, 
    'x_retrieve_file__mutmut_24': x_retrieve_file__mutmut_24, 
    'x_retrieve_file__mutmut_25': x_retrieve_file__mutmut_25, 
    'x_retrieve_file__mutmut_26': x_retrieve_file__mutmut_26, 
    'x_retrieve_file__mutmut_27': x_retrieve_file__mutmut_27, 
    'x_retrieve_file__mutmut_28': x_retrieve_file__mutmut_28, 
    'x_retrieve_file__mutmut_29': x_retrieve_file__mutmut_29, 
    'x_retrieve_file__mutmut_30': x_retrieve_file__mutmut_30, 
    'x_retrieve_file__mutmut_31': x_retrieve_file__mutmut_31, 
    'x_retrieve_file__mutmut_32': x_retrieve_file__mutmut_32, 
    'x_retrieve_file__mutmut_33': x_retrieve_file__mutmut_33, 
    'x_retrieve_file__mutmut_34': x_retrieve_file__mutmut_34, 
    'x_retrieve_file__mutmut_35': x_retrieve_file__mutmut_35, 
    'x_retrieve_file__mutmut_36': x_retrieve_file__mutmut_36, 
    'x_retrieve_file__mutmut_37': x_retrieve_file__mutmut_37, 
    'x_retrieve_file__mutmut_38': x_retrieve_file__mutmut_38, 
    'x_retrieve_file__mutmut_39': x_retrieve_file__mutmut_39, 
    'x_retrieve_file__mutmut_40': x_retrieve_file__mutmut_40, 
    'x_retrieve_file__mutmut_41': x_retrieve_file__mutmut_41, 
    'x_retrieve_file__mutmut_42': x_retrieve_file__mutmut_42, 
    'x_retrieve_file__mutmut_43': x_retrieve_file__mutmut_43, 
    'x_retrieve_file__mutmut_44': x_retrieve_file__mutmut_44, 
    'x_retrieve_file__mutmut_45': x_retrieve_file__mutmut_45, 
    'x_retrieve_file__mutmut_46': x_retrieve_file__mutmut_46, 
    'x_retrieve_file__mutmut_47': x_retrieve_file__mutmut_47, 
    'x_retrieve_file__mutmut_48': x_retrieve_file__mutmut_48, 
    'x_retrieve_file__mutmut_49': x_retrieve_file__mutmut_49, 
    'x_retrieve_file__mutmut_50': x_retrieve_file__mutmut_50, 
    'x_retrieve_file__mutmut_51': x_retrieve_file__mutmut_51
}

def retrieve_file(*args, **kwargs):
    result = _mutmut_trampoline(x_retrieve_file__mutmut_orig, x_retrieve_file__mutmut_mutants, args, kwargs)
    return result 

retrieve_file.__signature__ = _mutmut_signature(x_retrieve_file__mutmut_orig)
x_retrieve_file__mutmut_orig.__name__ = 'x_retrieve_file'


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_orig(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_1(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_2(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            None
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_3(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = None
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_4(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get(None)
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_5(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("XXsizeXX")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_6(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("SIZE")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_7(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None or file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_8(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_9(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size == expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_10(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(None)
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_11(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = None
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_12(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get(None)
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_13(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("XXsha256XX")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_14(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("SHA256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_15(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None or compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_16(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is None and compute_sha256(file_path) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_17(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(None) != expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_18(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) == expected_hash:
        raise ValueError(f"{ERROR_PREFIX} hash SHA-256 invalide pour {file_path}")


# Vérifie la taille et le hash pour sécuriser la matière première
def x_validate_file__mutmut_19(file_path: Path, entry: dict) -> None:
    """Valide les métadonnées d'un fichier téléchargé ou copié."""
    # S'assure que le fichier existe avant toute vérification
    if not file_path.exists():
        raise FileNotFoundError(
            f"{ERROR_PREFIX} fichier manquant après récupération: {file_path}"
        )
    # Compare la taille réelle avec celle attendue si fournie
    expected_size = entry.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        raise ValueError(f"{ERROR_PREFIX} taille inattendue pour {file_path}")
    # Compare le hash réel avec celui attendu si fourni
    expected_hash = entry.get("sha256")
    if expected_hash is not None and compute_sha256(file_path) != expected_hash:
        raise ValueError(None)

x_validate_file__mutmut_mutants : ClassVar[MutantDict] = {
'x_validate_file__mutmut_1': x_validate_file__mutmut_1, 
    'x_validate_file__mutmut_2': x_validate_file__mutmut_2, 
    'x_validate_file__mutmut_3': x_validate_file__mutmut_3, 
    'x_validate_file__mutmut_4': x_validate_file__mutmut_4, 
    'x_validate_file__mutmut_5': x_validate_file__mutmut_5, 
    'x_validate_file__mutmut_6': x_validate_file__mutmut_6, 
    'x_validate_file__mutmut_7': x_validate_file__mutmut_7, 
    'x_validate_file__mutmut_8': x_validate_file__mutmut_8, 
    'x_validate_file__mutmut_9': x_validate_file__mutmut_9, 
    'x_validate_file__mutmut_10': x_validate_file__mutmut_10, 
    'x_validate_file__mutmut_11': x_validate_file__mutmut_11, 
    'x_validate_file__mutmut_12': x_validate_file__mutmut_12, 
    'x_validate_file__mutmut_13': x_validate_file__mutmut_13, 
    'x_validate_file__mutmut_14': x_validate_file__mutmut_14, 
    'x_validate_file__mutmut_15': x_validate_file__mutmut_15, 
    'x_validate_file__mutmut_16': x_validate_file__mutmut_16, 
    'x_validate_file__mutmut_17': x_validate_file__mutmut_17, 
    'x_validate_file__mutmut_18': x_validate_file__mutmut_18, 
    'x_validate_file__mutmut_19': x_validate_file__mutmut_19
}

def validate_file(*args, **kwargs):
    result = _mutmut_trampoline(x_validate_file__mutmut_orig, x_validate_file__mutmut_mutants, args, kwargs)
    return result 

validate_file.__signature__ = _mutmut_signature(x_validate_file__mutmut_orig)
x_validate_file__mutmut_orig.__name__ = 'x_validate_file'


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_orig(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_1(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = None
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_2(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(None)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_3(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_4(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(None)
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_5(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = None
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_6(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(None, entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_7(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, None, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_8(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, None)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_9(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(entry, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_10(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, destination_root)
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_11(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, )
        validate_file(retrieved, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_12(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(None, entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_13(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, None)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_14(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(entry)


# Implémente la boucle principale de récupération contrôlée
def x_fetch_dataset__mutmut_15(source: str, manifest_path: Path, destination_root: Path) -> None:
    """Orchestre la copie/téléchargement et la validation des fichiers."""
    # Charge le manifeste pour connaître les fichiers attendus
    entries = load_manifest(manifest_path)
    # Signale explicitement l'absence d'entrées à traiter
    if not entries:
        raise ValueError(f"{ERROR_PREFIX} aucun fichier listé dans le manifeste")
    # Traite chaque fichier un par un pour isoler les erreurs
    for entry in entries:
        retrieved = retrieve_file(source, entry, destination_root)
        validate_file(retrieved, )

x_fetch_dataset__mutmut_mutants : ClassVar[MutantDict] = {
'x_fetch_dataset__mutmut_1': x_fetch_dataset__mutmut_1, 
    'x_fetch_dataset__mutmut_2': x_fetch_dataset__mutmut_2, 
    'x_fetch_dataset__mutmut_3': x_fetch_dataset__mutmut_3, 
    'x_fetch_dataset__mutmut_4': x_fetch_dataset__mutmut_4, 
    'x_fetch_dataset__mutmut_5': x_fetch_dataset__mutmut_5, 
    'x_fetch_dataset__mutmut_6': x_fetch_dataset__mutmut_6, 
    'x_fetch_dataset__mutmut_7': x_fetch_dataset__mutmut_7, 
    'x_fetch_dataset__mutmut_8': x_fetch_dataset__mutmut_8, 
    'x_fetch_dataset__mutmut_9': x_fetch_dataset__mutmut_9, 
    'x_fetch_dataset__mutmut_10': x_fetch_dataset__mutmut_10, 
    'x_fetch_dataset__mutmut_11': x_fetch_dataset__mutmut_11, 
    'x_fetch_dataset__mutmut_12': x_fetch_dataset__mutmut_12, 
    'x_fetch_dataset__mutmut_13': x_fetch_dataset__mutmut_13, 
    'x_fetch_dataset__mutmut_14': x_fetch_dataset__mutmut_14, 
    'x_fetch_dataset__mutmut_15': x_fetch_dataset__mutmut_15
}

def fetch_dataset(*args, **kwargs):
    result = _mutmut_trampoline(x_fetch_dataset__mutmut_orig, x_fetch_dataset__mutmut_mutants, args, kwargs)
    return result 

fetch_dataset.__signature__ = _mutmut_signature(x_fetch_dataset__mutmut_orig)
x_fetch_dataset__mutmut_orig.__name__ = 'x_fetch_dataset'


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_orig() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_1() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = None
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_2() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description=None)
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_3() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="XXRécupère Physionet dans dataXX")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_4() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="récupère physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_5() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="RÉCUPÈRE PHYSIONET DANS DATA")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_6() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        None, required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_7() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=None, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_8() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help=None
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_9() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_10() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_11() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_12() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "XX--sourceXX", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_13() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--SOURCE", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_14() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=False, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_15() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="XXURL Physionet ou répertoire localXX"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_16() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="url physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_17() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL PHYSIONET OU RÉPERTOIRE LOCAL"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_18() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        None, required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_19() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=None, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_20() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help=None
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_21() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_22() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_23() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_24() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "XX--manifestXX", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_25() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--MANIFEST", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_26() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=False, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_27() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="XXManifeste JSON avec hashes et taillesXX"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_28() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="manifeste json avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_29() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="MANIFESTE JSON AVEC HASHES ET TAILLES"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_30() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument(None, default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_31() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default=None, help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_32() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help=None)
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_33() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument(default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_34() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_35() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", )
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_36() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("XX--destinationXX", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_37() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--DESTINATION", default="data", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_38() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="XXdataXX", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_39() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="DATA", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_40() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="XXRépertoire cibleXX")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_41() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Paramètre l'interface utilisateur en ligne de commande
def x_parse_args__mutmut_42() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data par défaut
    parser.add_argument("--destination", default="data", help="RÉPERTOIRE CIBLE")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()

x_parse_args__mutmut_mutants : ClassVar[MutantDict] = {
'x_parse_args__mutmut_1': x_parse_args__mutmut_1, 
    'x_parse_args__mutmut_2': x_parse_args__mutmut_2, 
    'x_parse_args__mutmut_3': x_parse_args__mutmut_3, 
    'x_parse_args__mutmut_4': x_parse_args__mutmut_4, 
    'x_parse_args__mutmut_5': x_parse_args__mutmut_5, 
    'x_parse_args__mutmut_6': x_parse_args__mutmut_6, 
    'x_parse_args__mutmut_7': x_parse_args__mutmut_7, 
    'x_parse_args__mutmut_8': x_parse_args__mutmut_8, 
    'x_parse_args__mutmut_9': x_parse_args__mutmut_9, 
    'x_parse_args__mutmut_10': x_parse_args__mutmut_10, 
    'x_parse_args__mutmut_11': x_parse_args__mutmut_11, 
    'x_parse_args__mutmut_12': x_parse_args__mutmut_12, 
    'x_parse_args__mutmut_13': x_parse_args__mutmut_13, 
    'x_parse_args__mutmut_14': x_parse_args__mutmut_14, 
    'x_parse_args__mutmut_15': x_parse_args__mutmut_15, 
    'x_parse_args__mutmut_16': x_parse_args__mutmut_16, 
    'x_parse_args__mutmut_17': x_parse_args__mutmut_17, 
    'x_parse_args__mutmut_18': x_parse_args__mutmut_18, 
    'x_parse_args__mutmut_19': x_parse_args__mutmut_19, 
    'x_parse_args__mutmut_20': x_parse_args__mutmut_20, 
    'x_parse_args__mutmut_21': x_parse_args__mutmut_21, 
    'x_parse_args__mutmut_22': x_parse_args__mutmut_22, 
    'x_parse_args__mutmut_23': x_parse_args__mutmut_23, 
    'x_parse_args__mutmut_24': x_parse_args__mutmut_24, 
    'x_parse_args__mutmut_25': x_parse_args__mutmut_25, 
    'x_parse_args__mutmut_26': x_parse_args__mutmut_26, 
    'x_parse_args__mutmut_27': x_parse_args__mutmut_27, 
    'x_parse_args__mutmut_28': x_parse_args__mutmut_28, 
    'x_parse_args__mutmut_29': x_parse_args__mutmut_29, 
    'x_parse_args__mutmut_30': x_parse_args__mutmut_30, 
    'x_parse_args__mutmut_31': x_parse_args__mutmut_31, 
    'x_parse_args__mutmut_32': x_parse_args__mutmut_32, 
    'x_parse_args__mutmut_33': x_parse_args__mutmut_33, 
    'x_parse_args__mutmut_34': x_parse_args__mutmut_34, 
    'x_parse_args__mutmut_35': x_parse_args__mutmut_35, 
    'x_parse_args__mutmut_36': x_parse_args__mutmut_36, 
    'x_parse_args__mutmut_37': x_parse_args__mutmut_37, 
    'x_parse_args__mutmut_38': x_parse_args__mutmut_38, 
    'x_parse_args__mutmut_39': x_parse_args__mutmut_39, 
    'x_parse_args__mutmut_40': x_parse_args__mutmut_40, 
    'x_parse_args__mutmut_41': x_parse_args__mutmut_41, 
    'x_parse_args__mutmut_42': x_parse_args__mutmut_42
}

def parse_args(*args, **kwargs):
    result = _mutmut_trampoline(x_parse_args__mutmut_orig, x_parse_args__mutmut_mutants, args, kwargs)
    return result 

parse_args.__signature__ = _mutmut_signature(x_parse_args__mutmut_orig)
x_parse_args__mutmut_orig.__name__ = 'x_parse_args'


# Expose l'exécution directe du script
def x_main__mutmut_orig() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_1() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = None
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_2() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = None
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_3() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(None)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_4() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = None
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_5() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(None)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_6() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(None, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_7() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, None, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_8() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, None)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_9() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_10() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_11() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, )
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_12() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(None)
        raise SystemExit(1) from error


# Expose l'exécution directe du script
def x_main__mutmut_13() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
        raise SystemExit(None) from error


# Expose l'exécution directe du script
def x_main__mutmut_14() -> None:
    """Point d'entrée CLI pour récupérer les données Physionet."""
    # Récupère les arguments fournis par l'utilisateur
    args = parse_args()
    # Convertit le chemin du manifeste en objet Path pour le valider
    manifest_path = Path(args.manifest)
    # Convertit la destination en Path pour créer l'arborescence
    destination_root = Path(args.destination)
    # Tente la récupération et capture les erreurs pour retour clair
    try:
        fetch_dataset(args.source, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        print(error)
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
    'x_main__mutmut_14': x_main__mutmut_14
}

def main(*args, **kwargs):
    result = _mutmut_trampoline(x_main__mutmut_orig, x_main__mutmut_mutants, args, kwargs)
    return result 

main.__signature__ = _mutmut_signature(x_main__mutmut_orig)
x_main__mutmut_orig.__name__ = 'x_main'


# Active l'exécution uniquement lorsque le script est appelé directement
if __name__ == "__main__":
    main()

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


# Encapsule le chargement du manifeste pour centraliser les contrôles
def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
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
    # Charge le contenu JSON de manière déterministe
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
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


# Calcule la somme SHA-256 pour vérifier l'intégrité demandée
def compute_sha256(file_path: Path) -> str:
    """Retourne le hash SHA-256 d'un fichier donné."""
    # Initialise l'objet de hachage avec l'algorithme voulu
    sha256 = hashlib.sha256()
    # Parcourt le fichier par blocs pour limiter la mémoire utilisée
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    # Fournit la représentation hexadécimale pour la comparaison
    return sha256.hexdigest()


# Choisit automatiquement la méthode (download ou copie locale)
def retrieve_file(source_root: str, entry: dict, destination_root: Path) -> Path:
    """Copie ou télécharge un fichier unique selon la source fournie."""
    # Récupère le chemin relatif attendu depuis le manifeste
    relative_path = Path(entry["path"])
    # Construit le chemin final de destination dans data/raw
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


# Vérifie la taille et le hash pour sécuriser la matière première
def validate_file(file_path: Path, entry: dict) -> None:
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


# Implémente la boucle principale de récupération contrôlée
def fetch_dataset(source: str, manifest_path: Path, destination_root: Path) -> None:
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


# Paramètre l'interface utilisateur en ligne de commande
def parse_args() -> argparse.Namespace:
    """Construit les options CLI attendues pour la récupération Physionet."""
    # Initialise le parseur avec une description explicite
    parser = argparse.ArgumentParser(description="Récupère Physionet dans data/raw")
    # Ajoute la source (URL ou chemin local) comme paramètre requis
    parser.add_argument(
        "--source", required=True, help="URL Physionet ou répertoire local"
    )
    # Demande un manifeste JSON pour activer les contrôles d'intégrité
    parser.add_argument(
        "--manifest", required=True, help="Manifeste JSON avec hashes et tailles"
    )
    # Permet de personnaliser la destination tout en conservant data/raw par défaut
    parser.add_argument("--destination", default="data/raw", help="Répertoire cible")
    # Retourne la structure d'arguments interprétée
    return parser.parse_args()


# Expose l'exécution directe du script
def main() -> None:
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


# Active l'exécution uniquement lorsque le script est appelé directement
if __name__ == "__main__":
    main()

# Gère le découpage des arguments en ligne de commande
# Centralise la collecte des paramètres depuis la CLI
import argparse

# Manipule les chemins pour uniformiser les conversions de paramètres
from pathlib import Path

# Réutilise les primitives de téléchargement et de validation déjà testées
from scripts import fetch_physionet


# Encapsule l'exécution pour fournir des messages cohérents
def prepare_physionet(source: str, manifest: str, destination: str) -> None:
    """Prépare les données Physionet en vérifiant taille et hash."""
    # Prévient l'utilisateur que la récupération démarre avec les paramètres fournis
    print(f"INFO: préparation Physionet depuis '{source}' (manifest: '{manifest}')")
    # Convertit le chemin du manifeste pour sécuriser les accès disque
    manifest_path = Path(manifest)
    # Convertit la destination pour créer l'arborescence si besoin
    destination_path = Path(destination)
    # Exécute la récupération en s'appuyant sur le module spécialisé
    try:
        fetch_physionet.fetch_dataset(source, manifest_path, destination_path)
    except Exception as error:  # noqa: BLE001
        # Signale explicitement l'échec en reprenant le préfixe d'erreur commun
        print(f"{fetch_physionet.ERROR_PREFIX} préparation échouée: {error}")
        # Assure une terminaison CLI avec code d'erreur non nul
        raise SystemExit(1) from error
    # Confirme la disponibilité de l'arborescence cible après validation
    print(f"INFO: données disponibles dans '{destination_path}'")


# Construit l'interface CLI alignée avec les scripts existants
def parse_args() -> argparse.Namespace:
    """Construit les paramètres CLI pour la préparation Physionet."""
    # Initialise un parseur descriptif pour guider l'utilisateur
    parser = argparse.ArgumentParser(
        description="Télécharge ou copie Physionet dans data/raw avec vérification"
    )
    # Ajoute la source obligatoire (URL ou dossier local)
    parser.add_argument(
        "--source",
        required=True,
        help="URL Physionet ou répertoire local contenant les EDF",
    )
    # Ajoute le chemin du manifeste décrivant hashes et tailles attendus
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifeste JSON listant les fichiers à récupérer",
    )
    # Permet de surcharger la destination tout en conservant data/raw par défaut
    parser.add_argument(
        "--destination",
        default="data/raw",
        help="Répertoire cible pour les données brutes",
    )
    # Retourne les arguments interprétés pour la fonction de préparation
    return parser.parse_args()


# Expose le point d'entrée utilisé par le CLI
def main() -> None:
    """Point d'entrée principal du script prepare_physionet."""
    # Récupère les paramètres utilisateur depuis la ligne de commande
    args = parse_args()
    # Délègue la logique métier à la fonction dédiée
    prepare_physionet(args.source, args.manifest, args.destination)


# Active l'exécution directe du script
if __name__ == "__main__":
    main()

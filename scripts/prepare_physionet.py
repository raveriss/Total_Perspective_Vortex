# Centralise la préparation locale du dataset Physionet
# Construit le parseur CLI requis pour orchestrer la récupération
import argparse

# Normalise les chemins d'entrée / sortie
from pathlib import Path

# Garantit l'accès aux opérations de copie/validation existantes
from scripts import fetch_physionet


# Orchestre la récupération complète via fetch_physionet
def prepare_physionet(source_root: str, manifest: str, output_dir: str) -> None:
    """Copie ou télécharge Physionet en validant l'intégrité déclarée."""

    # Normalise le chemin du manifeste pour activer les validations internes
    manifest_path = Path(manifest)
    # Normalise la destination pour préparer l'arborescence cible
    destination_root = Path(output_dir)
    # Crée l'arborescence de destination pour éviter les erreurs de copie
    destination_root.mkdir(parents=True, exist_ok=True)
    # Délègue la récupération en capturant toute erreur pour un message clair
    try:
        fetch_physionet.fetch_dataset(source_root, manifest_path, destination_root)
    except Exception as error:  # noqa: BLE001
        # Signale l'échec de préparation avec un préfixe explicite
        print(f"préparation échouée : {error}")
        # Termine l'exécution avec un code non nul pour la CI
        raise SystemExit(1) from error


# Construit l'interface CLI compatible avec les tests actuels
def parse_args() -> argparse.Namespace:
    """Construit les paramètres CLI pour prepare_physionet."""

    # Initialise un parseur aligné sur fetch_physionet
    parser = argparse.ArgumentParser(
        description=(
            "Prépare localement le dataset Physionet en validant un manifeste"
        ),
    )
    # Racine des données brutes ou URL distante
    parser.add_argument(
        "--source",
        required=True,
        help="Chemin local ou URL HTTP(s) de Physionet",
    )
    # Manifeste JSON décrivant les fichiers attendus
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifeste JSON listant chemins, tailles et hashes",
    )
    # Destination finale (par défaut data via la CI)
    parser.add_argument(
        "--destination",
        required=True,
        help="Répertoire cible pour copier ou télécharger les données",
    )
    # Retourne les arguments parsés
    return parser.parse_args()


# Point d'entrée du script en mode CLI
def main() -> None:
    """Point d'entrée principal pour la préparation Physionet."""

    # Récupère les paramètres fournis en CLI
    args = parse_args()
    # Délègue la logique métier à la fonction dédiée
    prepare_physionet(args.source, args.manifest, args.destination)


# Active l'exécution lorsqu'on lance le module en script
if __name__ == "__main__":
    main()

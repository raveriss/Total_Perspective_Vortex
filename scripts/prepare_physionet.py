# Centralise la préparation locale du dataset Physionet
import argparse

# Normalise les chemins d'entrée / sortie
from pathlib import Path

# Copie les fichiers en préservant les métadonnées
import shutil


# Regroupe la logique de copie des fichiers EDF / events
def prepare_physionet(
    source_root: str,
    subject: str,
    runs: list[str],
    output_dir: str,
) -> None:
    """Copie les fichiers EDF/.event d'un sujet vers un dossier de travail.

    Exemple attendu de layout :

        source_root/
          S001/
            S001R01.edf
            S001R01.edf.event
            ...
        output_dir/
          S001R01.edf
          S001R01.edf.event
          ...

    Ce script NE télécharge PAS les données : il suppose que les EDF
    existent déjà dans source_root.
    """
    # Normalise la racine source (ex: data/raw)
    source_root_path = Path(source_root)
    # Construit le dossier du sujet (ex: data/raw/S001)
    subject_dir = source_root_path / subject
    # Normalise le dossier de sortie (ex: data/S001)
    output_path = Path(output_dir)

    # Crée le dossier cible si nécessaire
    output_path.mkdir(parents=True, exist_ok=True)

    # Log d’info sur la configuration utilisée
    print(
        f"INFO: préparation locale de Physionet pour sujet '{subject}' "
        f"depuis '{subject_dir}' vers '{output_path}'",
    )

    # Vérifie que le dossier du sujet existe
    if not subject_dir.is_dir():
        msg = f"ERREUR: dossier sujet introuvable: '{subject_dir}'"
        raise SystemExit(msg)

    # Parcourt chaque run demandé (R01, R02, ...)
    for run in runs:
        # Construit le préfixe de fichier (ex: S001R01)
        stem = f"{subject}{run}"

        # Pour chaque extension attendue (*.edf, *.edf.event)
        for ext in (".edf", ".edf.event"):
            # Chemin source attendu (ex: data/raw/S001/S001R01.edf)
            src = subject_dir / f"{stem}{ext}"
            # Chemin cible correspondant (ex: data/S001/S001R01.edf)
            dst = output_path / f"{stem}{ext}"

            # Vérifie l'existence du fichier source
            if not src.is_file():
                print(f"ERREUR: fichier manquant: '{src}'")
                raise SystemExit(1)

            # Copie le fichier en conservant les métadonnées
            shutil.copy2(src, dst)
            # Log de confirmation par fichier
            print(f"INFO: copié '{src}' → '{dst}'")

    # Résumé final une fois tous les fichiers copiés
    print("INFO: préparation Physionet terminée avec succès.")


# Construit l'interface CLI compatible avec tes commandes actuelles
def parse_args() -> argparse.Namespace:
    """Construit les paramètres CLI pour prepare_physionet."""
    # Initialise un parseur avec une description claire
    parser = argparse.ArgumentParser(
        description=(
            "Prépare localement le dataset Physionet en copiant les "
            "fichiers EDF et .event d'un sujet vers un dossier de travail."
        ),
    )

    # Racine des données brutes (ex: data/raw)
    parser.add_argument(
        "--source",
        required=True,
        help="Racine des données brutes (ex: data/raw)",
    )

    # Sujet (ex: S001)
    parser.add_argument(
        "--subject",
        required=True,
        help="Identifiant du sujet (ex: S001)",
    )

    # Liste des runs à copier (ex: R01 R02 ... R08)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Liste des runs à copier (ex: R01 R02 R03 ...)",
    )

    # Dossier de sortie (ex: data/S001)
    parser.add_argument(
        "--output",
        required=True,
        help="Dossier de sortie où copier les fichiers du sujet",
    )

    # Retourne les arguments parsés
    return parser.parse_args()


# Point d'entrée du script en mode CLI
def main() -> None:
    """Point d'entrée principal pour la préparation Physionet."""
    # Récupère les paramètres fournis en CLI
    args = parse_args()
    # Délègue la logique métier à la fonction dédiée
    prepare_physionet(args.source, args.subject, args.runs, args.output)


# Active l'exécution lorsqu'on lance le module en script
if __name__ == "__main__":
    main()

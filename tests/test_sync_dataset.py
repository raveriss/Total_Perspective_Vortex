# Oriente le test vers l'exécution réelle du script CLI
# Fournit la sérialisation du manifeste de test
import json
import subprocess

# Centralise la gestion des chemins temporaires
from pathlib import Path


# Vérifie que le script échoue proprement quand la source est absente
def test_sync_dataset_fails_on_missing_file(tmp_path: Path) -> None:
    # Prépare un répertoire source vide pour simuler l'absence de données
    source_root = tmp_path / "source"
    # Crée effectivement le répertoire pour éviter une erreur d'accès
    source_root.mkdir()
    # Décrit un manifeste pointant vers un fichier manquant
    manifest_content = {"files": [{"path": "missing.bin", "size": 1, "sha256": "00"}]}
    # Sérialise le manifeste pour l'injecter dans le script
    manifest_path = tmp_path / "manifest.json"
    # Écrit le manifeste sur disque afin que le CLI puisse le lire
    manifest_path.write_text(json.dumps(manifest_content), encoding="utf-8")
    # Construit la commande CLI avec python pour portabilité
    command = [
        "python",
        "scripts/sync_dataset.py",
        "--source",
        str(source_root),
        "--manifest",
        str(manifest_path),
    ]
    # Exécute le script en capturant stdout et stderr pour validation
    # Désactive la levée automatique pour inspecter manuellement le retour
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    # Confirme que le code de retour signale bien l'échec
    assert result.returncode == 1
    # Vérifie que le message d'erreur mentionne l'absence du fichier
    assert "fichier source absent" in result.stderr


# Vérifie que le script détecte une corruption via le hash
def test_sync_dataset_fails_on_corrupted_file(tmp_path: Path) -> None:
    # Prépare un répertoire source contenant un fichier altéré
    source_root = tmp_path / "source"
    # Crée l'arborescence source pour héberger le fichier
    source_root.mkdir()
    # Crée le fichier corrompu avec un contenu volontairement invalide
    corrupted_file = source_root / "sample.bin"
    # Écrit un contenu simple dont le hash ne correspondra pas au manifeste
    corrupted_file.write_bytes(b"bad")
    # Calcule le hash attendu pour un autre contenu afin d'induire un écart
    expected_hash = "5f16f94e5a0c4d3e6e0682fb9496e91e8847b0d6a0c63798d540e00e5f6a3a7c"
    # Décrit un manifeste aligné sur le fichier mais avec un hash divergent
    manifest_content = {
        "files": [
            {
                "path": "sample.bin",
                "size": corrupted_file.stat().st_size,
                "sha256": expected_hash,
            }
        ]
    }
    # Sérialise le manifeste pour alimenter le script CLI
    manifest_path = tmp_path / "manifest.json"
    # Écrit le manifeste sur disque afin que le CLI puisse le lire
    manifest_path.write_text(json.dumps(manifest_content), encoding="utf-8")
    # Construit la commande CLI pour lancer la synchronisation
    command = [
        "python",
        "scripts/sync_dataset.py",
        "--source",
        str(source_root),
        "--manifest",
        str(manifest_path),
    ]
    # Exécute le script en capturant stdout et stderr pour validation
    # Désactive la levée automatique pour inspecter manuellement le retour
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    # Confirme que le code de retour signale bien l'échec
    assert result.returncode == 1
    # Vérifie que le message d'erreur mentionne le hash invalide
    assert "hash SHA-256 invalide" in result.stderr

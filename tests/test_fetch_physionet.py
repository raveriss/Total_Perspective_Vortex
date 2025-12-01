# Utilise subprocess pour exécuter le script comme un utilisateur final
# Produit rapidement un manifeste JSON pour le test
# Lance des sous-processus pour simuler un usage utilisateur
# Sérialise les manifestes temporaires pour piloter les tests
import json

# Lance des sous-processus pour simuler un usage utilisateur complet
import subprocess

# Fournit l'interpréteur actif pour reproduire l'environnement de test
import sys

# Gère la création de répertoires temporaires isolés
from pathlib import Path

# Encadre les assertions explicites sur les exceptions attendues
import pytest

# Importe le module utilitaire pour l'accès direct aux primitives de téléchargement
from scripts import fetch_physionet


# Vérifie que le script échoue clairement quand les fichiers sont absents
def test_fetch_physionet_fails_without_files(tmp_path: Path) -> None:
    # Prépare un dossier source vide simulant un export incomplet
    source_dir = tmp_path / "source"
    # Crée physiquement le dossier vide pour reproduire l'absence de données
    source_dir.mkdir()
    # Localise le manifeste artificiel utilisé pour ce scénario d'échec
    manifest = tmp_path / "manifest.json"
    # Stocke une entrée fictive afin de provoquer un manque côté source
    manifest.write_text(
        json.dumps({"files": [{"path": "S001/S001R01.edf", "size": 1}]}),
        encoding="utf-8",
    )
    # Définit un répertoire de destination distinct pour le test
    destination_dir = tmp_path / "destination"
    # Sélectionne le chemin absolu du script pour les environnements clonés
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_physionet.py"
    # Lance la commande python pour appeler le script CLI
    result = subprocess.run(
        [
            # Utilise l'interpréteur Python présent dans l'environnement de test
            sys.executable,
            # Cible le nouveau script de récupération Physionet
            str(script_path),
            # Spécifie la source vide pour déclencher l'erreur recherchée
            "--source",
            # Fournit le chemin réel du dossier vide
            str(source_dir),
            # Indique le manifeste construit pour le test
            "--manifest",
            # Passe le chemin vers le fichier JSON temporaire
            str(manifest),
            # Configure le dossier de destination isolé
            "--destination",
            # Ajoute le chemin vers la destination factice
            str(destination_dir),
        ],
        # Capture stdout/stderr pour analyser le message d'échec
        capture_output=True,
        # Force le décodage texte pour faciliter les assertions
        text=True,
        # Empêche subprocess de lever une exception automatique
        check=False,
    )
    # S'assure que l'exécution renvoie un code d'erreur
    assert result.returncode == 1
    # Contrôle la présence d'un message explicite pour guider l'utilisateur
    assert "fichier source absent" in result.stdout


# Vérifie que le chargement du manifeste échoue quand le fichier manque
def test_load_manifest_requires_existing_file(tmp_path: Path) -> None:
    # Construit un chemin inexistant pour simuler un oubli utilisateur
    missing_manifest = tmp_path / "manifest.json"
    # Vérifie que l'appel signale explicitement l'absence du manifeste
    with pytest.raises(FileNotFoundError):
        fetch_physionet.load_manifest(missing_manifest)


# Garantit le rejet des manifestes mal typés
def test_load_manifest_rejects_non_list_files(tmp_path: Path) -> None:
    # Prépare un manifeste avec un champ files invalide pour stresser la validation
    manifest_path = tmp_path / "manifest.json"
    # Sérialise un dictionnaire au lieu d'une liste pour déclencher l'erreur attendue
    manifest_path.write_text(json.dumps({"files": {"path": "bad"}}), encoding="utf-8")
    # Vérifie que la validation signale le format incorrect
    with pytest.raises(ValueError):
        fetch_physionet.load_manifest(manifest_path)


# Confirme que la validation détecte une taille de fichier incorrecte
def test_validate_file_rejects_size_mismatch(tmp_path: Path) -> None:
    # Crée un fichier de deux octets pour provoquer un écart de taille
    file_path = tmp_path / "sample.edf"
    # Écrit un contenu minimal pour matérialiser la différence de taille
    file_path.write_bytes(b"ab")
    # Déclare une taille attendue erronée pour forcer la validation à échouer
    entry = {"size": 1}
    # Vérifie que l'erreur est levée dès la comparaison de taille
    with pytest.raises(ValueError):
        fetch_physionet.validate_file(file_path, entry)


# Confirme que la validation détecte un hash SHA-256 divergent
def test_validate_file_rejects_hash_mismatch(tmp_path: Path) -> None:
    # Crée un fichier stable pour contrôler le hash calculé
    file_path = tmp_path / "sample.edf"
    # Écrit un contenu fixe afin de comparer le hash produit
    file_path.write_bytes(b"content")
    # Fixe un hash attendu incorrect pour forcer l'exception
    entry = {"sha256": "deadbeef"}
    # Vérifie que le hash divergent déclenche une erreur explicite
    with pytest.raises(ValueError):
        fetch_physionet.validate_file(file_path, entry)


# Valide que la copie locale est opérante lorsque la source n'est pas distante
def test_retrieve_file_copies_local_source(tmp_path: Path) -> None:
    # Crée un dossier source peuplé pour simuler une extraction locale
    source_root = tmp_path / "source"
    # Assure l'existence du dossier racine côté source
    source_root.mkdir()
    # Prépare un fichier EDF minimal pour vérifier la copie
    source_file = source_root / "S01" / "run.edf"
    # Crée les répertoires imbriqués nécessaires
    source_file.parent.mkdir(parents=True, exist_ok=True)
    # Dépose un contenu repérable dans le fichier à copier
    source_file.write_text("signal", encoding="utf-8")
    # Définit le dossier de destination où sera copié le fichier
    destination_root = tmp_path / "destination"
    # Spécifie l'entrée de manifeste correspondant au fichier créé
    entry = {"path": "S01/run.edf"}
    # Exécute la récupération en mode copie locale
    retrieved = fetch_physionet.retrieve_file(str(source_root), entry, destination_root)
    # Vérifie que le chemin retourné correspond à la destination attendue
    assert retrieved == destination_root / "S01" / "run.edf"
    # Confirme que le contenu copié reste identique à la source
    assert retrieved.read_text(encoding="utf-8") == "signal"


# Vérifie que les erreurs réseau sont remontées comme ConnectionError
def test_retrieve_file_propagates_network_failure(monkeypatch, tmp_path: Path) -> None:
    # Paramètre une destination valide pour couvrir la branche distante
    destination_root = tmp_path / "destination"
    # Spécifie une entrée minimale pour déclencher le téléchargement
    entry = {"path": "S01/run.edf"}

    # Définit un faux opener qui lève une erreur réseau contrôlée
    class FakeOpener:
        # Bloque toute ouverture en renvoyant une URLError déterministe
        def open(self, *_: object, **__: object) -> None:
            # Propage une URLError pour vérifier la conversion en ConnectionError
            raise fetch_physionet.urllib.error.URLError("boom")

    # Substitue l'opener pour forcer la branche d'erreur réseau
    monkeypatch.setattr(
        fetch_physionet.urllib.request, "build_opener", lambda: FakeOpener()
    )
    # Vérifie que l'erreur réseau se traduit en ConnectionError explicite
    with pytest.raises(ConnectionError):
        fetch_physionet.retrieve_file("https://example.com", entry, destination_root)

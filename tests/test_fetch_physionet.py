# Construit des espaces de noms similaires à argparse pour main()
import argparse

# Stabilise les valeurs de hash pour les vérifications d'intégrité
import hashlib

# Produit rapidement un manifeste JSON pour le test
import json

# Exécute un module comme si le script était lancé depuis le CLI
import runpy

# Fournit l'interpréteur actif pour reproduire l'environnement de test
import sys

# Délimite un flux mémoire pour simuler des téléchargements HTTP sûrs
from io import BytesIO

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
    # Vérifie que la récupération déclenche immédiatement une erreur lisible
    with pytest.raises(FileNotFoundError) as excinfo:
        fetch_physionet.fetch_dataset(str(source_dir), manifest, destination_dir)
    # Confirme que le message pointe explicitement l'absence du fichier source
    assert "fichier source absent" in str(excinfo.value)


# Vérifie que le chargement du manifeste échoue quand le fichier manque
def test_load_manifest_requires_existing_file(tmp_path: Path) -> None:
    # Construit un chemin inexistant pour simuler un oubli utilisateur
    missing_manifest = tmp_path / "manifest.json"
    # Vérifie que l'appel signale explicitement l'absence du manifeste
    with pytest.raises(FileNotFoundError) as error:
        fetch_physionet.load_manifest(missing_manifest)
    # Contrôle que le message d'erreur mentionne le chemin incriminé
    assert str(missing_manifest) in str(error.value)


# Garantit le rejet des manifestes mal typés
def test_load_manifest_rejects_non_list_files(tmp_path: Path) -> None:
    # Prépare un manifeste avec un champ files invalide pour stresser la validation
    manifest_path = tmp_path / "manifest.json"
    # Sérialise un dictionnaire au lieu d'une liste pour déclencher l'erreur attendue
    manifest_path.write_text(json.dumps({"files": {"path": "bad"}}), encoding="utf-8")
    # Vérifie que la validation signale le format incorrect
    with pytest.raises(ValueError) as error:
        fetch_physionet.load_manifest(manifest_path)
    # Confirme que le message cite le champ files pour guider l'utilisateur
    assert "files" in str(error.value)


# Confirme que la validation détecte une taille de fichier incorrecte
def test_validate_file_rejects_size_mismatch(tmp_path: Path) -> None:
    # Crée un fichier de deux octets pour provoquer un écart de taille
    file_path = tmp_path / "sample.edf"
    # Écrit un contenu minimal pour matérialiser la différence de taille
    file_path.write_bytes(b"ab")
    # Déclare une taille attendue erronée pour forcer la validation à échouer
    entry = {"size": 1}
    # Vérifie que l'erreur est levée dès la comparaison de taille
    with pytest.raises(ValueError) as error:
        fetch_physionet.validate_file(file_path, entry)
    # Vérifie que le message mentionne la taille inattendue
    assert "taille inattendue" in str(error.value)


# Confirme que la validation détecte un hash SHA-256 divergent
def test_validate_file_rejects_hash_mismatch(tmp_path: Path) -> None:
    # Crée un fichier stable pour contrôler le hash calculé
    file_path = tmp_path / "sample.edf"
    # Écrit un contenu fixe afin de comparer le hash produit
    file_path.write_bytes(b"content")
    # Fixe un hash attendu incorrect pour forcer l'exception
    entry = {"sha256": "deadbeef"}
    # Vérifie que le hash divergent déclenche une erreur explicite
    with pytest.raises(ValueError) as error:
        fetch_physionet.validate_file(file_path, entry)
    # Vérifie que le message mentionne le hash SHA-256 invalide
    assert "SHA-256" in str(error.value)


# Valide que la copie locale est opérante lorsque la source n'est pas distante
def test_retrieve_file_copies_local_source(tmp_path: Path) -> None:
    # Crée un dossier source peuplé pour simuler une extraction locale
    source_root = tmp_path / "source"
    # Assure l'existence du dossier racine côté source
    source_root.mkdir()
    # Prépare un fichier EDF minimal pour vérifier la copie
    source_file = source_root / "S001" / "run.edf"
    # Crée les répertoires imbriqués nécessaires
    source_file.parent.mkdir(parents=True, exist_ok=True)
    # Dépose un contenu repérable dans le fichier à copier
    source_file.write_text("signal", encoding="utf-8")
    # Définit le dossier de destination où sera copié le fichier
    destination_root = tmp_path / "destination"
    # Spécifie l'entrée de manifeste correspondant au fichier créé
    entry = {"path": "S001/run.edf"}
    # Exécute la récupération en mode copie locale
    retrieved = fetch_physionet.retrieve_file(str(source_root), entry, destination_root)
    # Vérifie que le chemin retourné correspond à la destination attendue
    assert retrieved == destination_root / "S001" / "run.edf"
    # Confirme que le contenu copié reste identique à la source
    assert retrieved.read_text(encoding="utf-8") == "signal"


# Vérifie que les erreurs réseau sont remontées comme ConnectionError
def test_retrieve_file_propagates_network_failure(monkeypatch, tmp_path: Path) -> None:
    # Paramètre une destination valide pour couvrir la branche distante
    destination_root = tmp_path / "destination"
    # Spécifie une entrée minimale pour déclencher le téléchargement
    entry = {"path": "S001/run.edf"}

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
    with pytest.raises(ConnectionError) as error:
        fetch_physionet.retrieve_file("https://example.com", entry, destination_root)
    # Vérifie que le message conserve le préfixe d'erreur attendu
    assert "téléchargement impossible" in str(error.value)


# Refuse les manifestes qui pointent vers un répertoire au lieu d'un fichier
def test_load_manifest_rejects_directory(tmp_path: Path) -> None:
    # Crée un dossier utilisé à tort comme manifeste JSON
    manifest_dir = tmp_path / "manifest_dir"
    # Instancie physiquement le dossier pour déclencher la vérification
    manifest_dir.mkdir()
    # Vérifie que la validation signale l'usage d'un dossier
    with pytest.raises(IsADirectoryError) as error:
        fetch_physionet.load_manifest(manifest_dir)
    # Vérifie que le message pointe le dossier erroné
    assert str(manifest_dir) in str(error.value)


# Accepte un manifeste bien formé avec des entrées dictionnaires
def test_load_manifest_returns_typed_entries(tmp_path: Path) -> None:
    # Crée un chemin pour le manifeste temporaire valide
    manifest_path = tmp_path / "manifest.json"
    # Stocke un contenu JSON minimal comprenant deux fichiers
    manifest_path.write_text(
        json.dumps({"files": [{"path": "a"}, {"path": "b"}]}),
        encoding="utf-8",
    )
    # Récupère la liste typée et vérifie le nombre d'entrées
    entries = fetch_physionet.load_manifest(manifest_path)
    # Confirme que les deux fichiers sont présents dans le manifeste
    assert [entry["path"] for entry in entries] == ["a", "b"]


# Refuse les entrées de manifeste qui ne sont pas des dictionnaires
def test_load_manifest_rejects_non_dict_entries(tmp_path: Path) -> None:
    # Crée un manifeste avec une entrée invalide pour tester la validation
    manifest_path = tmp_path / "manifest.json"
    # Ajoute une chaîne au lieu d'un dictionnaire pour forcer l'échec
    manifest_path.write_text(json.dumps({"files": ["bad-entry"]}), encoding="utf-8")
    # Vérifie que l'appel signale immédiatement l'erreur de type
    with pytest.raises(ValueError):
        fetch_physionet.load_manifest(manifest_path)


# Calcule un hash stable pour sécuriser la validation SHA-256
def test_compute_sha256_matches_expected(tmp_path: Path) -> None:
    # Crée un fichier avec un contenu déterministe
    file_path = tmp_path / "sample.bin"
    # Écrit les données contrôlées pour le calcul du hash
    file_path.write_bytes(b"hashable")
    # Construit le hash attendu via hashlib pour comparaison
    expected = hashlib.sha256(b"hashable").hexdigest()
    # Vérifie que la fonction retourne exactement le hash attendu
    assert fetch_physionet.compute_sha256(file_path) == expected


# Bloque les schémas non HTTP pour les téléchargements distants
def test_retrieve_file_rejects_unsupported_scheme(monkeypatch, tmp_path: Path) -> None:
    # Définit un dossier de destination pour la copie simulée
    destination_root = tmp_path / "destination"
    # Spécifie l'entrée de manifeste minimale pour reproduire la branche distante
    entry = {"path": "S001/run.edf"}

    # Construit un objet minimal imitant le résultat de urlparse
    class Parsed:
        # Expose un schéma interdit pour forcer la ValueError
        scheme = "ftp"

    # Substitue urlparse pour renvoyer un schéma non autorisé malgré un préfixe HTTP
    monkeypatch.setattr(fetch_physionet.urllib.parse, "urlparse", lambda _: Parsed())
    # Vérifie que les schémas non pris en charge déclenchent une ValueError
    with pytest.raises(ValueError):
        fetch_physionet.retrieve_file("https://example.com", entry, destination_root)


# Télécharge un fichier distant via un flux mémoire contrôlé
def test_retrieve_file_downloads_remote_success(monkeypatch, tmp_path: Path) -> None:
    # Prépare le répertoire cible pour stocker le fichier simulé
    destination_root = tmp_path / "destination"
    # Déclare une entrée de manifeste cohérente avec le chemin attendu
    entry = {"path": "S001/run.edf"}

    # Construit un flux binaire imitant la réponse HTTP
    fake_stream = BytesIO(b"payload")

    # Définit un faux opener qui renvoie le flux préparé
    class FakeOpener:
        # Retourne le flux mémoire pour simuler urlopen
        def open(self, *_: object, **__: object) -> BytesIO:
            return fake_stream

    # Injecte le faux opener pour contrôler la branche de téléchargement
    monkeypatch.setattr(
        fetch_physionet.urllib.request, "build_opener", lambda: FakeOpener()
    )
    # Vérifie que le téléchargement écrit correctement le contenu sur disque
    retrieved = fetch_physionet.retrieve_file(
        "https://example.com", entry, destination_root
    )
    # Contrôle que le chemin retourné correspond à la destination attendue
    assert retrieved == destination_root / "S001" / "run.edf"
    # Confirme que le contenu copié provient bien du flux simulé
    assert retrieved.read_bytes() == b"payload"


# Remonte une erreur explicite lorsque la source locale est absente
def test_retrieve_file_requires_existing_local_source(tmp_path: Path) -> None:
    # Déclare un dossier source inexistant pour simuler un oubli utilisateur
    source_root = tmp_path / "missing_source"
    # Spécifie une entrée de manifeste cohérente avec la hiérarchie attendue
    entry = {"path": "S001/run.edf"}
    # Définit un dossier de destination valide pour l'appel
    destination_root = tmp_path / "destination"
    # Vérifie qu'une FileNotFoundError est levée quand le fichier source manque
    with pytest.raises(FileNotFoundError):
        fetch_physionet.retrieve_file(str(source_root), entry, destination_root)


# Signale l'absence de fichier récupéré avant validation
def test_validate_file_requires_presence(tmp_path: Path) -> None:
    # Déclare un chemin inexistant pour simuler un oubli en amont
    missing_file = tmp_path / "ghost.edf"
    # Vérifie que la validation refuse un fichier manquant
    with pytest.raises(FileNotFoundError) as error:
        fetch_physionet.validate_file(missing_file, {})
    # Vérifie que le chemin absent est reporté dans le message
    assert str(missing_file) in str(error.value)


# Valide un fichier existant lorsque la taille et le hash correspondent
def test_validate_file_accepts_matching_metadata(tmp_path: Path) -> None:
    # Crée un fichier avec un contenu contrôlé
    file_path = tmp_path / "sample.edf"
    # Écrit le contenu attendu pour stabiliser les métadonnées
    payload = b"signal"
    # Persiste les octets pour établir la référence de taille et de hash
    file_path.write_bytes(payload)
    # Calcule la taille et le hash attendus pour la validation
    entry = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    # Vérifie qu'aucune exception n'est levée pour un fichier conforme
    fetch_physionet.validate_file(file_path, entry)


# Refuse un manifeste vide pour éviter des exécutions silencieuses
def test_fetch_dataset_rejects_empty_manifest(tmp_path: Path) -> None:
    # Crée un manifeste dont la liste de fichiers est vide
    manifest_path = tmp_path / "manifest.json"
    # Inscrit explicitement un tableau vide pour simuler l'erreur
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    # Définit un dossier de destination pour l'appel simulé
    destination_root = tmp_path / "destination"
    # Vérifie que le pipeline refuse de fonctionner sans fichiers listés
    with pytest.raises(ValueError) as error:
        fetch_physionet.fetch_dataset("/source", manifest_path, destination_root)
    # Confirme que le message signale explicitement l'absence d'entrées
    assert "aucun fichier" in str(error.value)


# Parcourt un manifeste valide et copie les fichiers locaux attendus
def test_fetch_dataset_processes_local_entries(tmp_path: Path) -> None:
    # Prépare un dossier source contenant un fichier EDF factice
    source_root = tmp_path / "source"
    # Instancie le dossier source pour accueillir les données de test
    source_root.mkdir()
    # Crée un fichier identique à celui déclaré dans le manifeste
    source_file = source_root / "S001" / "run.edf"
    # Génère l'arborescence de dossiers attendue
    source_file.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu stable afin de calculer les métadonnées
    payload = b"edf-content"
    # Persiste les octets pour calculer taille et hash
    source_file.write_bytes(payload)
    # Construit le manifeste aligné sur le fichier créé
    manifest_path = tmp_path / "manifest.json"
    # Fixe les métadonnées attendues pour validation
    manifest_entry = {
        "path": "S001/run.edf",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    # Écrit le manifeste avec l'entrée unique validée
    manifest_path.write_text(json.dumps({"files": [manifest_entry]}), encoding="utf-8")
    # Définit le répertoire de destination pour le pipeline
    destination_root = tmp_path / "destination"
    # Exécute le pipeline complet pour couvrir la boucle principale
    fetch_physionet.fetch_dataset(str(source_root), manifest_path, destination_root)
    # Vérifie que le fichier copié existe dans la destination
    copied = destination_root / "S001" / "run.edf"
    # Confirme que le contenu recopié correspond exactement à la source
    assert copied.read_bytes() == payload


# Analyse la ligne de commande pour valider les arguments attendus
def test_parse_args_reads_cli(monkeypatch, tmp_path: Path) -> None:
    # Définit une source simulée pour le téléchargement
    source_root = tmp_path / "source"
    # Crée la source pour refléter un usage réel
    source_root.mkdir()
    # Prépare un chemin de manifeste fictif
    manifest_path = tmp_path / "manifest.json"
    # Inscrit un contenu JSON minimal pour satisfaire l'argument
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    # Configure le dossier de destination pour l'exécution
    destination_root = tmp_path / "destination"
    # Alimente sys.argv pour reproduire un appel utilisateur
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_physionet.py",
            "--source",
            str(source_root),
            "--manifest",
            str(manifest_path),
            "--destination",
            str(destination_root),
        ],
    )
    # Récupère les arguments parsés par argparse
    parsed = fetch_physionet.parse_args()
    # Vérifie que chaque option est correctement transcrite
    assert parsed.source == str(source_root)
    # Contrôle que le manifeste CLI est bien capturé
    assert parsed.manifest == str(manifest_path)
    # Confirme la prise en compte de la destination personnalisée
    assert parsed.destination == str(destination_root)


# Détecte un manifeste dont la racine n'est pas un dictionnaire JSON
def test_load_manifest_rejects_non_mapping(tmp_path: Path) -> None:
    # Stocke un manifeste racine sous forme de liste pour simuler une erreur de format
    manifest_path = tmp_path / "manifest.json"
    # Écrit un tableau vide pour déclencher la validation de type
    manifest_path.write_text(json.dumps([]), encoding="utf-8")
    # Vérifie que la fonction remonte une ValueError explicite
    with pytest.raises(ValueError) as error:
        fetch_physionet.load_manifest(manifest_path)
    # Confirme que le type erroné est mentionné dans le message
    assert "list" in str(error.value)


# Vérifie que l'aide CLI expose la description et les options attendues
def test_parse_args_displays_help(monkeypatch, capsys) -> None:
    # Force la demande d'aide via l'argument --help
    monkeypatch.setattr(sys, "argv", ["fetch_physionet.py", "--help"], raising=True)
    # Capture la sortie standard pour inspecter l'aide générée
    with pytest.raises(SystemExit):
        fetch_physionet.parse_args()
    # Récupère la sortie standard formatée par argparse
    output = capsys.readouterr().out
    # Vérifie que la description personnalisée est bien présente
    assert "Récupère Physionet" in output
    # Confirme que les paramètres source et manifeste sont documentés
    assert "--source" in output
    assert "--manifest" in output


# Vérifie que les options obligatoires sont réellement requises
def test_parse_args_requires_mandatory(monkeypatch) -> None:
    # Supprime tous les arguments pour provoquer un échec argparse
    monkeypatch.setattr(sys, "argv", ["fetch_physionet.py"], raising=True)
    # Vérifie qu'un SystemExit est levé faute d'arguments requis
    with pytest.raises(SystemExit):
        fetch_physionet.parse_args()


# Garantit que le hachage lit les blocs avec la taille attendue
def test_compute_sha256_uses_chunk_size(monkeypatch, tmp_path: Path) -> None:
    # Prépare un fichier fictif pour fournir des octets à hacher
    file_path = tmp_path / "data.bin"
    # Écrit un contenu supérieur à un octet pour déclencher plusieurs lectures
    file_path.write_bytes(b"abc")

    # Enregistre les tailles de lecture observées
    read_sizes: list[int | None] = []

    # Définit un flux factice qui mémorise la taille demandée
    class FakeHandle:
        # Initialise la séquence de blocs à retourner
        def __init__(self) -> None:
            self.chunks = [b"ab", b"c", b""]

        # Enregistre chaque taille demandée et renvoie le bloc suivant
        def read(self, size: int | None) -> bytes:
            read_sizes.append(size)
            return self.chunks.pop(0)

        # Fournit le support du contexte pour imiter Path.open
        def __enter__(self) -> "FakeHandle":
            return self

        # Nettoie la ressource sans action spécifique
        def __exit__(self, *args: object) -> None:
            return None

    # Redirige l'ouverture de fichier vers le flux factice
    monkeypatch.setattr(Path, "open", lambda *_: FakeHandle())
    # Calcule le hash en utilisant le flux instrumenté
    fetch_physionet.compute_sha256(file_path)
    # Vérifie que les lectures utilisent la taille configurée
    assert read_sizes == [8192, 8192, 8192]


# Vérifie que la destination par défaut est appliquée sans argument explicite
def test_parse_args_defaults_destination(monkeypatch, tmp_path: Path) -> None:
    # Prépare une source locale minimale pour respecter les arguments requis
    source_root = tmp_path / "source"
    # Crée physiquement la source pour refléter un appel utilisateur réel
    source_root.mkdir()
    # Positionne un manifeste JSON factice attendu par la CLI
    manifest_path = tmp_path / "manifest.json"
    # Écrit une structure JSON vide mais valide pour satisfaire argparse
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    # Injecte une ligne de commande sans destination pour tester le défaut
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_physionet.py",
            "--source",
            str(source_root),
            "--manifest",
            str(manifest_path),
        ],
    )
    # Parse les arguments simulés via l'interface standard
    parsed = fetch_physionet.parse_args()
    # Vérifie que la destination par défaut pointe vers data
    assert parsed.destination == "data"


# Vérifie que main retourne un code d'erreur lorsque la récupération échoue
def test_main_propagates_fetch_failure(monkeypatch) -> None:
    # Prépare des arguments simulés pour éviter la lecture de sys.argv
    args = type(
        "Args", (), {"source": "SRC", "manifest": "MAN", "destination": "DEST"}
    )()
    # Force parse_args à renvoyer les arguments simulés
    monkeypatch.setattr(fetch_physionet, "parse_args", lambda: args)
    # Simule un échec de récupération pour déclencher SystemExit
    monkeypatch.setattr(
        fetch_physionet,
        "fetch_dataset",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # Vérifie que main propage un code de sortie non nul après l'échec
    with pytest.raises(SystemExit) as exit_info:
        fetch_physionet.main()
    # Vérifie que le code de sortie correspond à l'échec attendu
    assert exit_info.value.code == 1


# Vérifie que main invoque fetch_dataset exactement une fois en succès
def test_main_calls_fetch_once(monkeypatch) -> None:
    # Prépare des arguments simulés pour éviter la lecture de sys.argv
    args = type(
        "Args", (), {"source": "SRC", "manifest": "MAN", "destination": "DEST"}
    )()
    # Force parse_args à renvoyer les arguments simulés
    monkeypatch.setattr(fetch_physionet, "parse_args", lambda: args)
    # Compte les appels pour vérifier le déclenchement de la récupération
    calls = {"count": 0}

    # Simule une récupération réussie en incrémentant le compteur
    def succeed_fetch(source: str, manifest: Path, destination: Path) -> None:
        calls["count"] += 1

    # Injecte la fonction réussie à la place de fetch_dataset
    monkeypatch.setattr(fetch_physionet, "fetch_dataset", succeed_fetch)
    # Exécute main pour vérifier le routage nominal
    fetch_physionet.main()
    # Vérifie que la récupération a été appelée exactement une fois
    assert calls["count"] == 1


# Exécute le script comme module principal pour couvrir le garde __main__
def test_main_runs_under_runpy(monkeypatch, tmp_path: Path) -> None:
    # Prépare une source locale avec un fichier EDF minimal
    source_root = tmp_path / "source"
    # Crée la hiérarchie source pour l'exécution réelle
    source_root.mkdir()
    # Positionne le fichier attendu par le manifeste
    source_file = source_root / "S001" / "run.edf"
    # Génère la structure de dossiers attendue
    source_file.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu stable pour la validation
    payload = b"edf"
    # Persiste le flux pour aligner taille et hash
    source_file.write_bytes(payload)
    # Construit un manifeste aligné sur la source créée
    manifest_path = tmp_path / "manifest.json"
    # Calcule les métadonnées nécessaires au contrôle d'intégrité
    manifest_entry = {
        "path": "S001/run.edf",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    # Écrit le manifeste dans le dossier temporaire
    manifest_path.write_text(json.dumps({"files": [manifest_entry]}), encoding="utf-8")
    # Spécifie la destination où sera recopié le fichier
    destination_root = tmp_path / "destination"
    # Prépare sys.argv pour simuler l'appel direct du script
    argv = [
        "fetch_physionet.py",
        "--source",
        str(source_root),
        "--manifest",
        str(manifest_path),
        "--destination",
        str(destination_root),
    ]
    # Remplace sys.argv afin que argparse lise les arguments simulés
    monkeypatch.setattr(sys, "argv", argv)
    # Localise le chemin absolu du script à exécuter avec runpy
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_physionet.py"
    # Exécute le module avec __name__="__main__" pour franchir le garde final
    runpy.run_path(str(script_path), run_name="__main__")
    # Vérifie que le fichier a bien été copié dans la destination
    copied = destination_root / "S001" / "run.edf"
    # Confirme que le contenu recopié correspond exactement aux octets source
    assert copied.read_bytes() == payload


# Propage un code de sortie non nul en cas d'erreur dans le pipeline
def test_main_exits_on_error(monkeypatch, tmp_path: Path) -> None:
    # Prépare un objet Namespace minimal pour éviter argparse
    fake_args = argparse.Namespace(
        source="/absent",
        manifest=str(tmp_path / "manifest.json"),
        destination=str(tmp_path / "destination"),
    )
    # Force parse_args à renvoyer les arguments simulés
    monkeypatch.setattr(fetch_physionet, "parse_args", lambda: fake_args)

    # Provoque une exception dès l'appel du pipeline pour couvrir l'exception
    def failing_dataset(*_: object) -> None:
        # Lève immédiatement une ValueError pour déclencher la capture
        raise ValueError("boom")

    # Substitue fetch_dataset par la version défaillante
    monkeypatch.setattr(fetch_physionet, "fetch_dataset", failing_dataset)
    # Vérifie que main traduit l'erreur en SystemExit avec code 1
    with pytest.raises(SystemExit) as error:
        fetch_physionet.main()
    # Confirme que le code de sortie correspond à l'échec attendu
    assert error.value.code == 1


# Vérifie que main transmet les arguments bruts à fetch_dataset
def test_main_passes_parsed_arguments(monkeypatch, tmp_path: Path) -> None:
    # Crée un manifeste minimal pour exercer la conversion en Path
    manifest_path = tmp_path / "manifest.json"
    # Inscrit un manifeste vide pour éviter les accès disque en aval
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    # Déclare une destination cible pour vérifier la conversion en Path
    destination_root = tmp_path / "destination"
    # Construit des arguments simulés pour contourner argparse
    fake_args = argparse.Namespace(
        source="http://example.com",
        manifest=str(manifest_path),
        destination=str(destination_root),
    )
    # Mémorise les paramètres reçus pour vérifier l'appel unique
    received: list[tuple[str, Path, Path]] = []
    # Force parse_args à renvoyer l'espace de noms contrôlé
    monkeypatch.setattr(fetch_physionet, "parse_args", lambda: fake_args)

    # Capture les arguments transmis à fetch_dataset pour tuer les mutants
    def recording_dataset(source: str, manifest: Path, destination: Path) -> None:
        # Valide que la source CLI est passée telle quelle
        assert source == fake_args.source
        # Valide que le manifeste est converti en Path attendu
        assert manifest == manifest_path
        # Valide que la destination est convertie en Path attendu
        assert destination == destination_root
        # Archive l'appel pour vérifier l'unicité de l'exécution
        received.append((source, manifest, destination))

    # Remplace fetch_dataset par la variante instrumentée
    monkeypatch.setattr(fetch_physionet, "fetch_dataset", recording_dataset)
    # Exécute main pour déclencher l'appel observé
    fetch_physionet.main()
    # Confirme que fetch_dataset a été invoqué une seule fois
    assert received == [(fake_args.source, manifest_path, destination_root)]


# Contrôle que fetch_dataset appelle bien validate_file pour chaque entrée
def test_fetch_dataset_invokes_validation(monkeypatch, tmp_path: Path) -> None:
    # Crée un manifeste avec deux entrées à traiter
    manifest_path = tmp_path / "manifest.json"
    # Définit les chemins relatifs attendus dans le manifeste
    manifest_entries = [{"path": "S001/a.edf"}, {"path": "S02/b.edf"}]
    # Enregistre le manifeste sur disque pour l'appel testé
    manifest_path.write_text(json.dumps({"files": manifest_entries}), encoding="utf-8")
    # Déclare un dossier de destination pour la copie simulée
    destination_root = tmp_path / "destination"
    # Capture les appels à validate_file pour vérifier la propagation
    validated: list[tuple[Path, dict]] = []

    # Simule retrieve_file en créant un fichier destination pour chaque entrée
    def fake_retrieve(
        source_root: str, entry: dict[str, str], destination: Path
    ) -> Path:
        # Construit le chemin final de la copie factice
        target: Path = destination / Path(entry["path"])
        # Crée l'arborescence cible pour simuler une récupération réussie
        target.parent.mkdir(parents=True, exist_ok=True)
        # Écrit un contenu minimal pour matérialiser le fichier copié
        target.write_bytes(b"ok")
        # Retourne le chemin du fichier copié pour l'étape de validation
        return target

    # Remplace retrieve_file par la version simulée afin d'éviter l'I/O réel
    monkeypatch.setattr(fetch_physionet, "retrieve_file", fake_retrieve)

    # Intercepte validate_file pour enregistrer chaque appel effectué
    def fake_validate(file_path: Path, entry: dict) -> None:
        # Archive le couple chemin/entrée pour vérifier la couverture des deux fichiers
        validated.append((file_path, entry))

    # Substitue validate_file pour observer les chemins reçus
    monkeypatch.setattr(fetch_physionet, "validate_file", fake_validate)
    # Exécute le pipeline complet avec une source locale symbolique
    fetch_physionet.fetch_dataset("/source", manifest_path, destination_root)
    # Vérifie que les deux entrées ont été validées via validate_file
    assert [entry["path"] for _, entry in validated] == ["S001/a.edf", "S02/b.edf"]
    # Contrôle que les chemins passés à validate_file existent réellement
    assert all(path.exists() for path, _ in validated)

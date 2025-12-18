# Sérialise les manifestes pour construire les fichiers temporaires
import json

# Exécute un module comme script pour couvrir le garde main
import runpy

# Manipule les chemins dans les répertoires temporaires isolés
from pathlib import Path

# Capture les exceptions liées à l'arrêt volontaire du CLI
import pytest

# Importe la surface CLI nouvellement exposée
from scripts import prepare_physionet

# Fige le code de sortie standard pour absence d'arguments obligatoires
USAGE_ERROR_CODE = 2


# Vérifie que l'absence de fichiers source provoque un arrêt propre
def test_prepare_physionet_stops_on_missing_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Définit un dossier source inexistant pour simuler un oubli utilisateur
    missing_source = tmp_path / "missing_source"
    # Rédige un manifeste pointant vers un fichier attendu absent
    manifest_path = tmp_path / "manifest.json"
    # Écrit le manifeste minimal attendu par la CLI
    manifest_path.write_text(
        json.dumps({"files": [{"path": "S001/S001R01.edf"}]}),
        encoding="utf-8",
    )
    # Localise la destination de test pour valider le chemin imprimé
    destination_path = tmp_path / "data_raw"
    # Attend une sortie explicite avec un code de retour non nul
    with pytest.raises(SystemExit) as exit_info:
        prepare_physionet.prepare_physionet(
            str(missing_source), str(manifest_path), str(destination_path)
        )
    # Valide que le code de sortie reflète l'échec annoncé
    assert exit_info.value.code == 1
    # Inspecte la sortie standard pour confirmer la mention d'une erreur
    captured = capsys.readouterr()
    # Vérifie la présence d'un préfixe d'erreur cohérent
    assert "préparation échouée" in captured.out


# Vérifie que la corruption des fichiers détectée par le hash arrête le script
def test_prepare_physionet_stops_on_corrupted_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Crée la source locale avec un fichier volontairement invalide
    source_root = tmp_path / "source"
    # Matérialise l'arborescence attendue par le manifeste
    edf_path = source_root / "S001" / "S001R01.edf"
    # Génère les dossiers nécessaires avant l'écriture du fichier
    edf_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu qui ne correspondra pas au hash déclaré
    edf_path.write_bytes(b"bad")
    # Déclare un manifeste avec une taille attendue divergente
    manifest_path = tmp_path / "manifest.json"
    # Stocke un hash irréaliste pour provoquer l'échec de validation
    zero_hash = "0" * 64
    # Écrit le manifeste avec un hash forcé pour provoquer la détection d'erreur
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "S001/S001R01.edf",
                        "size": 10,
                        "sha256": zero_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # Désigne une destination différente pour suivre l'affichage
    destination_path = tmp_path / "data_raw"
    # Attend une fin de programme avec code d'erreur après validation ratée
    with pytest.raises(SystemExit) as exit_info:
        prepare_physionet.prepare_physionet(
            str(source_root), str(manifest_path), str(destination_path)
        )
    # Confirme que le script retourne un code d'échec
    assert exit_info.value.code == 1
    # Capture la sortie affichée pour vérifier le message d'intégrité
    captured = capsys.readouterr()
    # Contrôle la présence du préfixe d'erreur pour les fichiers corrompus
    assert "préparation échouée" in captured.out


# Vérifie que l'analyse des arguments conserve les valeurs fournies
def test_parse_args_preserves_explicit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Déclare une ligne de commande complète simulée
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_physionet",
            "--source",
            "SOURCE",  # pragma: allowlist secret
            "--manifest",
            "MANIFEST",
            "--destination",
            "DESTINATION",
        ],
    )
    # Exécute le parseur pour récupérer les valeurs fournies
    args = prepare_physionet.parse_args()
    # Confirme que la source CLI est correctement relayée
    assert args.source == "SOURCE"  # pragma: allowlist secret
    # Vérifie que le manifeste CLI est bien propagé
    assert args.manifest == "MANIFEST"
    # Garantit que la destination explicite est conservée
    assert args.destination == "DESTINATION"


# Vérifie que chaque option obligatoire est réellement exigée par le parseur
def test_parse_args_enforces_required_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simule une ligne de commande incomplète sans manifeste fourni
    monkeypatch.setattr(
        "sys.argv", ["prepare_physionet", "--source", "SRC", "--destination", "DEST"]
    )
    # Attend un arrêt usage avec un code spécifique dû au flag manquant
    with pytest.raises(SystemExit) as exit_info:
        prepare_physionet.parse_args()
    # Vérifie que le parseur retourne le code usage standard lorsqu'il manque un flag
    assert exit_info.value.code == USAGE_ERROR_CODE


# Vérifie que main relaie l'exception quand la préparation échoue
def test_main_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prépare un objet d'arguments simulant une invocation CLI complète
    args = type(
        "Args",
        (),
        {"source": "SRC", "manifest": "MAN", "destination": "DEST"},
    )()
    # Force parse_args à retourner l'objet simulé sans dépendre de sys.argv
    monkeypatch.setattr(prepare_physionet, "parse_args", lambda: args)
    # Compte les appels pour vérifier que la préparation est bien déclenchée
    calls = {"count": 0}

    # Simule une erreur métier pour vérifier sa propagation directe
    def fail_prepare(source: str, manifest: str, destination: str) -> None:
        calls["count"] += 1
        raise RuntimeError("boom")

    # Injecte la fonction défaillante à la place de la préparation réelle
    monkeypatch.setattr(prepare_physionet, "prepare_physionet", fail_prepare)
    # Vérifie que l'exception remonte telle quelle lorsque la préparation échoue
    with pytest.raises(RuntimeError):
        prepare_physionet.main()
    # Vérifie que la préparation a été tentée exactement une fois
    assert calls["count"] == 1


# Vérifie que main relaie correctement le succès de la préparation
def test_main_calls_prepare_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prépare un objet d'arguments simulant une invocation CLI complète
    args = type(
        "Args",
        (),
        {"source": "SRC", "manifest": "MAN", "destination": "DEST"},
    )()
    # Force parse_args à retourner l'objet simulé sans dépendre de sys.argv
    monkeypatch.setattr(prepare_physionet, "parse_args", lambda: args)
    # Compte les appels pour vérifier que la préparation est bien déclenchée
    calls = {"count": 0}

    # Simule une préparation réussie pour mesurer le comptage
    def succeed_prepare(source: str, manifest: str, destination: str) -> None:
        calls["count"] += 1

    # Injecte la fonction réussie à la place de la préparation réelle
    monkeypatch.setattr(prepare_physionet, "prepare_physionet", succeed_prepare)
    # Exécute main pour vérifier le routage nominal
    prepare_physionet.main()
    # Vérifie que la préparation a été appelée exactement une fois
    assert calls["count"] == 1


# Valide l'exécution nominale de main avec un téléchargement simulé
def test_main_invokes_fetch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prépare des chemins bidons compatibles avec la CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_physionet",
            "--source",
            "SOURCE",  # pragma: allowlist secret
            "--manifest",
            "MANIFEST",
            "--destination",
            "DESTINATION",
        ],
    )
    # Compte le nombre d'appels pour vérifier l'exécution
    calls = {"count": 0}
    # Simule la récupération réussie sans accès réseau
    monkeypatch.setattr(
        prepare_physionet.fetch_physionet,
        "fetch_dataset",
        lambda source, manifest, destination: calls.update(
            {"count": calls["count"] + 1}
        ),
    )
    # Exécute main pour couvrir le flux nominal
    prepare_physionet.main()
    # Confirme que fetch_dataset a été sollicité une fois
    assert calls["count"] == 1


# Couvre l'exécution du garde main lorsque le module est lancé directement
def test_module_guard_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rejoue les paramètres CLI attendus pour l'exécution module
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_physionet",
            "--source",
            "SOURCE",  # pragma: allowlist secret
            "--manifest",
            "MANIFEST",
            "--destination",
            "DESTINATION",
        ],
    )
    # Observe le passage dans main via un compteur partagé
    calls = {"count": 0}
    # Force fetch_dataset à incrémenter le compteur sans I/O
    monkeypatch.setattr(
        prepare_physionet.fetch_physionet,
        "fetch_dataset",
        lambda source, manifest, destination: calls.update(
            {"count": calls["count"] + 1}
        ),
    )
    # Exécute le module comme s'il était lancé en ligne de commande
    runpy.run_module("scripts.prepare_physionet", run_name="__main__")
    # Vérifie que main a été déclenché une fois
    assert calls["count"] == 1

# Utilise subprocess pour exécuter le script comme un utilisateur final
import subprocess
# Gère la création de répertoires temporaires isolés
from pathlib import Path
# Produit rapidement un manifeste JSON pour le test
import json

# Vérifie que le script échoue clairement quand les fichiers sont absents
def test_fetch_physionet_fails_without_files(tmp_path: Path) -> None:
    # Prépare un dossier source vide simulant un export incomplet
    source_dir = tmp_path / "source"
    # Crée physiquement le dossier vide pour reproduire l'absence de données
    source_dir.mkdir()
    # Localise le manifeste artificiel utilisé pour ce scénario d'échec
    manifest = tmp_path / "manifest.json"
    # Stocke une entrée fictive afin de provoquer un manque côté source
    manifest.write_text(json.dumps({"files": [{"path": "S001/S001R01.edf", "size": 1}]}), encoding="utf-8")
    # Définit un répertoire de destination distinct pour le test
    destination_dir = tmp_path / "destination"
    # Lance la commande python pour appeler le script CLI
    result = subprocess.run(
        [
            # Utilise l'interpréteur Python présent dans l'environnement de test
            "python",
            # Cible le nouveau script de récupération Physionet
            "scripts/fetch_physionet.py",
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

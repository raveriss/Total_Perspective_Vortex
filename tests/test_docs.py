"""Tests documentaires pour verrouiller les sections README critiques."""

# Importe pathlib pour lire le README avec un chemin stable
from pathlib import Path


# Vérifie la présence de la section scientifique et des mentions clés
def test_readme_mentions_channels_and_windows() -> None:
    """Assure que le README documente canaux, fenêtres et script associé."""

    # Déduit la racine du dépôt même lorsque mutmut exécute depuis mutants/
    readme_path = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "README.md"
        if candidate.exists():
            readme_path = candidate
            break
    if readme_path is None:
        raise FileNotFoundError("README.md introuvable depuis le test.")
    readme_text = readme_path.read_text(encoding="utf-8")

    # Verrouille la présence du titre de section attendu
    assert "Justification scientifique : canaux & fenêtres temporelles" in readme_text
    # Vérifie que les canaux sensorimoteurs sont cités
    assert "C3" in readme_text
    assert "C4" in readme_text
    assert "Cz" in readme_text
    # Vérifie que les fenêtres temporelles par défaut sont mentionnées
    assert "0.5–2.5" in readme_text
    assert "1.0–3.0" in readme_text
    # Vérifie que le script de visualisation est relié à la justification
    assert "scripts/visualize_raw_filtered.py" in readme_text

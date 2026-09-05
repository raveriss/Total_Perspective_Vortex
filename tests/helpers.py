"""Helpers explicites partagés par les tests de cache numpy."""

# Argparse permet d'inspecter les contrats CLI dans plusieurs suites.
import argparse

# Path garde les répertoires temporaires portables dans les trois suites
from pathlib import Path

# Callable décrit le chargeur train ou predict injecté par chaque test
from typing import Callable

# NumPy fournit le contrat de retour commun aux deux chargeurs de cache
import numpy as np

# Ces structures sont l'implémentation unique déjà partagée par train et predict
from tpv.preprocessing import NpyBuildContext, PreprocessingConfig

# Le type explicite empêche le helper de dépendre d'un module CLI particulier
NpyLoader = Callable[
    # La signature reprend sujet, run et contexte utilisés par les deux chargeurs
    [str, str, NpyBuildContext],
    # Les deux commandes retournent toujours les essais et leurs labels
    tuple[np.ndarray, np.ndarray],
]


# Retrouve une action argparse sans dupliquer l'accès protégé dans chaque test CLI.
def get_parser_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    """Retourne l'action correspondant au `dest` demandé."""

    # Les actions constituent la seule source disponible pour inspecter un parser.
    for action in parser._actions:  # pylint: disable=protected-access
        # La destination est stable même lorsque les libellés d'aide évoluent.
        if action.dest == dest:
            # L'appelant peut maintenant vérifier choix, défaut ou type.
            return action
    # Une assertion produit un diagnostic plus précis qu'un StopIteration.
    raise AssertionError(f"Action argparse introuvable: dest={dest!r}")


# Matérialise les deux fichiers minimaux attendus par les flux EDF des tests.
def write_physionet_stub(raw_dir: Path, subject: str, run: str) -> Path:
    """Crée un EDF et son fichier événement factices puis retourne le chemin EDF."""

    # La structure reproduit exactement la convention utilisée en production.
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Les répertoires temporaires sont créés une seule fois par le helper.
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Un contenu non vide suffit pour franchir le contrôle d'intégrité du test.
    raw_path.write_text("stub")
    # Le fichier événement associé est obligatoire avant tout appel du chargeur.
    raw_path.with_suffix(".edf.event").write_text("stub")
    # Les tests peuvent réutiliser le chemin pour leurs assertions ou monkeypatchs.
    return raw_path


# Construit le contexte minimal réutilisé dans les scénarios train et predict
def build_npy_context(
    data_dir: Path,
    raw_dir: Path,
    eeg_reference: str,
) -> NpyBuildContext:
    """Retourne un contexte de cache avec le prétraitement par défaut."""

    # Une configuration neuve évite tout état mutable partagé entre les tests
    preprocess_config = PreprocessingConfig()
    # Le contexte central conserve exactement le contrat des anciens helpers locaux
    return NpyBuildContext(
        # Ce répertoire reçoit ou fournit les deux tableaux numpy du test
        data_dir=data_dir,
        # Ce répertoire porte les éventuels enregistrements EDF factices
        raw_dir=raw_dir,
        # Cette valeur vérifie la propagation du re-référencement demandé
        eeg_reference=eeg_reference,
        # Cette configuration maintient les valeurs par défaut historiques
        preprocess_config=preprocess_config,
    )


# Injecte le chargeur ciblé pour partager la construction sans coupler les CLIs
def load_data_with_context(
    load_data: NpyLoader,
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
    eeg_reference: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Appelle un chargeur de cache avec un contexte construit explicitement."""

    # Le helper unique garantit le même contexte dans les trois fichiers de tests
    build_context = build_npy_context(data_dir, raw_dir, eeg_reference)
    # Le callback conserve les politiques distinctes de train et de predict
    return load_data(subject, run, build_context)

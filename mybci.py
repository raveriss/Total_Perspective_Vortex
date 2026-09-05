#!/usr/bin/env python3

"""Routeur CLI minimal pour les workflows évalués du projet TPV."""

# Fournit le contrat d'arguments commun aux modes train et predict.
import argparse

# Vérifie la présence de scikit-learn avant de lancer un workflow ML.
import importlib.util

# Isole chaque commande métier dans son module déjà testé.
import subprocess

# Fournit l'interpréteur courant et le code de sortie de la CLI.
import sys

# Regroupe les paramètres transmis à un module sans tuple opaque.
from dataclasses import dataclass, field

# Rend le dossier src importable lors d'un lancement direct du fichier.
from pathlib import Path

# Type les arguments injectés par les tests et le point d'entrée.
from typing import Sequence

# Ajoute src sans dépendre d'une installation editable du projet.
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import tpv.pipeline as tpv_pipeline

# Réutilise les conventions sujet/run partagées par toutes les CLI.
import tpv.utils as tpv_utils

# Réutilise l'unique moteur de score global du dépôt.
from scripts import aggregate_experience_scores

# Les alias de réduction restent acceptés pour préserver l'ancienne interface.
DIMENSIONALITY_ALIASES = set(tpv_pipeline.DIMENSIONALITY_METHODS)
# Deux positionnels distinguent une commande par run des options globales.
MODE_POSITIONAL_COUNT = 2


# Regroupe les valeurs nécessaires à l'appel d'un module spécialisé.
@dataclass
class ModuleCallConfig:
    """Décrit un appel train ou predict sans logique métier."""

    # Identifie le sujet PhysioNet ciblé par la commande.
    subject: str
    # Identifie le run PhysioNet ciblé par la commande.
    run: str
    # Conserve uniquement les options explicitement fournies.
    module_args: list[str] = field(default_factory=list)


# Valide une stratégie ou un alias avant de déléguer au module métier.
def _parse_feature_strategy(value: str) -> str:
    """Normalise une stratégie de features ou de réduction."""

    # Accepte la casse utilisateur sans multiplier les branches plus loin.
    cleaned_value = value.strip().lower()
    # Regroupe exactement les options implémentées et évaluables.
    allowed_values = {"fft", "welch", "wavelet", *DIMENSIONALITY_ALIASES}
    # Refuse tôt une valeur qui échouerait plus tard dans la pipeline.
    if cleaned_value not in allowed_values:
        # argparse affiche ce diagnostic avec l'usage complet de la commande.
        raise argparse.ArgumentTypeError(f"Stratégie invalide: {value!r}")
    # Retourne la forme canonique utilisée par les modules spécialisés.
    return cleaned_value


# Construit uniquement l'interface train/predict exigée par le sujet.
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser des commandes par sujet et par run."""

    # L'usage court rend le chemin de soutenance immédiatement visible.
    parser = argparse.ArgumentParser(
        description="Entraîne ou interroge la pipeline EEG TPV",
        usage="python mybci.py <subject> <run> {train,predict}",
    )
    # Partage la normalisation Sxxx avec train, predict et realtime.
    parser.add_argument("subject", type=tpv_utils.parse_subject)
    # Partage la normalisation Rxx avec train, predict et realtime.
    parser.add_argument("run", type=tpv_utils.parse_run)
    # Limite le routeur aux deux modes obligatoires de la grille.
    parser.add_argument("mode", choices=("train", "predict"))
    # Relaye le choix du classifieur sans reconstruire la pipeline ici.
    parser.add_argument(
        "--classifier",
        choices=tpv_pipeline.CLASSIFIER_CHOICES,
        default=argparse.SUPPRESS,
    )
    # Relaye le scaler optionnel sans logique de transformation locale.
    parser.add_argument(
        "--scaler",
        choices=tpv_pipeline.SCALER_CHOICES,
        default=argparse.SUPPRESS,
    )
    # Conserve les bonus FFT, Welch et wavelet ainsi que les anciens alias.
    parser.add_argument(
        "--feature-strategy",
        type=_parse_feature_strategy,
        default=argparse.SUPPRESS,
    )
    # Relaye explicitement la réduction maison choisie par l'utilisateur.
    parser.add_argument(
        "--dim-method",
        choices=tpv_pipeline.DIMENSIONALITY_METHODS,
        default=argparse.SUPPRESS,
    )
    # Retourne le parser sans exécuter de traitement EEG.
    return parser


# Expose un helper testable sans dépendre directement de sys.argv.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse une invocation train/predict."""

    # Délègue toute validation syntaxique au parser unique ci-dessus.
    return build_parser().parse_args(argv)


# Vérifie la dépendance indispensable avant de créer un sous-processus.
def _ensure_ml_dependencies() -> None:
    """Produit un diagnostic court lorsque scikit-learn manque."""

    # find_spec évite un import coûteux pour cette simple vérification.
    if importlib.util.find_spec("sklearn") is None:
        # Le message indique la commande compatible avec les machines de l'école.
        raise SystemExit(
            "ERROR: dépendance Python manquante: sklearn. "
            "Installez via `uv sync --frozen --all-groups`."
        )


# Traduit les options communes vers les scripts spécialisés existants.
def _build_module_args(args: argparse.Namespace) -> list[str]:
    """Retourne les options explicites destinées à train ou predict."""

    # Prépare une liste vide pour préserver les défauts des modules métier.
    module_args: list[str] = []
    # Récupère la stratégie seulement si l'utilisateur l'a fournie.
    feature_strategy = getattr(args, "feature_strategy", None)
    # Récupère la réduction seulement si l'utilisateur l'a fournie.
    dimensionality_method = getattr(args, "dim_method", None)
    # Les anciens alias deviennent une réduction explicite et non une feature.
    if feature_strategy in DIMENSIONALITY_ALIASES:
        # L'alias conserve la compatibilité sans branche dans train/predict.
        dimensionality_method = feature_strategy
        # Empêche de relayer PCA/CSP comme extracteur fréquentiel.
        feature_strategy = None
    # Relaye le classifieur uniquement lorsqu'il remplace le défaut.
    if hasattr(args, "classifier"):
        # Deux éléments distincts évitent toute interprétation par un shell.
        module_args.extend(["--classifier", str(args.classifier)])
    # Relaye le scaler uniquement lorsqu'il remplace le défaut.
    if hasattr(args, "scaler"):
        # Le module spécialisé interprète lui-même l'alias none.
        module_args.extend(["--scaler", str(args.scaler)])
    # Relaye une véritable extraction fréquentielle si elle est demandée.
    if feature_strategy is not None:
        # Le nom de l'option reste identique dans les deux scripts.
        module_args.extend(["--feature-strategy", feature_strategy])
    # Relaye la réduction finale après résolution d'un éventuel alias.
    if dimensionality_method is not None:
        # La valeur est déjà bornée par argparse ou le jeu d'alias.
        module_args.extend(["--dim-method", dimensionality_method])
    # Retourne une liste directement injectable dans subprocess.
    return module_args


# Lance le vrai point d'entrée sans copier son implémentation dans mybci.
def _call_module(module_name: str, config: ModuleCallConfig) -> int:
    """Exécute tpv.train ou tpv.predict avec les identifiants normalisés."""

    # Une liste d'arguments évite les problèmes d'échappement du shell.
    command = [
        # Réutilise exactement l'environnement Python actif.
        sys.executable,
        # Le mode module garde des imports identiques en local et en CI.
        "-m",
        # Le nom est fourni uniquement par le routeur borné plus bas.
        module_name,
        # Le sujet est déjà normalisé au format Sxxx.
        config.subject,
        # Le run est déjà normalisé au format Rxx.
        config.run,
        # Les options restantes sont déjà séparées et validées.
        *config.module_args,
    ]
    # Le sous-processus transmet naturellement les sorties attendues par la grille.
    completed_process = subprocess.run(command, check=False)
    # Propage le code de sortie pour que Make et la CI détectent les échecs.
    return int(completed_process.returncode)


# Détecte la forme positionnelle attendue sans dupliquer un second parser global.
def _looks_like_mode_invocation(argv: Sequence[str]) -> bool:
    """Distingue subject/run/mode des options du score global."""

    # Deux positionnels suffisent pour reconnaître une tentative train/predict.
    return len(argv) >= MODE_POSITIONAL_COUNT and not argv[0].startswith("-")


# Préserve les alias historiques dans la commande d'agrégation unique.
def _normalize_global_args(argv: Sequence[str]) -> list[str]:
    """Traduit --feature-strategy pca/csp/cssp/svd vers --dim-method."""

    # Copie la séquence pour ne jamais modifier l'entrée de l'appelant.
    normalized_args = list(argv)
    # Recherche l'option historique uniquement si elle est présente.
    if "--feature-strategy" not in normalized_args:
        # Aucun travail n'est nécessaire pour les appels globaux ordinaires.
        return normalized_args
    # Repère la valeur qui suit l'option conformément au contrat argparse.
    option_index = normalized_args.index("--feature-strategy")
    # Laisse argparse produire son diagnostic si la valeur est absente.
    if option_index + 1 >= len(normalized_args):
        # Le moteur unique recevra l'argument incomplet sans le masquer.
        return normalized_args
    # Isole la stratégie afin de détecter les alias de réduction.
    strategy = normalized_args[option_index + 1].lower()
    # Les extracteurs réels sont déjà compris par l'agrégateur.
    if strategy not in DIMENSIONALITY_ALIASES:
        # Conserve fft, welch ou wavelet sans transformation.
        return normalized_args
    # Remplace seulement le nom d'option, en conservant sa valeur.
    normalized_args[option_index] = "--dim-method"
    # Retourne une invocation comprise par le moteur de score unique.
    return normalized_args


# Route les deux chemins publics sans contenir de logique EEG ou de scoring.
def main(argv: Sequence[str] | None = None) -> int:
    """Exécute train/predict ou l'agrégation globale en absence de positionnels."""

    # Copie sys.argv pour rendre le comportement identique sous tests et en CLI.
    provided_args = list(argv) if argv is not None else list(sys.argv[1:])
    # Les deux premiers positionnels identifient une commande par run.
    if _looks_like_mode_invocation(provided_args):
        # Le parser produit l'usage correct si le mode manque ou est invalide.
        parsed_args = parse_args(provided_args)
        # Vérifie l'environnement avant de lancer le workflow coûteux.
        _ensure_ml_dependencies()
        # Prépare les paramètres communs sans reconstruire la pipeline.
        call_config = ModuleCallConfig(
            # Conserve le sujet normalisé par argparse.
            subject=parsed_args.subject,
            # Conserve le run normalisé par argparse.
            run=parsed_args.run,
            # Traduit seulement les options explicitement fournies.
            module_args=_build_module_args(parsed_args),
        )
        # Le choix est borné à train/predict par le parser.
        module_name = f"tpv.{parsed_args.mode}"
        # Retourne directement le résultat du module spécialisé.
        return _call_module(module_name, call_config)
    # Sans positionnels, délègue tous les calculs au moteur d'agrégation unique.
    return aggregate_experience_scores.main(_normalize_global_args(provided_args))


# Protège les imports tout en exposant un exécutable autonome à l'évaluateur.
if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    # Transforme le résultat du routeur en code de sortie du processus.
    raise SystemExit(main())

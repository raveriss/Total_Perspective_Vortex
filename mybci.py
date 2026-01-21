#!/usr/bin/env python3

"""Interface CLI pour piloter les workflows d'entraînement et de prédiction."""

# Préserve argparse pour parser les options CLI avec validation
import argparse

# Préserve importlib pour charger des modules sans import direct
import importlib

# Préserve importlib pour détecter des dépendances installées
import importlib.util

# Préserve subprocess pour lancer les modules en sous-processus isolés
import subprocess

# Préserve sys pour identifier l'interpréteur courant
import sys

# Préserve dataclass pour regrouper les paramètres du pipeline
from dataclasses import dataclass

# Facilite la gestion portable des chemins de données et artefacts
from pathlib import Path

# Centralise la moyenne arithmétique pour agréger les accuracies
from statistics import mean

# Garantit l'accès aux séquences typées pour mypy
from typing import Iterable, Mapping, Sequence

# Fournit une barre de progression compacte pendant l'évaluation globale
from tqdm import tqdm

# Assure l'accès à tpv via src lors d'une exécution locale
sys.path.append(str(Path(__file__).resolve().parent / "src"))


# Normalise un identifiant brut en appliquant un préfixe standard
def _normalize_identifier(value: str, prefix: str, width: int, label: str) -> str:
    """Normalise un identifiant pour respecter le format Physionet."""

    # Nettoie la valeur reçue pour éviter des espaces parasites
    cleaned_value = value.strip()
    # Refuse une valeur vide pour éviter un identifiant incomplet
    if not cleaned_value:
        # Signale une valeur vide pour forcer la correction côté CLI
        raise argparse.ArgumentTypeError(f"{label} vide")
    # Récupère le premier caractère pour détecter un préfixe explicite
    first_char = cleaned_value[0]
    # Déduit si l'utilisateur a fourni le préfixe attendu
    has_prefix = first_char.upper() == prefix.upper()
    # Extrait la portion numérique selon la présence du préfixe
    numeric_part = cleaned_value[1:] if has_prefix else cleaned_value
    # Refuse les valeurs non numériques pour garantir un ID valide
    if not numeric_part.isdigit():
        # Signale l'identifiant invalide pour guider l'utilisateur
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Convertit en entier pour normaliser les zéros initiaux
    numeric_value = int(numeric_part)
    # Refuse les index non positifs pour respecter la base Physionet
    if numeric_value < 1:
        # Signale l'identifiant non valide pour arrêter le parsing
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Reconstruit l'identifiant normalisé avec le padding attendu
    return f"{prefix}{numeric_value:0{width}d}"


# Normalise un identifiant de sujet pour la CLI mybci
def _parse_subject(value: str) -> str:
    """Normalise un identifiant de sujet en format Sxxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="S", width=3, label="Sujet")


# Normalise un identifiant de run pour la CLI mybci
def _parse_run(value: str) -> str:
    """Normalise un identifiant de run en format Rxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="R", width=2, label="Run")


# Regroupe les specs pour éviter les répétitions dans build_parser
_ARGUMENT_SPECS: tuple[tuple[tuple[str, ...], dict], ...] = (
    (
        ("subject",),
        {
            "help": "Identifiant du sujet (ex: 4)",
            "type": _parse_subject,
        },
    ),
    (
        ("run",),
        {
            "help": "Identifiant du run (ex: 14)",
            "type": _parse_run,
        },
    ),
    (
        ("mode",),
        {
            "choices": ("train", "predict"),
            "help": "Choix du pipeline à lancer",
        },
    ),
)


# Valide la présence d'une dépendance critique avant exécution
def _require_dependency(module_name: str, install_hint: str) -> None:
    """Interrompt l'exécution si une dépendance Python manque."""

    # Interroge l'environnement pour vérifier la disponibilité du module
    if importlib.util.find_spec(module_name) is None:
        # Prépare l'entête du message d'erreur utilisateur
        message = f"ERROR: dépendance Python manquante: {module_name}."
        # Ajoute la consigne d'installation pour rendre l'action explicite
        message = f"{message} {install_hint}"
        # Interrompt avec un message actionnable pour l'utilisateur
        raise SystemExit(message)


# Centralise le message d'installation pour les dépendances ML
def _ensure_ml_dependencies() -> None:
    """Vérifie les dépendances nécessaires au ML temps réel."""

    # Prépare un rappel d'installation cohérent avec Poetry
    hint = "Installez via `poetry install --with dev` (ou `poetry install`)."
    # Vérifie la présence de scikit-learn avant les imports ML
    _require_dependency("sklearn", hint)


# Importe tpv.predict uniquement quand il est nécessaire
def _load_predict_module():
    """Charge le module predict après validation des dépendances."""

    # Vérifie que les dépendances ML sont disponibles
    _ensure_ml_dependencies()
    # Charge le module predict au dernier moment pour éviter les crashes
    tpv_predict = importlib.import_module("tpv.predict")

    # Retourne le module chargé pour l'appelant
    return tpv_predict


# Centralise les options nécessaires pour invoquer un module TPV
@dataclass
class ModuleCallConfig:
    """Conteneur des paramètres transmis aux modules train/predict."""

    # Identifie le sujet cible pour charger les données correspondantes
    subject: str
    # Identifie le run cible pour charger la bonne session
    run: str


# Centralise les répertoires nécessaires pendant l'évaluation globale
@dataclass
class EvaluationPaths:
    """Conteneur des chemins racine utilisés pendant l'évaluation."""

    # Stocke le chemin vers les données prétraitées pour les runs
    data_root: Path
    # Stocke le chemin vers les artefacts entraînés pour les runs
    artifacts_root: Path
    # Stocke le chemin vers les fichiers EDF bruts pour les runs
    raw_root: Path


# Construit la ligne de commande pour invoquer un module TPV
def _call_module(module_name: str, config: ModuleCallConfig) -> int:
    """Invoke un module TPV en ajoutant les options du pipeline."""

    # Initialise la commande avec l'interpréteur courant et le module ciblé
    command: list[str] = [
        sys.executable,
        "-m",
        module_name,
        config.subject,
        config.run,
    ]
    # Exécute la commande en capturant le code retour sans lever d'exception
    completed = subprocess.run(command, check=False)
    # Retourne le code retour pour propagation à l'appelant principal
    return completed.returncode


# Définit la structure décrivant un protocole expérimental
@dataclass
class ExperimentDefinition:
    """Associe un identifiant d'expérience au run correspondant."""

    # Identifie la position de l'expérience dans la séquence requise
    index: int
    # Associe l'expérience au run Physionet à évaluer
    run: str


# Construit la liste des six expériences décrites dans le sujet
def _build_default_experiments() -> list[ExperimentDefinition]:
    """Expose les six expériences demandées par la consigne."""

    # Mappe chaque expérience à un run Physionet pour l'évaluation
    return [
        # Explore le run R03 pour l'expérience 0
        ExperimentDefinition(index=0, run="R03"),
        # Explore le run R04 pour l'expérience 1
        ExperimentDefinition(index=1, run="R04"),
        # Explore le run R05 pour l'expérience 2
        ExperimentDefinition(index=2, run="R05"),
        # Explore le run R06 pour l'expérience 3
        ExperimentDefinition(index=3, run="R06"),
        # Explore le run R07 pour l'expérience 4
        ExperimentDefinition(index=4, run="R07"),
        # Explore le run R08 pour l'expérience 5
        ExperimentDefinition(index=5, run="R08"),
    ]


# Convertit un numéro de sujet numérique en identifiant Physionet
def _subject_identifier(subject_index: int) -> str:
    """Retourne l'identifiant Sxxx attendu dans les répertoires."""

    # Formate le numéro sur trois chiffres en préfixant le S imposé
    return f"S{subject_index:03d}"


# Calcule l'accuracy pour un couple (expérience, sujet)
def _evaluate_experiment_subject(
    experiment: ExperimentDefinition,
    subject_index: int,
    data_dir: Path,
    artifacts_dir: Path,
    raw_dir: Path,
) -> float:
    """Évalue un sujet sur le run associé à une expérience donnée."""

    # Construit l'identifiant complet du sujet pour les chemins disque
    subject = _subject_identifier(subject_index)
    # Charge le module predict seulement au moment de l'évaluation
    tpv_predict = _load_predict_module()
    # Exécute evaluate_run sur le run associé à l'expérience
    result = tpv_predict.evaluate_run(
        subject,
        experiment.run,
        data_dir,
        artifacts_dir,
        raw_dir,
    )
    # Convertit l'accuracy en float natif pour l'agrégation
    return float(result["accuracy"])


# Calcule la moyenne d'accuracies pour une séquence fournie
def _safe_mean(values: Iterable[float]) -> float:
    """Retourne 0.0 si la séquence est vide pour sécuriser l'affichage."""

    # Convertit l'itérable en liste pour gérer la longueur et le calcul
    measurements = list(values)
    # Retourne 0.0 si aucune valeur n'est disponible
    if not measurements:
        # Force une moyenne nulle pour éviter ZeroDivisionError
        return 0.0
    # Calcule la moyenne arithmétique standard
    return mean(measurements)


# Recense les sujets disposant d'un modèle entraîné pour un run donné
def _subjects_with_available_model(run: str, artifacts_root: Path) -> list[int]:
    """Liste les indices de sujets dont le modèle est présent sur disque."""

    # Prépare une collection ordonnée pour les indices extraits
    subjects: list[int] = []
    # Parcourt les fichiers model.joblib correspondant au run demandé
    for model_path in artifacts_root.glob(f"S*/{run}/model.joblib"):
        # Identifie le dossier sujet à partir du chemin du modèle
        subject_dir = model_path.parent.parent
        # Vérifie que le nom de dossier respecte le préfixe attendu
        if not subject_dir.name.startswith("S"):
            # Ignore les dossiers inattendus pour éviter des erreurs de parsing
            continue
        try:
            # Convertit la partie numérique du nom en entier pour l'itération
            subject_index = int(subject_dir.name[1:])
        except ValueError:
            # Ignore les dossiers mal nommés pour maintenir la robustesse
            continue
        # Ajoute l'indice extrait pour inclure le sujet dans l'évaluation
        subjects.append(subject_index)
    # Trie les indices pour respecter l'ordre croissant des sujets
    subjects.sort()
    # Retourne la liste triée pour itérer dans un ordre reproductible
    return subjects


# Prépare la disponibilité des modèles par run avant l'évaluation globale
def _collect_run_availability(
    experiments: Sequence[ExperimentDefinition],
    expected_subjects: Sequence[str],
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Associe chaque run aux sujets attendus pour déclencher l'auto-train."""

    # Prépare un cache pour associer chaque run aux sujets parcourus
    available_subjects_by_run: dict[str, list[int]] = {}
    # Prépare un relevé vide car l'auto-train doit combler les absences
    missing_models_by_run: dict[str, list[str]] = {}
    # Parcourt chaque expérience pour initialiser la liste des sujets
    for experiment in experiments:
        # Ignore le run déjà traité pour éviter les doublons
        if experiment.run in available_subjects_by_run:
            # Passe au run suivant dès que le cache contient le run
            continue
        # Convertit les identifiants Sxxx en indices numériques exploitables
        subject_indices = [int(subject[1:]) for subject in expected_subjects]
        # Associe tous les sujets au run pour laisser evaluate_run entraîner
        available_subjects_by_run[experiment.run] = subject_indices
        # Marque l'absence de modèles manquants grâce à l'auto-train
        missing_models_by_run[experiment.run] = []
    # Retourne les deux structures pour l'évaluation globale
    return available_subjects_by_run, missing_models_by_run


# Évalue chaque expérience en accumulant les résultats et les absences
def _evaluate_experiments(
    experiments: Sequence[ExperimentDefinition],
    available_subjects_by_run: Mapping[str, Sequence[int]],
    paths: EvaluationPaths,
    progress: tqdm | None = None,
) -> tuple[dict[int, list[float]], list[str], list[ExperimentDefinition]]:
    """Exécute les évaluations et retourne les scores et manquants."""

    # Prépare le stockage des accuracies par expérience
    per_experiment_scores: dict[int, list[float]] = {
        # Initialise la collection d'accuracies pour chaque expérience
        exp.index: []
        for exp in experiments
    }
    # Prépare la liste des sujets ou runs introuvables lors des calculs
    missing_entries: list[str] = []
    # Prépare la liste des expériences sans modèle pour les ignorer
    skipped_experiments: list[ExperimentDefinition] = []
    # Parcourt chaque expérience demandée
    for experiment in experiments:
        # Récupère la liste des sujets disposant d'un modèle pour ce run
        available_subjects = list(available_subjects_by_run.get(experiment.run, []))
        # Informe l'utilisateur si aucun modèle n'est disponible pour ce run
        if not available_subjects:
            # Signale que l'expérience sera ignorée faute de modèle présent
            print(
                "AVERTISSEMENT: aucun modèle disponible pour "
                f"{experiment.run}, expérience {experiment.index} ignorée"
            )
            # Archive l'expérience ignorée pour le résumé des moyennes
            skipped_experiments.append(experiment)
            # Passe à l'expérience suivante pour éviter une boucle vide
            continue
        # Parcourt l'ensemble des sujets disposant d'un modèle
        for subject_index in available_subjects:
            # Évalue le sujet courant sur l'expérience en cours
            try:
                # Calcule l'accuracy en rechargeant le modèle entraîné
                accuracy = _evaluate_experiment_subject(
                    experiment,
                    subject_index,
                    paths.data_root,
                    paths.artifacts_root,
                    paths.raw_root,
                )
            except FileNotFoundError as error:
                # Informe l'utilisateur qu'un prérequis manque pour ce run
                print(f"AVERTISSEMENT: {error}")
                # Calcule l'identifiant du sujet manquant pour le récapitulatif
                subject = _subject_identifier(subject_index)
                # Ajoute l'entrée manquante pour un récapitulatif final
                missing_entries.append(f"{subject}:{experiment.run}")
                # Ignore ce sujet pour poursuivre l'exploration globale
                continue
            finally:
                # Actualise la barre de progression lorsqu'elle est activée
                if progress is not None:
                    # Incrémente la progression d'un sujet évalué ou tenté
                    progress.update(1)
            # Formate l'index sujet en trois chiffres pour la sortie demandée
            subject_label = f"{subject_index:03d}"
            # Prépare le préfixe pour éviter une ligne trop longue
            prefix = f"experiment {experiment.index}: subject {subject_label}: "
            # Prépare le suffixe avec l'accuracy formatée
            suffix = f"accuracy = {accuracy:.4f}"
            # Affiche l'accuracy par expérience et sujet comme dans l'exemple
            print(f"{prefix}{suffix}")
            # Stocke l'accuracy pour le calcul des moyennes
            per_experiment_scores[experiment.index].append(accuracy)
    # Retourne les résultats et les expériences ignorées
    return per_experiment_scores, missing_entries, skipped_experiments


# Affiche les moyennes par expérience et retourne la moyenne globale
def _print_experiment_means(
    experiments: Sequence[ExperimentDefinition],
    per_experiment_scores: Mapping[int, Sequence[float]],
) -> float:
    """Calcule et affiche les moyennes d'accuracy par expérience."""

    # Affiche l'entête du bloc de moyennes par expérience
    print("\nMean accuracy of the six different experiments for all 109 subjects:")
    # Parcourt chaque expérience pour calculer sa moyenne
    for experiment in experiments:
        # Extrait les scores accumulés pour l'expérience courante
        experiment_scores = per_experiment_scores[experiment.index]
        # Contrôle la disponibilité d'artefacts avant de calculer la moyenne
        if not experiment_scores:
            # Mentionne explicitement l'absence d'artefacts pour l'expérience
            print(f"experiment {experiment.index}:\t\taccuracy = N/A (skipped)")
            # Passe au run suivant pour éviter une moyenne vide
            continue
        # Calcule la moyenne de l'expérience courante
        experiment_mean = _safe_mean(experiment_scores)
        # Affiche la moyenne alignée sur l'exemple fourni
        print(f"experiment {experiment.index}:\t\taccuracy = " f"{experiment_mean:.4f}")
    # Calcule la moyenne globale des six expériences
    global_mean = _safe_mean(
        # Agrège uniquement les expériences disposant d'artefacts
        _safe_mean(per_experiment_scores[exp.index])
        for exp in experiments
        if per_experiment_scores[exp.index]
    )
    # Affiche la moyenne globale demandée par la consigne
    print(f"\nMean accuracy of 6 experiments: {global_mean:.4f}")
    # Retourne la moyenne pour réutilisation éventuelle
    return global_mean


# Affiche les messages d'alerte pour guider l'utilisateur
def _report_missing_artifacts(
    missing_entries: Sequence[str],
    missing_models_by_run: Mapping[str, Sequence[str]],
    skipped_experiments: Sequence[ExperimentDefinition],
    expected_subject_count: int,
) -> None:
    """Émet les avertissements sur les données et modèles manquants."""

    # Vérifie si des données sont manquantes pour informer l'utilisateur
    if missing_entries:
        # Résume le volume d'entrées absentes pour déclencher une action
        print(
            "AVERTISSEMENT: certaines données EDF ou .npy sont manquantes. "
            f"Couples sujet/run concernés: {len(missing_entries)}. "
            "Téléchargez les EDF dans data ou regénérez les .npy."
        )
        # Affiche un aperçu des premières références manquantes pour guider
        print("Premiers manquants: " + ", ".join(missing_entries[:10]))
    # Vérifie s'il manque des modèles entraînés pour certains runs
    if any(missing_models_by_run.values()):
        # Informe l'utilisateur qu'il manque des artefacts pour plusieurs sujets
        print(
            "AVERTISSEMENT: certains modèles entraînés sont absents. "
            "Générez ou copiez les artifacts manquants pour compléter l'évaluation."
        )
        # Identifie les runs totalement dépourvus de modèles pour prioriser les actions
        fully_missing_runs = [
            run
            for run, subjects in sorted(missing_models_by_run.items())
            if subjects and len(subjects) == expected_subject_count
        ]
        # Met en avant les runs sans modèles pour débloquer la génération
        if fully_missing_runs:
            print("Runs sans aucun modèle disponible: " + ", ".join(fully_missing_runs))
        # Parcourt les runs pour afficher un extrait des sujets à compléter
        for run, subjects in sorted(missing_models_by_run.items()):
            # Ignore l'affichage si aucun modèle ne manque pour ce run
            if not subjects:
                # Passe au run suivant lorsqu'il est complet
                continue
            # Affiche le nombre total de modèles manquants pour ce run
            print(
                f"Run {run}: modèles manquants pour {len(subjects)} sujets "
                f"(exemples: {', '.join(subjects[:5])})"
            )
        # Aide l'utilisateur en rappelant la commande de génération d'artefacts
        print(
            "Pour générer un modèle manquant, lancez par exemple :\n"
            "  poetry run python scripts/train.py S001 R04 --feature-strategy fft "
            "--dim-method pca"
        )
    # Vérifie si certaines expériences ont été ignorées pour le calcul global
    if skipped_experiments:
        # Résume les expériences ignorées pour clarifier le global_mean affiché
        skipped_labels = ", ".join(
            f"{exp.index} ({exp.run})" for exp in skipped_experiments
        )
        # Invite l'utilisateur à générer les artefacts avant de relancer
        print(
            "AVERTISSEMENT: les expériences suivantes ont été ignorées "
            f"faute de modèles: {skipped_labels}. "
            "Générez les artefacts correspondants pour obtenir une moyenne "
            "complète."
        )


# Parcourt les 6 expériences et les 109 sujets en affichant les accuracies
def _run_global_evaluation(
    experiments: Sequence[ExperimentDefinition] | None = None,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> int:
    """Exécute la boucle d'évaluation globale décrite dans le sujet."""

    # Vérifie les dépendances ML avant l'évaluation globale
    _ensure_ml_dependencies()
    # Utilise les expériences par défaut si aucune liste n'est fournie
    experiment_definitions = list(experiments or _build_default_experiments())
    # Normalise les chemins racine de données pour les appels descendants
    data_root = data_dir or Path("data")
    # Normalise le répertoire d'artefacts pour les modèles entraînés
    artifacts_root = artifacts_dir or Path("artifacts")
    # Normalise le répertoire des EDF bruts désormais stockés dans data/
    raw_root = raw_dir or Path("data")
    # Construit la liste des identifiants attendus pour les 109 sujets
    expected_subjects = [_subject_identifier(idx) for idx in range(1, 110)]
    # Calcule la disponibilité des modèles pour chaque run
    available_subjects_by_run, missing_models_by_run = _collect_run_availability(
        experiment_definitions, expected_subjects
    )
    # Calcule le nombre total de sujets évalués pour calibrer la progression
    total_subjects = sum(
        # Calcule le nombre de sujets associés à chaque run
        len(available_subjects_by_run.get(experiment.run, []))
        # Parcourt toutes les expériences à traiter
        for experiment in experiment_definitions
    )

    # Exécute les évaluations et collecte les résultats
    per_experiment_scores, missing_entries, skipped_experiments = _evaluate_experiments(
        experiment_definitions,
        available_subjects_by_run,
        EvaluationPaths(
            # Fournit le répertoire racine des données prétraitées
            data_root=data_root,
            # Fournit le répertoire racine des artefacts entraînés
            artifacts_root=artifacts_root,
            # Fournit le répertoire racine des fichiers EDF bruts
            raw_root=raw_root,
        ),
    )

    # Calcule et affiche les moyennes par expérience
    _print_experiment_means(experiment_definitions, per_experiment_scores)
    # Émet un récapitulatif des artefacts manquants
    _report_missing_artifacts(
        missing_entries,
        missing_models_by_run,
        skipped_experiments,
        len(expected_subjects),
    )
    # Retourne 0 pour signaler le succès global
    return 0


# Imprime les prédictions epoch par epoch dans un format compact
def _print_epoch_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    accuracy: float,
) -> None:
    """Affiche les prédictions détaillées comme dans l'exemple mybci."""

    # Affiche l'en-tête décrivant les colonnes
    print("epoch nb: [prediction] [truth] equal?")
    # Calcule la largeur minimale pour l'index d'epoch
    n_epochs = len(y_true)
    # Utilise au moins deux chiffres pour mimer l'exemple fourni
    index_width = max(2, len(str(max(n_epochs - 1, 0))))
    # Parcourt chaque paire vérité terrain / prédiction
    for idx, (pred, truth) in enumerate(zip(y_pred, y_true, strict=True)):
        # Calcule si la prédiction correspond à la vérité terrain
        is_equal = bool(int(pred) == int(truth))
        # Affiche la ligne formatée pour l'epoch courante
        print(
            f"epoch {idx:0{index_width}d}: " f"[{int(pred)}] [{int(truth)}] {is_equal}"
        )
    # Affiche l'accuracy globale formatée sur quatre décimales
    print(f"Accuracy: {accuracy:.4f}")


# Centralise le pattern add_argument pour appliquer une liste de specs
def _add_argument_specs(parser: argparse.ArgumentParser) -> None:
    """Ajoute les arguments déclarés dans _ARGUMENT_SPECS au parser."""
    for flags, kwargs in _ARGUMENT_SPECS:
        parser.add_argument(*flags, **kwargs)


# Construit le parser CLI avec toutes les options du pipeline
def build_parser() -> argparse.ArgumentParser:
    """Construit l'argument parser pour mybci."""
    parser = argparse.ArgumentParser(
        description="Pilote un workflow d'entraînement ou de prédiction TPV",
        usage="python mybci.py <subject> <run> {train,predict}",
    )
    _add_argument_specs(parser)
    return parser


# Parse les arguments fournis à la CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse les arguments passés à mybci."""

    # Construit le parser pour traiter argv
    parser = build_parser()
    # Retourne l'espace de noms après parsing
    return parser.parse_args(argv)


# Point d'entrée principal de la CLI
def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée exécutable de mybci."""

    # Capture les arguments fournis ou la ligne de commande réelle
    provided_args = list(argv) if argv is not None else list(sys.argv[1:])
    # Lance le runner global lorsque la commande ne fournit aucun argument
    if not provided_args:
        # Exécute la boucle des six expériences sur les 109 sujets
        return _run_global_evaluation()
    # Parse les arguments fournis par l'utilisateur
    args = parse_args(provided_args)
    # Vérifie les dépendances ML pour les modes qui en ont besoin
    if args.mode in {"train", "predict"}:
        # Interrompt avec un message actionnable si scikit-learn manque
        _ensure_ml_dependencies()

    # Construit la configuration de pipeline commune
    config = ModuleCallConfig(
        subject=args.subject,
        run=args.run,
    )

    # Appelle le module train si le mode le demande
    if args.mode == "train":
        # Retourne le code retour du module train avec la configuration
        return _call_module(
            "tpv.train",
            config,
        )

    # Appelle le module predict pour préserver la sortie CLI attendue
    if args.mode == "predict":
        # Retourne le code retour du module predict avec la configuration
        return _call_module(
            "tpv.predict",
            config,
        )

    # Retourne un code explicite si aucun mode valide n'est routé
    return 1


# Protège l'exécution directe pour déléguer au main
if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    # Expose le code retour comme exit code du processus
    raise SystemExit(main())

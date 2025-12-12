#!/usr/bin/env python3

"""Interface CLI pour piloter les workflows d'entraînement et de prédiction."""

# Préserve argparse pour parser les options CLI avec validation
import argparse

# Centralise la moyenne arithmétique pour agréger les accuracies
from statistics import mean

# Préserve subprocess pour lancer les modules en sous-processus isolés
import subprocess

# Préserve sys pour identifier l'interpréteur courant
import sys

# Préserve dataclass pour regrouper les paramètres du pipeline
from dataclasses import dataclass

# Garantit l'accès aux séquences typées pour mypy
from typing import Iterable, Sequence

# Facilite la gestion portable des chemins de données et artefacts
from pathlib import Path


# Centralise les options nécessaires pour invoquer le mode realtime
@dataclass
class RealtimeCallConfig:
    """Conteneur des paramètres transmis au module realtime."""

    # Identifie le sujet cible pour cibler les artefacts du modèle
    subject: str
    # Identifie le run cible pour sélectionner la session
    run: str
    # Fixe la taille de fenêtre glissante en échantillons
    window_size: int
    # Fixe le pas entre deux fenêtres successives
    step_size: int
    # Fixe la taille du buffer utilisé pour lisser les prédictions
    buffer_size: int
    # Renseigne la fréquence d'échantillonnage pour calculer les offsets
    sfreq: float
    # Spécifie la latence maximale autorisée pour chaque fenêtre
    max_latency: float
    # Spécifie le répertoire contenant les fichiers numpy streamés
    data_dir: str
    # Spécifie le répertoire racine où lire les artefacts entraînés
    artifacts_dir: str


# Centralise les options nécessaires pour invoquer un module TPV
@dataclass
class ModuleCallConfig:
    """Conteneur des paramètres transmis aux modules train/predict."""

    # Identifie le sujet cible pour charger les données correspondantes
    subject: str
    # Identifie le run cible pour charger la bonne session
    run: str
    # Sélectionne le classifieur pour harmoniser train et predict
    classifier: str
    # Sélectionne le scaler optionnel pour stabiliser les features
    scaler: str | None
    # Harmonise la stratégie d'extraction des features
    feature_strategy: str
    # Choisit la méthode de réduction de dimension
    dim_method: str
    # Spécifie le nombre de composantes projetées
    n_components: int | None
    # Indique si les features doivent être normalisées
    normalize_features: bool


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
    # Ajoute le classifieur choisi pour transmettre la préférence utilisateur
    command.extend(["--classifier", config.classifier])
    # Ajoute la stratégie de scaler uniquement lorsqu'elle est définie
    if config.scaler is not None:
        # Propulse le scaler choisi vers le module appelé
        command.extend(["--scaler", config.scaler])
    # Ajoute la stratégie de features pour harmoniser train et predict
    command.extend(["--feature-strategy", config.feature_strategy])
    # Ajoute la méthode de réduction de dimension pour la cohérence
    command.extend(["--dim-method", config.dim_method])
    # Ajoute le nombre de composantes si fourni pour contrôler la compression
    if config.n_components is not None:
        # Passe n_components sous forme de chaîne pour argparse descendant
        command.extend(["--n-components", str(config.n_components)])
    # Ajoute un indicateur pour désactiver la normalisation si demandé
    if not config.normalize_features:
        # Utilise un flag booléen pour inverser la valeur par défaut
        command.append("--no-normalize-features")
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
    # Import direct pour aligner avec l'appel CLI existant
    from tpv import predict as tpv_predict

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


# Parcourt les 6 expériences et les 109 sujets en affichant les accuracies
def _run_global_evaluation(
    experiments: Sequence[ExperimentDefinition] | None = None,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> int:
    """Exécute la boucle d'évaluation globale décrite dans le sujet."""

    # Utilise les expériences par défaut si aucune liste n'est fournie
    experiment_definitions = list(experiments or _build_default_experiments())
    # Normalise les chemins racine de données pour les appels descendants
    data_root = data_dir or Path("data")
    # Normalise le répertoire d'artefacts pour les modèles entraînés
    artifacts_root = artifacts_dir or Path("artifacts")
    # Normalise le répertoire des EDF bruts nécessaires à evaluate_run
    raw_root = raw_dir or Path("data/raw")
    # Prépare le stockage des accuracies par expérience
    per_experiment_scores: dict[int, list[float]] = {
        # Initialise la collection d'accuracies pour chaque expérience
        exp.index: []
        for exp in experiment_definitions
    }

    # Parcourt chaque expérience demandée
    for experiment in experiment_definitions:
        # Parcourt l'ensemble des 109 sujets numérotés de 1 à 109
        for subject_index in range(1, 110):
            # Évalue le sujet courant sur l'expérience en cours
            try:
                # Calcule l'accuracy en rechargeant le modèle entraîné
                accuracy = _evaluate_experiment_subject(
                    experiment,
                    subject_index,
                    data_root,
                    artifacts_root,
                    raw_root,
                )
            except FileNotFoundError as error:
                # Informe l'utilisateur qu'un prérequis manque pour ce run
                print(f"ERREUR: {error}")
                # Stoppe l'exécution globale pour signaler l'anomalie
                return 1
            # Stocke l'accuracy pour le calcul des moyennes
            per_experiment_scores[experiment.index].append(accuracy)
            # Affiche l'accuracy du sujet au format imposé
            print(
                f"experiment {experiment.index}: "
                f"subject {subject_index:03d}: accuracy = {accuracy:.4f}"
            )

    # Affiche l'entête du bloc de moyennes par expérience
    print("Mean accuracy of the six different experiments for all 109 subjects:")
    # Calcule et affiche la moyenne de chaque expérience
    for experiment in experiment_definitions:
        # Calcule la moyenne de l'expérience courante
        experiment_mean = _safe_mean(per_experiment_scores[experiment.index])
        # Affiche la moyenne alignée sur l'exemple fourni
        print(
            f"experiment {experiment.index}:\t\taccuracy = "
            f"{experiment_mean:.4f}"
        )
    # Calcule la moyenne globale des six expériences
    global_mean = _safe_mean(
        _safe_mean(per_experiment_scores[exp.index]) for exp in experiment_definitions
    )
    # Affiche la moyenne globale demandée par la consigne
    print(f"Mean accuracy of 6 experiments: {global_mean:.4f}")
    # Retourne 0 pour signaler le succès global
    return 0


# Construit la ligne de commande pour invoquer le mode realtime
def _call_realtime(config: RealtimeCallConfig) -> int:
    """Invoke le module tpv.realtime avec les paramètres streaming."""

    # Initialise la commande avec l'interpréteur courant et le module ciblé
    command: list[str] = [
        sys.executable,
        "-m",
        "tpv.realtime",
        config.subject,
        config.run,
    ]
    # Ajoute la taille de fenêtre demandée pour le streaming
    command.extend(["--window-size", str(config.window_size)])
    # Ajoute le pas de glissement entre fenêtres successives
    command.extend(["--step-size", str(config.step_size)])
    # Ajoute la taille du buffer utilisé pour lisser les prédictions
    command.extend(["--buffer-size", str(config.buffer_size)])
    # Ajoute la latence maximale autorisée pour surveiller le SLA
    command.extend(["--max-latency", str(config.max_latency)])
    # Ajoute la fréquence d'échantillonnage pour calculer les offsets
    command.extend(["--sfreq", str(config.sfreq)])
    # Ajoute le répertoire des données streamées
    command.extend(["--data-dir", config.data_dir])
    # Ajoute le répertoire d'artefacts contenant le modèle entraîné
    command.extend(["--artifacts-dir", config.artifacts_dir])
    # Exécute la commande en capturant le code retour sans lever d'exception
    completed = subprocess.run(command, check=False)
    # Retourne le code retour pour propagation à l'appelant principal
    return completed.returncode


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
            f"epoch {idx:0{index_width}d}: "
            f"[{int(pred)}] [{int(truth)}] {is_equal}"
        )
    # Affiche l'accuracy globale formatée sur quatre décimales
    print(f"Accuracy: {accuracy:.4f}")


# Construit le parser CLI avec toutes les options du pipeline
def build_parser() -> argparse.ArgumentParser:
    """Construit l'argument parser pour mybci."""

    # Instancie le parser avec description et usage explicite
    parser = argparse.ArgumentParser(
        description="Pilote un workflow d'entraînement ou de prédiction TPV",
        usage="python mybci.py <subject> <run> {train,predict,realtime}",
    )
    # Ajoute l'identifiant du sujet pour cibler les données
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'identifiant du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    # Ajoute le mode pour distinguer entraînement et prédiction
    parser.add_argument(
        "mode",
        choices=("train", "predict", "realtime"),
        help="Choix du pipeline à lancer",
    )
    # Ajoute le choix du classifieur pour composer le pipeline
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm", "centroid"),
        default="lda",
        help="Choix du classifieur final",
    )
    # Ajoute le scaler optionnel pour stabiliser les features
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "none"),
        default="none",
        help="Scaler optionnel appliqué après les features",
    )
    # Ajoute la stratégie d'extraction de features pour harmoniser train/predict
    parser.add_argument(
        "--feature-strategy",
        choices=("fft", "wavelet"),
        default="fft",
        help="Méthode d'extraction des features",
    )
    # Ajoute la méthode de réduction de dimension pour ajuster la compression
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp"),
        default="pca",
        help="Technique de réduction de dimension",
    )
    # Ajoute le nombre de composantes pour contrôler la taille projetée
    parser.add_argument(
        "--n-components",
        type=int,
        default=argparse.SUPPRESS,
        help="Nombre de composantes à conserver",
    )
    # Ajoute un flag pour désactiver la normalisation des features
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Désactive la normalisation des features",
    )
    # Ajoute la taille de fenêtre pour la lecture streaming en realtime
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Taille de fenêtre glissante pour le mode realtime",
    )
    # Ajoute le pas entre deux fenêtres successives
    parser.add_argument(
        "--step-size",
        type=int,
        default=25,
        help="Pas entre deux fenêtres en streaming realtime",
    )
    # Ajoute la taille du buffer pour lisser les prédictions instantanées
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=3,
        help="Taille du buffer de lissage pour le mode realtime",
    )
    # Ajoute la fréquence d'échantillonnage utilisée pour les offsets
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage appliquée au flux realtime",
    )
    # Ajoute la latence maximale tolérée pour surveiller la boucle
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Latence maximale autorisée par fenêtre realtime",
    )
    # Ajoute le répertoire de données nécessaire au streaming
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Répertoire racine contenant les fichiers numpy",
    )
    # Ajoute le répertoire d'artefacts où lire le modèle entraîné
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Répertoire racine où récupérer le modèle entraîné",
    )
    # Ajoute le répertoire racine des fichiers EDF bruts
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Répertoire racine contenant les fichiers EDF bruts",
    )
    # Retourne le parser configuré
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
    # Interprète le choix du scaler pour convertir "none" en None
    scaler = None if args.scaler == "none" else args.scaler
    # Applique la normalisation en inversant le flag d'opt-out
    normalize_features = not args.no_normalize_features
    # Construit la configuration partagée entre les modules train et predict
    n_components = getattr(args, "n_components", None)

    # Construit la configuration de pipeline commune
    config = ModuleCallConfig(
        subject=args.subject,
        run=args.run,
        classifier=args.classifier,
        scaler=scaler,
        feature_strategy=args.feature_strategy,
        dim_method=args.dim_method,
        n_components=n_components,
        normalize_features=normalize_features,
    )

    # Appelle le module train si le mode le demande
    if args.mode == "train":
        # Retourne le code retour du module train avec la configuration
        return _call_module(
            "tpv.train",
            config,
        )

    # Bascule vers le module realtime pour le streaming fenêtré
    if args.mode == "realtime":
        # Construit la configuration spécifique au lissage et fenêtrage
        realtime_config = RealtimeCallConfig(
            subject=args.subject,
            run=args.run,
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            sfreq=args.sfreq,
            max_latency=args.max_latency,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
        )
        # Retourne le code retour du module realtime avec la configuration
        return _call_realtime(realtime_config)

    # Traite le mode prédiction avec un appel direct au module tpv.predict
    # pour pouvoir structurer l'affichage epoch par epoch
    from tpv import predict as tpv_predict

    # Convertit les répertoires en Path pour l'appel direct
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    raw_dir = Path(args.raw_dir)

    # Évalue le run en récupérant prédictions et vérité terrain
    try:
        # Lance l'évaluation en rechargeant la pipeline entraînée
        result = tpv_predict.evaluate_run(
            args.subject,
            args.run,
            data_dir,
            artifacts_dir,
            raw_dir,
        )
    except FileNotFoundError as error:
        # Affiche une erreur compréhensible pour l'utilisateur CLI
        print(f"ERREUR: {error}")
        # Retourne un code non nul pour signaler l'échec
        return 1

    # Construit le rapport agrégé (accuracy globale, confusion, etc.)
    _ = tpv_predict.build_report(result)
    # Récupère les prédictions calculées par evaluate_run
    y_pred = result["predictions"]
    # Récupère la vérité terrain renvoyée par evaluate_run
    y_true = result["truth"]
    # Récupère l'accuracy globale pour l'affichage final
    accuracy = float(result["accuracy"])
    # Affiche le détail epoch par epoch dans le format attendu
    _print_epoch_predictions(y_true, y_pred, accuracy)
    # Retourne 0 pour signaler un succès CLI à l'utilisateur
    return 0


# Protège l'exécution directe pour déléguer au main
if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    # Expose le code retour comme exit code du processus
    raise SystemExit(main())

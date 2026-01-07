"""Visualisation WBS 3.3 : brut vs filtré Physionet."""

# Force un backend sans affichage pour les environnements CI et serveurs
import matplotlib

# Active Agg pour générer les PNG sans dépendre d'un écran
matplotlib.use("Agg")

# Importe argparse pour exposer les options CLI sujet/run/canaux
import argparse

# Importe json pour sérialiser la configuration accompagnant les figures
import json

# Importe math pour dimensionner la légende
import math

# Importe dataclass pour encapsuler les paramètres de visualisation
from dataclasses import dataclass

# Importe pathlib pour gérer les chemins dataset et de sortie
from pathlib import Path

# Importe typing pour typer explicitement les séquences et tuples
from typing import Sequence, Tuple

# Importe pyplot pour tracer les figures comparatives
import matplotlib.pyplot as plt

# Importe BaseRaw pour typer précisément les enregistrements EEG
from mne.io import BaseRaw

# Importe le filtrage validé pour rester aligné avec preprocessing
from tpv.preprocessing import apply_bandpass_filter, load_physionet_raw

MAX_SUBJECTS_PREVIEW = 5

# Fixe le nombre maximal de lignes par colonne dans la légende
LEGEND_MAX_ROWS = 16

# Fixe une limite haute de colonnes pour préserver la zone de tracé
LEGEND_MAX_COLS = 6


# Calcule le nombre de colonnes pour afficher toutes les entrées de légende
def _infer_legend_ncol(channel_count: int) -> int:
    """Retourne ncol afin d'éviter une légende trop haute."""

    # Retourne une colonne par défaut si aucun canal n'est fourni
    if channel_count <= 0:
        return 1

    # Calcule le nombre de colonnes requis pour borner la hauteur
    required = int(math.ceil(channel_count / float(LEGEND_MAX_ROWS)))

    # Borne ncol pour préserver une largeur de tracé acceptable
    return max(1, min(LEGEND_MAX_COLS, required))


# Calcule la fraction de figure dédiée aux axes en réservant une zone légende
def _infer_tight_layout_right(ncol: int) -> float:
    """Retourne le paramètre rect.right pour tight_layout()."""

    # Définit une réserve minimale suffisante pour une colonne unique
    base_reserved = 0.17

    # Ajoute une réserve par colonne supplémentaire pour éviter le recouvrement
    per_col_reserved = 0.06

    # Calcule la réserve totale à droite selon le nombre de colonnes
    reserved = base_reserved + per_col_reserved * max(0, ncol - 1)

    # Limite la réserve pour éviter d'écraser totalement les graphiques
    reserved = min(0.45, reserved)

    # Retourne la borne droite des axes pour laisser la place à la légende
    return 1.0 - reserved


# Fournit un récap rapide des sujets déjà présents sous data_root
def _describe_subjects(data_root: Path) -> str:
    """Retourne une ligne décrivant les sujets détectés sous data_root."""

    if not data_root.exists():
        return f"- Racine data absente : {data_root}"
    available_subjects = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if not available_subjects:
        return f"- Aucun sujet détecté dans {data_root}"
    preview = ", ".join(available_subjects[:MAX_SUBJECTS_PREVIEW])
    suffix = " …" if len(available_subjects) > MAX_SUBJECTS_PREVIEW else ""
    return f"- Sujets présents dans {data_root}: {preview}{suffix}"


# Fournit un récap rapide des runs disponibles pour le sujet demandé
def _describe_runs(subject_root: Path) -> str:
    """Retourne une ligne listant les runs EDF trouvés pour le sujet."""

    if not subject_root.exists():
        return f"- Répertoire sujet absent : {subject_root}"
    available_runs = sorted(p.stem for p in subject_root.glob("*.edf"))
    normalized_runs = []
    for run in available_runs:
        if run.startswith(subject_root.name):
            normalized_runs.append(run[len(subject_root.name) :])
        else:
            normalized_runs.append(run)
    if normalized_runs:
        return (
            f"- Runs disponibles pour {subject_root.name}: "
            f"{', '.join(normalized_runs)}"
        )
    if available_runs:
        return (
            f"- Runs disponibles pour {subject_root.name}: {', '.join(available_runs)}"
        )
    return f"- Aucun run EDF trouvé dans {subject_root}"


# Construit un message clair pour guider l'utilisateur lorsque le fichier est absent
def _format_missing_recording_message(recording_path: Path) -> str:
    """Assemble un message multi-lignes décrivant l'arborescence attendue."""

    data_root = recording_path.parent.parent
    subject_root = recording_path.parent
    subject = subject_root.name
    run_stem = recording_path.stem
    normalized_run = (
        run_stem[len(subject) :] if run_stem.startswith(subject) else run_stem
    )
    primary_candidate = subject_root / f"{subject}{normalized_run}.edf"
    secondary_candidate = subject_root / f"{normalized_run}.edf"
    candidates = {primary_candidate, secondary_candidate}
    return "\n".join(
        [
            f"Recording not found: {recording_path}",
            "- Chemins tentés : "
            + ", ".join(str(candidate) for candidate in sorted(candidates)),
            (
                f"- Structure attendue : {data_root}/<subject>/<subject><run>.edf "
                "(ex: S001/S001R03.edf) ou <subject>/<run>.edf si déjà préfixé"
            ),
            _describe_subjects(data_root),
            _describe_runs(subject_root),
            (
                "Téléchargez le dataset Physionet EEG Motor Movement/Imagery "
                "et pointez --data-root vers sa racine."
            ),
        ]
    )


# Regroupe les paramètres de visualisation pour limiter les arguments
@dataclass
class VisualizationConfig:
    """Conteneur des options CLI pour la visualisation brut/filtré."""

    # Définit la sélection facultative de canaux à tracer
    channels: Sequence[str] | None
    # Définit le répertoire de sortie des figures générées
    output_dir: Path
    # Choisit la méthode de filtrage alignée avec preprocessing
    filter_method: str
    # Spécifie la bande fréquentielle appliquée au signal
    freq_band: Tuple[float, float]
    # Spécifie la durée de padding pour réduire les artefacts de bord
    pad_duration: float
    # Définit un titre optionnel pour annoter la figure
    title: str | None


# Centralise la construction du parseur pour harmoniser l'interface CLI
def build_parser() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument(
        "subject",
        help="Identifiant du sujet ex: S001",
    )
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument(
        "run",
        help="Identifiant du run ex: R01",
    )
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir",
        default="docs/viz",
        help="Répertoire de sauvegarde PNG",
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Construit le chemin EDF attendu pour un sujet et un run donnés
def build_recording_path(
    data_root: Path,
    subject: str,
    run: str,
) -> Path:
    """Retourne le chemin EDF data/<subject>/<subject><run>.edf si présent."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    subject_root = normalized_root / subject
    run_stem = Path(run).stem
    candidates: list[Path] = []
    if run_stem.startswith(subject):
        candidates.append(subject_root / f"{run_stem}.edf")
    else:
        candidates.append(subject_root / f"{subject}{run_stem}.edf")
        candidates.append(subject_root / f"{run_stem}.edf")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Retourne la préférence avec préfixe sujet pour matcher Physionet
    return candidates[0]


# Charge un enregistrement Physionet complet avec métadonnées associées
def load_recording(
    recording_path: Path,
) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(_format_missing_recording_message(recording_path))
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    subject = recording_path.parent.name
    run_label = recording_path.stem
    if run_label.startswith(subject):
        run_label = run_label[len(subject) :]
    metadata.update(
        {
            "subject": subject,
            "run": run_label,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def pick_channels(
    raw: BaseRaw,
    channels: Sequence[str] | None,
) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Applique le filtre passe-bande avec les paramètres CLI
def filter_recording(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def plot_raw_vs_filtered(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    title: str | None,
    metadata: dict,
) -> Path:
    """Enregistre une figure montrant brut et filtré canal par canal."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Crée une figure avec deux lignes pour juxtaposer brut et filtré
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # Définit le titre global pour rappeler sujet/run et méthode
    fig.suptitle(title or f"{metadata['subject']} {metadata['run']} raw vs filtered")

    # Capture les handles matplotlib pour construire une légende unique
    raw_handles: list[object] = []

    # Trace chaque canal brut avec légende pour lecture rapide
    for idx, channel in enumerate(raw.ch_names):
        # Trace le canal idx sur le premier subplot
        lines = axes[0].plot(times, raw_data[idx], label=channel)
        raw_handles.append(lines[0])
    # Ajoute un label d'axe pour contextualiser les valeurs brutes
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Titre spécifique à la partie brute pour distinguer visuellement
    axes[0].set_title("Signal brut")
    # Trace chaque canal filtré en alignant sur la même grille temporelle
    for idx, channel in enumerate(filtered.ch_names):
        # Trace le canal idx filtré sur le second subplot
        axes[1].plot(times, filtered_data[idx], label=channel)
    # Ajoute un label d'axe dédié aux données filtrées
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Ajoute un label de l'axe des temps pour les deux sous-graphiques
    axes[1].set_xlabel("Temps (s)")
    # Titre spécifique à la partie filtrée pour faciliter la comparaison
    axes[1].set_title("Signal filtré 8-40 Hz")

    # Calcule le nombre de colonnes pour éviter une légende trop haute
    legend_ncol = _infer_legend_ncol(len(raw.ch_names))

    # Réserve une zone à droite pour afficher la légende intégrale
    layout_right = _infer_tight_layout_right(legend_ncol)

    # Place la légende dans la zone réservée pour éviter la troncature
    legend_x = min(0.98, layout_right + 0.01)

    # Crée une légende unique pour éviter la duplication sur les deux axes
    fig.legend(
        handles=raw_handles,
        labels=list(raw.ch_names),
        loc="center left",
        bbox_to_anchor=(legend_x, 0.5),
        ncol=legend_ncol,
        fontsize="small",
    )

    # Ajuste le layout en gardant de la place pour la légende et le suptitle
    fig.tight_layout(rect=(0.0, 0.0, layout_right, 0.95))
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Orchestration complète du chargement, filtrage et traçage
def visualize_run(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config.title,
        metadata,
    )


# Point d'entrée CLI conforme aux instructions README
def main() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Active l'exécution du main uniquement en invocation directe
if __name__ == "__main__":
    main()

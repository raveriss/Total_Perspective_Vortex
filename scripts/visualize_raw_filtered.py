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

# Importe Axes pour typer explicitement les axes Matplotlib
from matplotlib.axes import Axes

# Importe BaseRaw pour typer précisément les enregistrements EEG
from mne.io import BaseRaw

# Importe le filtrage validé pour rester aligné avec preprocessing
from tpv.preprocessing import apply_bandpass_filter, load_physionet_raw

MAX_SUBJECTS_PREVIEW = 5

# Fixe le nombre maximal de lignes par colonne dans la légende
LEGEND_MAX_ROWS = 16

# Fixe une limite haute de colonnes pour préserver la zone de tracé
LEGEND_MAX_COLS = 6

# Fixe un seuil au-delà duquel la légende devient contre-productive
LEGEND_MAX_CHANNELS = 12

# Définit un bleu électrique unique pour un rendu cohérent
ELECTRIC_BLUE = "#12127E"

# Fixe la largeur standard des figures comparatives
FIGURE_WIDTH = 12.0

# Fixe la hauteur standard d'une ligne de sous-graphiques
FIGURE_ROW_HEIGHT = 4.0

# Associe une couleur stable à chaque région topographique
REGION_COLORS = {
    "Central": ELECTRIC_BLUE,
    "Frontal": "#FF7F0E",
    "Parietal": "#2CA02C",
    "Temporal": "#D62728",
    "Occipital": "#9467BD",
}


# Définit un sous-ensemble sensorimoteur lisible pour l’usage quotidien
DEFAULT_MOTOR_ROI = (
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
)


# Formate une fréquence en supprimant les décimales inutiles
def _format_frequency(value: float) -> str:
    """Retourne une fréquence lisible pour les titres de figure."""

    # Convertit en float pour sécuriser les valeurs provenant d'argparse
    numeric_value = float(value)
    # Supprime les décimales inutiles pour les valeurs entières
    if numeric_value.is_integer():
        # Retourne une chaîne compacte pour la fréquence entière
        return str(int(numeric_value))
    # Retourne un format court pour conserver la précision utile
    return f"{numeric_value:g}"


# Formate une bande de fréquences pour l'affichage dans le titre
def _format_freq_band(freq_band: Tuple[float, float]) -> str:
    """Retourne une bande formattée comme 8-40 pour les titres."""

    # Formate la borne basse pour un affichage lisible
    low = _format_frequency(freq_band[0])
    # Formate la borne haute pour un affichage lisible
    high = _format_frequency(freq_band[1])
    # Concatène les bornes avec un tiret pour les titres courts
    return f"{low}-{high}"


# Détermine la région scalp associée à un nom de canal
def _resolve_channel_region(channel: str) -> str:
    """Mappe un canal EEG vers une région topographique simple."""

    # Normalise le nom de canal pour comparer les préfixes
    normalized = channel.upper()
    # Route les canaux C vers la région centrale
    if normalized.startswith("C"):
        return "Central"
    # Route les canaux F vers la région frontale
    if normalized.startswith("F"):
        return "Frontal"
    # Route les canaux P vers la région pariétale
    if normalized.startswith("P"):
        return "Parietal"
    # Route les canaux T vers la région temporale
    if normalized.startswith("T"):
        return "Temporal"
    # Route les canaux O vers la région occipitale
    if normalized.startswith("O"):
        return "Occipital"
    # Retourne une région générique lorsque le préfixe est inconnu
    return "Autre"


# Regroupe les canaux par région dans l'ordre d'apparition
def _group_channels_by_region(channels: Sequence[str]) -> list[tuple[str, list[int]]]:
    """Retourne la liste ordonnée des régions et indices associés."""

    # Initialise la table de regroupement par région
    grouped: dict[str, list[int]] = {}
    # Conserve l'ordre de première apparition des régions
    ordered_regions: list[str] = []
    # Parcourt chaque canal pour l'associer à une région
    for idx, channel in enumerate(channels):
        # Déduit la région du canal courant
        region = _resolve_channel_region(channel)
        # Initialise la liste si la région n'a pas encore été vue
        if region not in grouped:
            # Ajoute la région à l'ordre conservé
            ordered_regions.append(region)
            # Initialise la liste des indices pour cette région
            grouped[region] = []
        # Ajoute l'indice du canal à la région correspondante
        grouped[region].append(idx)
    # Retourne les régions dans l'ordre original avec leurs indices
    return [(region, grouped[region]) for region in ordered_regions]


# Uniformise le style des séries temporelles pour une lecture plus rapide
def _style_timeseries_axis(axis: Axes) -> None:
    """Applique un style cohérent aux axes des séries temporelles."""

    # Place le cadrillage derrière les courbes pour préserver le contraste
    axis.set_axisbelow(True)
    # Active des ticks mineurs pour densifier la lecture sans surcharger
    axis.minorticks_on()
    # Ajoute le cadrillage principal pour guider les lectures x/y
    axis.grid(which="major", axis="both", linewidth=0.4, alpha=0.80)
    # Ajoute un cadrillage secondaire plus discret pour les interpolations
    axis.grid(which="minor", axis="both", linewidth=0.2, alpha=0.50)
    # Ajoute une ligne zéro pour ancrer la lecture des amplitudes
    axis.axhline(0.0, linewidth=0.6, alpha=0.25)
    # Supprime des bordures peu informatives pour alléger le rendu
    axis.spines["top"].set_visible(False)
    # Supprime des bordures peu informatives pour alléger le rendu
    axis.spines["right"].set_visible(False)
    # Harmonise les ticks pour une lecture plus propre
    axis.tick_params(which="both", direction="out", length=3, width=0.6)


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
    # Ajoute un flag pour forcer l’affichage intégral des 64 canaux
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Affiche tous les canaux (désactive la sélection ROI par défaut)",
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
    config: VisualizationConfig,
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
    # Regroupe les canaux par région pour les traces agrégées
    region_groups = _group_channels_by_region(raw.ch_names)
    # Crée une figure à deux colonnes pour brut et filtré
    fig, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey="row",
        figsize=(FIGURE_WIDTH, FIGURE_ROW_HEIGHT),
    )
    # Formate la bande fréquentielle pour les titres
    freq_label = _format_freq_band(config.freq_band)
    # Prépare le titre par défaut basé sur la bande fréquentielle
    default_title = f"EEG – Comparaison brut vs filtré ({freq_label} Hz)"
    # Définit le titre global de la figure
    fig.suptitle(config.title or default_title)
    # Calcule le nombre de canaux pour le sous-titre
    channel_count = len(raw.ch_names)
    # Sélectionne la forme singulière ou plurielle du libellé
    channel_label = "canal" if channel_count == 1 else "canaux"
    # Ajoute un sous-titre sujet/run centré
    fig.text(
        0.5,
        0.94,
        f"Sujet {metadata['subject']} • Run {metadata['run']} "
        f"• {channel_count} {channel_label}",
        ha="center",
        fontsize="small",
    )
    # Fixe le label Y pour l'axe brut
    axes[0].set_ylabel("Amplitude (a.u.)")
    # Fixe le label Y pour l'axe filtré
    axes[1].set_ylabel("Amplitude filtrée (a.u.)")
    # Fixe le label X pour l'axe brut
    axes[0].set_xlabel("Temps (s)")
    # Fixe le label X pour l'axe filtré
    axes[1].set_xlabel("Temps (s)")
    # Applique une grille légère sur l'axe brut
    axes[0].grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.6)
    # Applique une grille légère sur l'axe filtré
    axes[1].grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.6)
    # Initialise la liste de régions pour la légende globale
    legend_regions: list[str] = []
    # Trace les séries pour chaque région détectée
    for region, indices in region_groups:
        # Ajoute la région à la légende globale
        legend_regions.append(region)
        # Résout la couleur dédiée à la région
        color = REGION_COLORS.get(region, ELECTRIC_BLUE)
        # Trace chaque canal brut de la région
        for idx in indices:
            # Trace la série brute en transparence
            axes[0].plot(times, raw_data[idx], color=color, alpha=0.2, linewidth=0.6)
            # Trace la série filtrée en transparence
            axes[1].plot(
                times,
                filtered_data[idx],
                color=color,
                alpha=0.2,
                linewidth=0.6,
            )
        # Calcule la moyenne brute régionale
        raw_mean = raw_data[indices].mean(axis=0)
        # Calcule l'écart-type brut régional
        raw_std = raw_data[indices].std(axis=0)
        # Calcule la moyenne filtrée régionale
        filtered_mean = filtered_data[indices].mean(axis=0)
        # Calcule l'écart-type filtré régional
        filtered_std = filtered_data[indices].std(axis=0)
        # Trace la moyenne brute
        axes[0].plot(times, raw_mean, color=color, linewidth=1.6)
        # Trace l'enveloppe brute mean ± std
        axes[0].fill_between(
            times,
            raw_mean - raw_std,
            raw_mean + raw_std,
            color=color,
            alpha=0.2,
        )
        # Trace la moyenne filtrée
        axes[1].plot(times, filtered_mean, color=color, linewidth=1.6)
        # Trace l'enveloppe filtrée mean ± std
        axes[1].fill_between(
            times,
            filtered_mean - filtered_std,
            filtered_mean + filtered_std,
            color=color,
            alpha=0.2,
        )
        # Applique le titre brut par région
        axes[0].set_title(f"Brut — {region}")
        # Applique le titre filtré par région
        axes[1].set_title(f"Filtré {freq_label} Hz — {region}")
    # Ajoute la légende globale des régions
    fig.legend(
        labels=legend_regions,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, len(legend_regions)),
        fontsize="small",
        frameon=False,
    )
    # Ajuste le layout pour réserver la place du titre et de la légende
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
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
        config,
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
            channels=(
                None if args.all_channels or args.channels is None else args.channels
            ),
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

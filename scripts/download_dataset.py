#!/usr/bin/env python3
"""Télécharge EEGMMIDB depuis les sources officielles PhysioNet."""

# On expose une CLI standard pour l'appel depuis le Makefile.
import argparse

# On groupe les signatures réseau pour des diagnostics cohérents.
import re

# On vérifie la présence des outils système et des répertoires.
import shutil

# On pilote wget et les diagnostics système au moment de l'échec runtime.
import subprocess  # nosec B404

# On sépare stdout/stderr pour des messages utilisateur propres.
import sys

# On distingue les erreurs HTTP pour adapter le message affiché.
import urllib.error

# On contacte PhysioNet sans dépendance Python supplémentaire.
import urllib.request

# On structure les diagnostics locaux pour garder un contrat stable.
from dataclasses import dataclass

# On manipule les chemins sans dépendre du shell courant.
from pathlib import Path

# On conserve un typage souple sur les doubles de tests.
from typing import Any

# On limite volontairement les sources aux endpoints officiels supportés.
OFFICIAL_SOURCE_CANDIDATES = (
    "https://physionet.org/files/eegmmidb/1.0.0/",
    "https://physionet.org/static/published-projects/eegmmidb/1.0.0/",
)
# On tolère le fallback GET quand HEAD est refusé par le serveur.
HTTP_METHOD_NOT_ALLOWED = 405
# On traite explicitement le code wget dédié aux erreurs réseau.
WGET_NETWORK_ERROR_STATUS = 4
# On reconnaît les signatures réseau utiles à l'utilisateur final.
NETWORK_ERROR_PATTERN = re.compile(
    r"Temporary failure in name resolution|"
    r"unable to resolve host address|"
    r"Name or service not known|"
    r"Network is unreachable|"
    r"No route to host|"
    r"Connection timed out|"
    r"Connection refused|"
    r"Unable to establish SSL connection|"
    r"TLS",
    re.IGNORECASE,
)
# On isole les erreurs HTTP source pour ne pas les confondre avec le réseau.
HTTP_SOURCE_ERROR_PATTERN = re.compile(
    r"404 Not Found|ERROR 404|403 Forbidden|ERROR 403|400 Bad Request|ERROR 400",
    re.IGNORECASE,
)
# On teste un accès IP brut pour distinguer réseau et DNS.
NETWORK_TEST_IP = "1.1.1.1"
# On vérifie le nom d'hôte exact réellement requis par le téléchargement.
PHYSIONET_HOSTNAME = "physionet.org"
# On borne le diagnostic local pour éviter un script bloqué trop longtemps.
DIAGNOSTIC_TIMEOUT_SECONDS = 5


# On stabilise le format des diagnostics entre le script et les tests.
@dataclass(frozen=True)
class CommandDiagnostic:
    """Résultat structuré d'une commande de diagnostic locale."""

    # On garde la commande pour expliquer précisément ce qui a été testé.
    command: tuple[str, ...]
    # On conserve le retour système pour inférer une cause probable.
    returncode: int | None
    # On remonte stdout car certains outils y écrivent leur résultat utile.
    stdout: str
    # On remonte stderr car les causes réseau y sont souvent décrites.
    stderr: str
    # On encode les échecs de lancement sans lever une erreur fatale.
    failure: str | None = None

    # On expose une forme lisible de la commande pour les logs utilisateur.
    @property
    def display_command(self) -> str:
        # On évite les structures tuple dans le message final.
        return " ".join(self.command)

    # On centralise l'agrégation des sorties pour l'inférence de cause.
    @property
    def combined_output(self) -> str:
        # On garde seulement les sorties réellement présentes.
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


# On distingue les erreurs déjà traduites des erreurs internes inattendues.
class HandledDownloadError(Exception):
    """Erreur utilisateur déjà expliquée à afficher puis terminer proprement."""

    # On conserve les lignes déjà prêtes pour éviter une retraduction plus bas.
    def __init__(self, lines: list[str]) -> None:
        # On alimente Exception pour un débogage minimal si nécessaire.
        super().__init__("\n".join(lines))
        # On garde la structure ligne par ligne pour l'affichage final.
        self.lines = lines


# On normalise les URLs pour rendre les comparaisons et les logs stables.
def _normalize_candidate(url: str) -> str:
    """Normalise une URL candidate en ajoutant un slash terminal."""
    # On supprime les espaces parasites issus d'une configuration externe.
    normalized = url.strip()
    # On force une forme unique pour éviter des probes incohérents.
    if not normalized.endswith("/"):
        # On stabilise la concaténation des chemins côté PhysioNet.
        normalized = f"{normalized}/"
    # On renvoie une URL prête à être testée et affichée.
    return normalized


# On refuse les datasets partiels avant d'entraîner la suite du pipeline.
def check_dataset_complete(
    data_dir: Path, subject_count: int, run_count: int
) -> tuple[bool, str | None]:
    """Vérifie la présence des sujets et runs attendus."""
    # On évite de lancer la boucle si la racine n'existe même pas.
    if not data_dir.is_dir():
        # On donne une cause immédiate et actionnable à l'utilisateur.
        return False, f"Dataset incomplet: dossier racine manquant ({data_dir})."
    # On verrouille la présence de tous les sujets attendus par la soutenance.
    for subject_index in range(1, subject_count + 1):
        # On conserve la nomenclature canonique PhysioNet dans les chemins.
        subject = f"S{subject_index:03d}"
        # On travaille sur un chemin explicite pour produire un message précis.
        subject_dir = data_dir / subject
        # On échoue dès le premier sujet manquant pour un diagnostic court.
        if not subject_dir.is_dir():
            # On remonte le chemin concret à réparer ou à re-télécharger.
            return False, f"Dataset incomplet: dossier sujet manquant ({subject_dir})."
        # On valide tous les runs pour garantir une base de données homogène.
        for run_index in range(1, run_count + 1):
            # On garde la convention des noms de runs pour éviter les ambiguïtés.
            run = f"R{run_index:02d}"
            # On exige l'EDF car c'est la matière première du pipeline EEG.
            edf_path = subject_dir / f"{subject}{run}.edf"
            # On exige aussi l'event pour préserver le couplage signal/événements.
            event_path = subject_dir / f"{subject}{run}.edf.event"
            # On refuse un EDF vide pour éviter un faux sentiment de complétude.
            if not edf_path.is_file() or edf_path.stat().st_size == 0:
                # On pointe le fichier exact qui bloque la validation locale.
                return (
                    False,
                    f"Dataset incomplet: fichier manquant ou vide ({edf_path}).",
                )
            # On refuse un event vide pour protéger le découpage supervisé.
            if not event_path.is_file() or event_path.stat().st_size == 0:
                # On pointe le fichier exact qui empêche une exécution saine.
                return (
                    False,
                    f"Dataset incomplet: fichier manquant ou vide ({event_path}).",
                )
    # On confirme une base locale utilisable sans nouveau téléchargement.
    return True, None


# On sonde la source avant wget pour éviter un download coûteux inutile.
def probe_source_url(source_url: str, opener: Any | None = None) -> None:
    """Valide qu'une source officielle répond avant le download massif."""
    # On accepte un opener injecté pour tester les branches réseau finement.
    http_opener = opener if opener is not None else urllib.request.build_opener()
    # On privilégie HEAD pour un test léger avant un transfert de plusieurs Go.
    head_request = urllib.request.Request(source_url, method="HEAD")
    # On isole les réponses HTTP pour distinguer fallback et panne réelle.
    try:
        # On borne le probe pour éviter d'allonger inutilement l'expérience user.
        with http_opener.open(head_request, timeout=10):
            # On arrête le probe dès qu'une source répond correctement.
            return
    # On tolère l'erreur HTTP afin de gérer le cas HEAD refusé proprement.
    except urllib.error.HTTPError as error:
        # On remonte toute erreur HTTP autre qu'un refus de méthode.
        if error.code != HTTP_METHOD_NOT_ALLOWED:
            # On laisse l'appelant classifier cette erreur de manière adaptée.
            raise
    # On retente en GET pour les serveurs qui n'acceptent pas HEAD.
    with http_opener.open(source_url, timeout=10):
        # On valide silencieusement la source dès qu'un GET fonctionne.
        return


# On exécute un diagnostic système sans transformer le script en outil fatal.
def run_command_diagnostic(
    command: tuple[str, ...], runner: Any = subprocess.run
) -> CommandDiagnostic:
    """Exécute une commande locale et capture un diagnostic exploitable."""
    # On capture les erreurs locales pour conserver un message de haut niveau.
    try:
        # On borne le temps d'attente pour protéger la fluidité du CLI.
        result = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=DIAGNOSTIC_TIMEOUT_SECONDS,
        )
    # On remonte l'absence de binaire au lieu d'échouer brutalement.
    except FileNotFoundError:
        # On encode l'échec dans le résultat pour garder un log homogène.
        return CommandDiagnostic(
            command=command,
            returncode=None,
            stdout="",
            stderr="",
            failure=f"commande introuvable: {command[0]}",
        )
    # On signale un diagnostic bloqué sans empêcher la suite du message.
    except subprocess.TimeoutExpired:
        # On garde un état neutre plutôt qu'une exception non traitée.
        return CommandDiagnostic(
            command=command,
            returncode=None,
            stdout="",
            stderr="",
            failure="diagnostic expiré",
        )
    # On normalise les sorties pour faciliter l'affichage ultérieur.
    return CommandDiagnostic(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


# On transforme le diagnostic brut en lignes prêtes à afficher.
def format_command_diagnostic(result: CommandDiagnostic) -> list[str]:
    """Formate un diagnostic local pour le message utilisateur."""
    # On garde un rendu court quand la commande n'a même pas démarré.
    if result.failure is not None:
        # On indique exactement quelle commande manque ou a expiré.
        return [f"- {result.display_command} -> {result.failure}"]
    # On expose le code retour pour éviter un diagnostic opaque.
    lines = [f"- {result.display_command} -> code retour {result.returncode}"]
    # On garde stdout quand l'outil y écrit l'information utile.
    if result.stdout:
        # On donne la preuve textuelle qui motive l'action proposée.
        lines.append(f"  stdout: {result.stdout}")
    # On garde stderr car les causes réseau y sont souvent plus explicites.
    if result.stderr:
        # On montre la phrase exacte émise par l'outil système.
        lines.append(f"  stderr: {result.stderr}")
    # On évite un diagnostic vide qui laisserait l'utilisateur sans contexte.
    if not result.stdout and not result.stderr:
        # On rend visible l'absence totale de réponse du système.
        lines.append("  sortie: aucune réponse")
    # On renvoie des lignes directement injectables dans le message final.
    return lines


# On déduit une action ciblée pour éviter des conseils trop génériques.
def infer_network_failure_cause(
    ping_result: CommandDiagnostic, dns_result: CommandDiagnostic
) -> tuple[str, str]:
    """Déduit une cause probable et une action à partir des diagnostics."""
    # On compare sur une forme homogène pour éviter les variations de casse.
    ping_output = ping_result.combined_output.lower()
    # On compare sur une forme homogène pour reconnaître la panne DNS.
    dns_output = dns_result.combined_output.lower()
    # On priorise une panne de route car elle bloque tout le reste.
    if "network is unreachable" in ping_output:
        # On formule une cause directement corrélée au retour système observé.
        return (
            "Cause probable: la machine n'a plus d'accès réseau sortant.",
            (
                "Action: rétablissez la connexion réseau "
                "(Wi-Fi, câble, VPN ou routage), puis relancez "
                "make download_dataset."
            ),
        )
    # On isole le cas où internet marche mais pas la résolution de nom.
    if ping_result.returncode == 0 and not dns_result.stdout:
        # On évite d'accuser le réseau général quand seul le DNS est en faute.
        return (
            (
                "Cause probable: internet répond mais la résolution DNS de "
                f"{PHYSIONET_HOSTNAME} échoue."
            ),
            (
                "Action: corrigez la configuration DNS du poste ou du "
                "réseau, puis relancez make download_dataset."
            ),
        )
    # On reconnaît explicitement la signature classique de panne DNS.
    if "temporary failure in name resolution" in dns_output:
        # On donne une action cohérente avec ce signal faible mais utile.
        return (
            (
                "Cause probable: la résolution DNS de "
                f"{PHYSIONET_HOSTNAME} est indisponible."
            ),
            (
                "Action: corrigez le DNS actif ou reconnectez le réseau, "
                "puis relancez make download_dataset."
            ),
        )
    # On conserve un fallback explicite si aucun motif fort n'est reconnu.
    return (
        (
            "Cause probable: la machine ne parvient pas à joindre internet "
            "de façon fiable."
        ),
        (
            "Action: vérifiez la connectivité du poste et du réseau, "
            "puis relancez make download_dataset."
        ),
    )


# On centralise les diagnostics locaux pour tous les chemins réseau.
def collect_runtime_network_diagnostics(
    runner: Any = subprocess.run,
) -> list[str]:
    """Collecte les diagnostics locaux ajoutés aux erreurs réseau."""
    # On teste l'accès IP brut pour savoir si la route internet existe.
    ping_result = run_command_diagnostic(("ping", "-c", "1", NETWORK_TEST_IP), runner)
    # On teste la résolution DNS réellement requise par PhysioNet.
    dns_result = run_command_diagnostic(
        ("getent", "hosts", PHYSIONET_HOSTNAME),
        runner,
    )
    # On produit une cause unique pour éviter des actions contradictoires.
    cause_line, action_line = infer_network_failure_cause(ping_result, dns_result)
    # On assemble un bloc prêt à injecter dans les erreurs utilisateur.
    return [
        "Diagnostic local automatique:",
        *format_command_diagnostic(ping_result),
        *format_command_diagnostic(dns_result),
        cause_line,
        action_line,
    ]


# On garantit un message réseau complet et homogène sur toutes les branches.
def build_network_failure_lines(network_diagnostics: list[str]) -> list[str]:
    """Construit un message réseau complet avec diagnostics locaux."""
    # On annonce d'abord le symptôme métier observé côté utilisateur.
    return [
        (
            "❌ Connexion internet indisponible ou instable: impossible "
            "de joindre PhysioNet."
        ),
        *network_diagnostics,
        *collect_runtime_network_diagnostics(),
    ]


# On choisit dynamiquement une source officielle réellement joignable.
def select_official_source(
    opener: Any | None = None, probe_func: Any = probe_source_url
) -> str:
    """Choisit dynamiquement une source officielle PhysioNet disponible."""
    # On accumule les symptômes réseau pour expliquer l'échec final.
    network_diagnostics: list[str] = []
    # On accumule les symptômes HTTP pour distinguer source et connectivité.
    http_diagnostics: list[str] = []
    # On parcourt les endpoints officiels dans l'ordre du plus direct au fallback.
    for candidate in OFFICIAL_SOURCE_CANDIDATES:
        # On homogénéise l'URL avant probe et affichage utilisateur.
        normalized_candidate = _normalize_candidate(candidate)
        # On tente un probe léger avant le vrai téléchargement massif.
        try:
            # On délègue le probe pour faciliter les doubles de tests.
            probe_func(normalized_candidate, opener=opener)
        # On classe séparément les erreurs HTTP pour un message précis.
        except urllib.error.HTTPError as error:
            # On conserve la source et le code qui ont réellement échoué.
            http_diagnostics.append(
                f"- {normalized_candidate} -> HTTP {error.code} {error.reason}"
            )
        # On classe séparément les erreurs réseau pour lancer le bon diagnostic.
        except urllib.error.URLError as error:
            # On préserve la cause système exacte remontée par urllib.
            network_diagnostics.append(f"- {normalized_candidate} -> {error.reason}")
        # On choisit immédiatement la première source officielle utilisable.
        else:
            # On arrête les probes pour éviter des délais inutiles.
            return normalized_candidate
    # On injecte un diagnostic local si toutes les sources échouent par réseau.
    if network_diagnostics and not http_diagnostics:
        # On lève une erreur déjà traduite pour garder un flux simple dans main.
        raise HandledDownloadError(build_network_failure_lines(network_diagnostics))
    # On conserve un message distinct quand PhysioNet répond mais ne sert pas la source.
    raise HandledDownloadError(
        [
            "❌ Sources officielles EEGMMIDB indisponibles sur PhysioNet.",
            *http_diagnostics,
            *network_diagnostics,
            "Action 1: vérifiez que PhysioNet publie bien EEGMMIDB v1.0.0.",
            "Action 2: relancez make download_dataset plus tard.",
        ]
    )


# On délègue le transfert à wget pour bénéficier de la reprise native.
def run_wget_download(
    source_url: str, destination_dir: Path, popen_factory: Any = subprocess.Popen
) -> tuple[int, str]:
    """Exécute wget en relayant la sortie vers le terminal."""
    # On garde les options au même endroit pour stabiliser le comportement CLI.
    command = [
        "wget",
        "-r",
        "-N",
        "-c",
        "--tries=3",
        "--timeout=20",
        "--dns-timeout=15",
        "--connect-timeout=15",
        "--read-timeout=30",
        "-np",
        "-nH",
        "--cut-dirs=3",
        "--accept",
        "*.edf,*.edf.event",
        "-R",
        "index.html*",
        "-P",
        str(destination_dir),
        source_url,
    ]
    # On capture la sortie pour l'afficher et la classifier ensuite.
    process = popen_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # On garde l'historique pour construire un diagnostic lisible après échec.
    output_lines: list[str] = []
    # On protège la boucle contre un stdout absent sur certains doubles.
    if process.stdout is not None:
        # On stream la sortie pour ne pas figer le terminal pendant le download.
        for line in process.stdout:
            # On laisse l'utilisateur voir la progression réelle de wget.
            print(line, end="")
            # On mémorise la ligne pour le diagnostic post-mortem.
            output_lines.append(line)
    # On renvoie le statut et la trace complète pour la classification.
    return process.wait(), "".join(output_lines)


# On traduit les statuts wget en messages métier actionnables.
def classify_wget_error(wget_status: int, wget_output: str) -> list[str] | None:
    """Traduit les échecs wget en diagnostics utilisateur actionnables."""
    # On ne fabrique aucun message d'erreur si wget a terminé correctement.
    if wget_status == 0:
        # On laisse l'appelant poursuivre la validation normale du dataset.
        return None
    # On cherche une signature réseau explicite dans la sortie brute.
    diagnostic_match = NETWORK_ERROR_PATTERN.search(wget_output)
    # On traite le code réseau wget comme une panne réseau même sans motif texte.
    if diagnostic_match is not None or wget_status == WGET_NETWORK_ERROR_STATUS:
        # On privilégie le texte réel pour un log plus transparent.
        diagnostic = (
            diagnostic_match.group(0)
            if diagnostic_match
            else "erreur réseau signalée par wget."
        )
        # On enrichit le message avec les diagnostics locaux automatiques.
        return [
            (
                "❌ Dataset EEGMMIDB incomplet: téléchargement interrompu "
                "par une erreur réseau."
            ),
            f"Diagnostic wget: {diagnostic}",
            *collect_runtime_network_diagnostics(),
        ]
    # On garde un message dédié quand la source répond en erreur HTTP.
    if HTTP_SOURCE_ERROR_PATTERN.search(wget_output) is not None:
        # On évite de parler de réseau quand c'est une indisponibilité source.
        return [
            "❌ Sources officielles EEGMMIDB indisponibles sur PhysioNet.",
            (
                "Diagnostic: PhysioNet a répondu avec une erreur HTTP "
                "sur les endpoints officiels."
            ),
            "Action 1: vérifiez que PhysioNet publie bien EEGMMIDB v1.0.0.",
            "Action 2: relancez make download_dataset plus tard.",
        ]
    # On laisse l'appelant gérer les autres cas non classés.
    return None


# On centralise l'affichage final pour garder un code retour toujours neutre.
def print_error_lines(lines: list[str]) -> int:
    """Affiche une erreur déjà traitée sans faire échouer make."""
    # On écrit sur stderr pour distinguer l'information métier du flux normal.
    for line in lines:
        # On conserve l'ordre exact des lignes préparées en amont.
        print(line, file=sys.stderr)
    # On neutralise le code retour car l'erreur a déjà été expliquée.
    return 0


# On évite un téléchargement inutile si le dataset local est déjà complet.
def ensure_dataset_state(
    data_dir: Path, sentinel: Path, subject_count: int, run_count: int
) -> tuple[bool, int]:
    """Affiche l'état courant du dataset avant tout download."""
    # On valide la structure locale avant de toucher au réseau.
    is_complete, missing_message = check_dataset_complete(
        data_dir,
        subject_count,
        run_count,
    )
    # On matérialise le sentinel seulement si la validation est totale.
    if is_complete:
        # On garantit l'existence du dossier pour un état local cohérent.
        data_dir.mkdir(parents=True, exist_ok=True)
        # On marque explicitement le dataset comme vérifié localement.
        sentinel.touch()
        # On explique pourquoi aucun accès réseau n'a été tenté.
        print(f"Dataset EEGMMIDB déjà complet dans {data_dir} (aucun téléchargement).")
        # On indique à main qu'il peut sortir immédiatement sans erreur.
        return True, 0
    # On affiche la première incohérence locale détectée pour guider l'utilisateur.
    if missing_message is not None:
        # On garde ce message sur stdout comme état courant du dataset.
        print(missing_message)
    # On laisse l'appelant poursuivre vers la phase réseau.
    return False, 0


# On vérifie les dépendances runtime au plus près de leur utilisation réelle.
def ensure_runtime_prerequisites() -> None:
    """Valide les prérequis runtime avant de lancer wget."""
    # On échoue proprement si wget manque sur le poste évalué.
    if shutil.which("wget") is None:
        # On lève une erreur déjà traduite pour conserver un seul flux de sortie.
        raise HandledDownloadError(
            ["❌ wget introuvable. Installez wget puis relancez make download_dataset."]
        )


# On isole la résolution de source pour simplifier le point d'entrée.
def resolve_source_url() -> str:
    """Résout une source officielle utilisable au moment de l'exécution."""
    # On vérifie d'abord l'outil de transfert réellement requis.
    ensure_runtime_prerequisites()
    # On choisit ensuite la première source officielle effectivement joignable.
    return select_official_source()


# On décide une seule fois du message final après wget.
def finalize_download(
    data_dir: Path,
    sentinel: Path,
    args: argparse.Namespace,
    download_result: tuple[int, str],
) -> int:
    """Valide le dataset après wget et imprime le diagnostic final."""
    # On rend les deux composantes explicites pour les branches suivantes.
    wget_status, wget_output = download_result
    # On revérifie le dataset réel plutôt que de faire confiance au code retour.
    is_complete, _ = check_dataset_complete(
        data_dir,
        args.subject_count,
        args.run_count,
    )
    # On préfère l'état des fichiers locaux au statut brut de wget.
    if is_complete:
        # On matérialise le succès validé localement pour les relances futures.
        sentinel.touch()
        # On signale toute incohérence mineure sans invalider un dataset complet.
        if wget_status != 0:
            # On documente l'écart pour un débogage éventuel ultérieur.
            print(
                f"⚠️ wget a retourné {wget_status}, mais le dataset local est complet.",
                file=sys.stderr,
            )
        # On confirme explicitement que la base locale est prête à l'emploi.
        print(f"Dataset EEGMMIDB complet et validé dans {data_dir}.")
        # On termine sans erreur car l'objectif métier est atteint.
        return 0
    # On tente d'abord une traduction métier de l'échec wget.
    handled_lines = classify_wget_error(wget_status, wget_output)
    # On privilégie le message expert déjà prêt si on a classé la panne.
    if handled_lines is not None:
        # On délègue l'affichage standardisé à l'utilitaire dédié.
        return print_error_lines(handled_lines)
    # On garde un fallback explicite pour tout cas encore non classé.
    print(
        "❌ Dataset EEGMMIDB toujours incomplet après téléchargement.",
        file=sys.stderr,
    )
    # On rappelle la reprise wget pour éviter une relance depuis zéro.
    print("Relancez make download_dataset pour reprendre (wget -c).", file=sys.stderr)
    # On n'imprime une trace wget que s'il y a bien eu un échec côté outil.
    if wget_status != 0:
        # On limite la verbosité au bas de trace le plus utile à l'utilisateur.
        tail_lines = [line for line in wget_output.strip().splitlines() if line][-5:]
        # On évite un entête vide quand wget n'a presque rien produit.
        if tail_lines:
            # On annonce clairement la provenance des lignes suivantes.
            print("Diagnostic wget (5 dernières lignes):", file=sys.stderr)
            # On préserve l'ordre natif pour rester fidèle à la sortie réelle.
            for line in tail_lines:
                # On relaie chaque ligne telle quelle pour un debug transparent.
                print(line, file=sys.stderr)
    # On garde un code retour neutre car le cas a été expliqué à l'utilisateur.
    return 0


# On définit une petite CLI pour garder la recette Makefile minimale.
def parse_args() -> argparse.Namespace:
    """Construit la CLI de téléchargement."""
    # On documente le but du script à l'entrée utilisateur.
    parser = argparse.ArgumentParser(
        description="Télécharge le dataset EEGMMIDB depuis les sources officielles."
    )
    # On laisse la destination paramétrable sans exposer l'URL source.
    parser.add_argument(
        "--destination",
        default="data",
        help="Répertoire cible pour les fichiers EDF et EDF.event",
    )
    # On garde ce paramètre pour tester et valider des jeux réduits localement.
    parser.add_argument(
        "--subject-count",
        type=int,
        default=109,
        help="Nombre de sujets attendus pour valider le dataset",
    )
    # On garde ce paramètre pour réutiliser le validateur sur des cas réduits.
    parser.add_argument(
        "--run-count",
        type=int,
        default=14,
        help="Nombre de runs attendus par sujet pour valider le dataset",
    )
    # On renvoie une structure stable pour le point d'entrée principal.
    return parser.parse_args()


# On garde un point d'entrée très court pour limiter les effets de bord.
def main() -> int:
    """Point d'entrée principal pour make download_dataset."""
    # On récupère tous les paramètres avant d'accéder au système de fichiers.
    args = parse_args()
    # On normalise la destination pour partager le même type partout.
    data_dir = Path(args.destination)
    # On place le sentinel à côté du dataset pour refléter l'état local réel.
    sentinel = data_dir / ".eegmmidb.ok"
    # On vérifie d'abord si le dataset local évite tout accès réseau.
    dataset_complete, exit_code = ensure_dataset_state(
        data_dir,
        sentinel,
        args.subject_count,
        args.run_count,
    )
    # On sort immédiatement si la base locale est déjà exploitable.
    if dataset_complete:
        # On respecte le contrat de sortie décidé par le validateur local.
        return exit_code
    # On traduit les erreurs runtime de prérequis et de réseau en messages clairs.
    try:
        # On choisit une source officielle seulement si les prérequis sont satisfaits.
        source_url = resolve_source_url()
    # On capte uniquement les erreurs déjà préparées pour l'utilisateur.
    except HandledDownloadError as error:
        # On réutilise l'affichage standardisé et le code retour neutre.
        return print_error_lines(error.lines)
    # On supprime tout ancien état vert avant un nouveau téléchargement.
    sentinel.unlink(missing_ok=True)
    # On garantit que wget dispose bien d'une destination existante.
    data_dir.mkdir(parents=True, exist_ok=True)
    # On annonce le coût attendu pour éviter une impression de blocage.
    print("Téléchargement EEGMMIDB PhysioNet (~3.4GB), cela peut prendre du temps...")
    # On rend visible la source réellement retenue pour la transparence.
    print(f"Source: {source_url}")
    # On lance wget une seule fois et on garde sa sortie pour le diagnostic.
    download_result = run_wget_download(source_url, data_dir)
    # On délègue toute la décision finale à un validateur unique.
    return finalize_download(data_dir, sentinel, args, download_result)


# On garde un comportement CLI explicite sans effet lors d'un import de tests.
if __name__ == "__main__":
    # On transmet le code retour au shell pour préserver le contrat CLI.
    raise SystemExit(main())

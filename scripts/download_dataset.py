#!/usr/bin/env python3
"""Télécharge EEGMMIDB depuis les sources officielles PhysioNet."""

# Pour exposer une CLI stable utilisable depuis Makefile et les tests.
import argparse

# Pour regrouper des signatures d'erreur hétérogènes sous des règles stables.
import re

# Pour tester la présence de wget sans dépendre d'un shell interactif.
import shutil

# Pour piloter wget et les sondes locales sans ajouter de dépendance réseau.
import subprocess  # nosec B404

# Pour séparer explicitement les erreurs métier du flux normal du terminal.
import sys

# Pour distinguer les erreurs HTTP exploitables des autres pannes réseau.
import urllib.error

# Pour sonder PhysioNet avec la bibliothèque standard et rester léger.
import urllib.request

# Pour figer le contrat des diagnostics et stabiliser les tests associés.
from dataclasses import dataclass

# Pour manipuler les chemins sans ambiguïté de séparateur ou de cwd.
from pathlib import Path

# Pour accepter des doubles de test sans rigidifier inutilement les signatures.
from typing import Any

# Pour limiter la confiance aux seuls endpoints officiels validés du projet.
OFFICIAL_SOURCE_CANDIDATES = (
    "https://physionet.org/files/eegmmidb/1.0.0/",
    "https://physionet.org/static/published-projects/eegmmidb/1.0.0/",
)
# Pour tolérer les serveurs qui refusent HEAD mais servent correctement GET.
HTTP_METHOD_NOT_ALLOWED = 405
# Pour traiter le code réseau de wget comme un signal canonique de panne réseau.
WGET_NETWORK_ERROR_STATUS = 4
# Pour garder un diagnostic cohérent malgré les variantes de messages système.
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
# Pour éviter de qualifier une indisponibilité HTTP comme une panne internet.
HTTP_SOURCE_ERROR_PATTERN = re.compile(
    r"404 Not Found|ERROR 404|403 Forbidden|ERROR 403|400 Bad Request|ERROR 400",
    re.IGNORECASE,
)
# Pour tester la route réseau sans dépendre d'une résolution DNS préalable.
NETWORK_TEST_IP = "1.1.1.1"
# Pour sonder exactement le nom d'hôte réellement requis par le téléchargement.
PHYSIONET_HOSTNAME = "physionet.org"
# Pour empêcher qu'un diagnostic local bloque plus que l'erreur principale.
DIAGNOSTIC_TIMEOUT_SECONDS = 5


# Pour figer un format sérialisable et prédictible des diagnostics locaux.
@dataclass(frozen=True)
class CommandDiagnostic:
    """Résultat structuré d'une commande de diagnostic locale."""

    # Pour garder la commande exacte dans le message sans la reconstruire après coup.
    command: tuple[str, ...]
    # Pour distinguer commande absente, échec système et succès partiel.
    returncode: int | None
    # Pour préserver les outils qui écrivent leur résultat utile sur stdout.
    stdout: str
    # Pour préserver les causes techniques souvent émises sur stderr.
    stderr: str
    # Pour représenter les échecs de lancement sans casser le flux principal.
    failure: str | None = None

    # Pour exposer un rendu CLI lisible sans dupliquer cette logique partout.
    @property
    def display_command(self) -> str:
        # Pour garder un format de log proche de ce que l'utilisateur relancerait.
        return " ".join(self.command)

    # Pour centraliser l'analyse textuelle des diagnostics multi-canaux.
    @property
    def combined_output(self) -> str:
        # Pour éviter qu'une sortie vide ajoute du bruit aux heuristiques réseau.
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


# Pour marquer les erreurs déjà traduites et éviter une double interprétation.
class HandledDownloadError(Exception):
    """Erreur utilisateur déjà traduite en lignes directement affichables."""

    # Pour transporter un message multi-lignes sans perdre sa structure finale.
    def __init__(self, lines: list[str]) -> None:
        # Pour conserver un message exploitable si l'exception remonte en debug.
        super().__init__("\n".join(lines))
        # Pour réimprimer exactement les lignes préparées sans retraitement.
        self.lines = lines


# Pour imposer une représentation d'URL stable avant toute comparaison ou sonde.
def _normalize_candidate(url: str) -> str:
    """Normalise une URL candidate en ajoutant un slash terminal."""
    # Pour neutraliser les espaces parasites issus d'une configuration externe.
    normalized = url.strip()
    # Pour éviter que deux URLs équivalentes divergent dans les probes et les logs.
    if not normalized.endswith("/"):
        # Pour garder une convention canonique lors des concaténations de chemin.
        normalized = f"{normalized}/"
    # Pour fournir à tous les appelants une URL déjà assainie et comparable.
    return normalized


# Pour refuser tout entraînement sur un dataset incomplet mais plausible en surface.
def check_dataset_complete(
    data_dir: Path, subject_count: int, run_count: int
) -> tuple[bool, str | None]:
    """Vérifie la présence des sujets et runs attendus."""
    # Pour échouer tôt quand la racine attendue n'existe même pas.
    if not data_dir.is_dir():
        # Pour remonter immédiatement le premier manque structurel réellement utile.
        return False, f"Dataset incomplet: dossier racine manquant ({data_dir})."
    # Pour garantir que la validation couvre bien tout le périmètre contractuel.
    for subject_index in range(1, subject_count + 1):
        # Pour rester aligné sur la nomenclature canonique de PhysioNet.
        subject = f"S{subject_index:03d}"
        # Pour produire ensuite des messages qui pointent le chemin exact en défaut.
        subject_dir = data_dir / subject
        # Pour arrêter le diagnostic au premier sujet manquant réellement bloquant.
        if not subject_dir.is_dir():
            # Pour fournir un chemin précis à réparer plutôt qu'une erreur générique.
            return False, f"Dataset incomplet: dossier sujet manquant ({subject_dir})."
        # Pour garantir que chaque sujet possède l'ensemble des runs attendus.
        for run_index in range(1, run_count + 1):
            # Pour rester cohérent avec la convention de nommage des runs EEGMMIDB.
            run = f"R{run_index:02d}"
            # Pour protéger le signal brut réellement exploité par le pipeline.
            edf_path = subject_dir / f"{subject}{run}.edf"
            # Pour éviter un faux dataset complet sans événements ni labels temporels.
            event_path = subject_dir / f"{subject}{run}.edf.event"
            # Pour rejeter un fichier absent ou vide avant qu'il casse plus loin.
            if not edf_path.is_file() or edf_path.stat().st_size == 0:
                # Pour cibler précisément l'artefact manquant pour l'utilisateur.
                return (
                    False,
                    f"Dataset incomplet: fichier manquant ou vide ({edf_path}).",
                )
            # Pour garantir que le signal reste exploitable dans le découpage supervisé.
            if not event_path.is_file() or event_path.stat().st_size == 0:
                # Pour cibler exactement l'artefact qui rend le run inexploitable.
                return (
                    False,
                    f"Dataset incomplet: fichier manquant ou vide ({event_path}).",
                )
    # Pour signaler explicitement que la base locale satisfait le contrat attendu.
    return True, None


# Pour éliminer tôt une source officielle indisponible avant un transfert massif.
def probe_source_url(source_url: str, opener: Any | None = None) -> None:
    """Valide qu'une source officielle répond avant le download massif."""
    # Pour permettre des doubles de test sans dépendre du réseau réel.
    http_opener = opener if opener is not None else urllib.request.build_opener()
    # Pour sonder l'existence de la ressource avec un coût minimal côté réseau.
    head_request = urllib.request.Request(source_url, method="HEAD")
    # Pour distinguer proprement les erreurs HTTP des autres échecs de transport.
    try:
        # Pour borner la sonde et éviter un CLI bloqué avant même wget.
        with http_opener.open(head_request, timeout=10):
            # Pour arrêter la recherche dès qu'un endpoint officiel répond sainement.
            return
    # Pour préserver le fallback GET sur les serveurs allergiques à HEAD.
    except urllib.error.HTTPError as error:
        # Pour ne pas masquer une vraie indisponibilité HTTP derrière le fallback.
        if error.code != HTTP_METHOD_NOT_ALLOWED:
            # Pour laisser l'appelant classifier l'erreur avec plus de contexte.
            raise
    # Pour confirmer la disponibilité réelle quand HEAD n'est pas accepté.
    with http_opener.open(source_url, timeout=10):
        # Pour considérer l'endpoint comme valide dès qu'une requête légère aboutit.
        return


# Pour garder un diagnostic local utile sans transformer la sonde en panne fatale.
def run_command_diagnostic(
    command: tuple[str, ...], runner: Any = subprocess.run
) -> CommandDiagnostic:
    """Exécute une commande locale et capture un diagnostic exploitable."""
    # Pour encapsuler aussi les absences d'outil et les timeouts dans un format stable.
    try:
        # Pour conserver les sorties complètes nécessaires au diagnostic utilisateur.
        result = runner(
            # Pour accepter les runners de test qui attendent une séquence mutable.
            list(command),
            # Pour laisser la commande échouer sans interrompre le script principal.
            check=False,
            # Pour analyser la sortie après coup sans bruit terminal parasite.
            capture_output=True,
            # Pour garder des chaînes directement injectables dans les logs utilisateur.
            text=True,
            # Pour empêcher un outil local lent de masquer l'erreur réseau initiale.
            timeout=DIAGNOSTIC_TIMEOUT_SECONDS,
        )
    # Pour signaler l'absence de binaire sans dégrader l'erreur principale.
    except FileNotFoundError:
        # Pour conserver un objet homogène même quand la commande ne démarre pas.
        return CommandDiagnostic(
            # Pour rappeler exactement quelle sonde locale a échoué à se lancer.
            command=command,
            # Pour distinguer clairement ce cas d'un code retour système réel.
            returncode=None,
            # Pour éviter d'inventer une sortie qui n'a jamais existé.
            stdout="",
            # Pour éviter d'inventer une erreur système qui n'a jamais existé.
            stderr="",
            # Pour garder un message directement exploitable côté utilisateur.
            failure=f"commande introuvable: {command[0]}",
        )
    # Pour empêcher une sonde lente de noyer le diagnostic réellement important.
    except subprocess.TimeoutExpired:
        # Pour conserver un objet homogène même quand la sonde dépasse le délai.
        return CommandDiagnostic(
            # Pour rattacher le timeout à la commande réellement concernée.
            command=command,
            # Pour distinguer un timeout d'un code retour explicite.
            returncode=None,
            # Pour éviter de simuler un stdout qui n'est pas fiable.
            stdout="",
            # Pour éviter de simuler un stderr qui n'est pas fiable.
            stderr="",
            # Pour rendre le bloc de diagnostic directement actionnable.
            failure="diagnostic expiré",
        )
    # Pour normaliser la sortie des sondes avant toute logique d'inférence.
    return CommandDiagnostic(
        # Pour préserver l'identité de la commande dans le diagnostic final.
        command=command,
        # Pour garder le signal brut utile aux heuristiques de cause probable.
        returncode=result.returncode,
        # Pour éviter que les fins de ligne perturbent le rendu ou les tests.
        stdout=result.stdout.strip(),
        # Pour éviter que les fins de ligne perturbent le rendu ou les tests.
        stderr=result.stderr.strip(),
    )


# Pour transformer un diagnostic brut en lignes directement injectables aux logs.
def format_command_diagnostic(result: CommandDiagnostic) -> list[str]:
    """Formate un diagnostic local pour le message utilisateur."""
    # Pour court-circuiter tout traitement normal quand la sonde n'a pas démarré.
    if result.failure is not None:
        # Pour afficher une ligne immédiatement lisible et relançable par l'utilisateur.
        return [f"- {result.display_command} -> {result.failure}"]
    # Pour toujours exposer le code retour, même si aucune sortie textuelle n'existe.
    lines = [f"- {result.display_command} -> code retour {result.returncode}"]
    # Pour conserver les cas où l'information utile n'est émise que sur stdout.
    if result.stdout:
        # Pour rattacher explicitement la preuve textuelle à la bonne commande.
        lines.append(f"  stdout: {result.stdout}")
    # Pour conserver les cas où l'information utile n'est émise que sur stderr.
    if result.stderr:
        # Pour rattacher explicitement la preuve textuelle à la bonne commande.
        lines.append(f"  stderr: {result.stderr}")
    # Pour faire de l'absence de sortie un signal explicite plutôt qu'un blanc ambigu.
    if not result.stdout and not result.stderr:
        # Pour éviter qu'un diagnostic vide soit interprété comme un oubli d'affichage.
        lines.append("  sortie: aucune réponse")
    # Pour fournir à l'appelant un bloc déjà cohérent et ordonné.
    return lines


# Pour convertir deux sondes locales en cause probable et action ciblée.
def infer_network_failure_cause(
    ping_result: CommandDiagnostic, dns_result: CommandDiagnostic
) -> tuple[str, str]:
    """Déduit une cause probable et une action à partir des diagnostics."""
    # Pour neutraliser les variations de casse des messages système selon l'outil.
    ping_output = ping_result.combined_output.lower()
    # Pour neutraliser les variations de casse des messages système selon l'outil.
    dns_output = dns_result.combined_output.lower()
    # Pour prioriser une panne de route qui rend aussi le DNS secondairement invalide.
    if "network is unreachable" in ping_output:
        # Pour proposer une action alignée avec une coupure réseau globale observée.
        return (
            "Cause probable: la machine n'a plus d'accès réseau sortant.",
            (
                "Action: rétablissez la connexion réseau "
                "(Wi-Fi, câble, VPN ou routage), puis relancez "
                "make download_dataset."
            ),
        )
    # Pour distinguer un DNS cassé d'un internet globalement indisponible.
    if ping_result.returncode == 0 and not dns_result.stdout:
        # Pour éviter de conseiller à tort une reconnexion physique du poste.
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
    # Pour reconnaître le motif DNS même si ping n'a fourni aucun indice utile.
    if "temporary failure in name resolution" in dns_output:
        # Pour garder une action ciblée malgré une télémétrie locale incomplète.
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
    # Pour terminer sur une hypothèse prudente plutôt qu'un message vide.
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


# Pour enrichir toutes les pannes réseau avec le même bloc de diagnostic local.
def collect_runtime_network_diagnostics(
    runner: Any = subprocess.run,
) -> list[str]:
    """Collecte les diagnostics locaux ajoutés aux erreurs réseau."""
    # Pour séparer une panne de route globale d'une simple panne DNS.
    ping_result = run_command_diagnostic(("ping", "-c", "1", NETWORK_TEST_IP), runner)
    # Pour valider précisément la résolution du nom requis par PhysioNet.
    dns_result = run_command_diagnostic(
        ("getent", "hosts", PHYSIONET_HOSTNAME),
        runner,
    )
    # Pour dériver une synthèse unique et éviter des conseils contradictoires.
    cause_line, action_line = infer_network_failure_cause(ping_result, dns_result)
    # Pour renvoyer un bloc directement concaténable aux messages d'erreur finaux.
    return [
        "Diagnostic local automatique:",
        *format_command_diagnostic(ping_result),
        *format_command_diagnostic(dns_result),
        cause_line,
        action_line,
    ]


# Pour garantir un message réseau homogène quel que soit l'endroit de détection.
def build_network_failure_lines(network_diagnostics: list[str]) -> list[str]:
    """Construit un message réseau complet avec diagnostics locaux."""
    # Pour regrouper symptôme distant et diagnostic local dans un ordre lisible.
    return [
        (
            "❌ Connexion internet indisponible ou instable: impossible "
            "de joindre PhysioNet."
        ),
        *network_diagnostics,
        *collect_runtime_network_diagnostics(),
    ]


# Pour choisir dynamiquement un endpoint officiel réellement joignable.
def select_official_source(
    opener: Any | None = None, probe_func: Any = probe_source_url
) -> str:
    """Choisit dynamiquement une source officielle PhysioNet disponible."""
    # Pour accumuler des preuves réseau si aucun endpoint ne répond.
    network_diagnostics: list[str] = []
    # Pour accumuler séparément les erreurs HTTP qui exigent un autre message.
    http_diagnostics: list[str] = []
    # Pour tester les endpoints dans l'ordre défini comme préférence projet.
    for candidate in OFFICIAL_SOURCE_CANDIDATES:
        # Pour imposer une URL comparable avant affichage et avant sonde.
        normalized_candidate = _normalize_candidate(candidate)
        # Pour conserver la distinction réseau/HTTP portée par urllib.
        try:
            # Pour permettre aux tests d'injecter une sonde contrôlée.
            probe_func(normalized_candidate, opener=opener)
        # Pour mémoriser les erreurs source sans les confondre avec le réseau.
        except urllib.error.HTTPError as error:
            # Pour documenter quel endpoint officiel renvoie quel statut HTTP.
            http_diagnostics.append(
                f"- {normalized_candidate} -> HTTP {error.code} {error.reason}"
            )
        # Pour mémoriser les erreurs réseau avant de construire le message final.
        except urllib.error.URLError as error:
            # Pour conserver la cause exacte remontée par urllib dans le log final.
            network_diagnostics.append(f"- {normalized_candidate} -> {error.reason}")
        # Pour s'arrêter dès qu'une source officielle devient exploitable.
        else:
            # Pour éviter des probes supplémentaires inutiles et rallongeant le CLI.
            return normalized_candidate
    # Pour ne lancer les sondes locales que si l'échec paraît vraiment réseau.
    if network_diagnostics and not http_diagnostics:
        # Pour réutiliser un message réseau homogène déjà standardisé.
        raise HandledDownloadError(build_network_failure_lines(network_diagnostics))
    # Pour distinguer clairement une indisponibilité PhysioNet d'une panne locale.
    raise HandledDownloadError(
        [
            "❌ Sources officielles EEGMMIDB indisponibles sur PhysioNet.",
            *http_diagnostics,
            *network_diagnostics,
            "Action 1: vérifiez que PhysioNet publie bien EEGMMIDB v1.0.0.",
            "Action 2: relancez make download_dataset plus tard.",
        ]
    )


# Pour déléguer le transfert massif à wget et préserver sa reprise native.
def run_wget_download(
    source_url: str, destination_dir: Path, popen_factory: Any = subprocess.Popen
) -> tuple[int, str]:
    """Exécute wget en relayant la sortie vers le terminal."""
    # Pour centraliser les options et stabiliser le comportement CLI du projet.
    command = [
        # Pour répliquer l'arborescence distante utile au dataset complet.
        "wget",
        "-r",
        # Pour éviter de retraiter les fichiers déjà à jour lors des relances.
        "-N",
        # Pour permettre la reprise après coupure réseau ou arrêt volontaire.
        "-c",
        # Pour éviter une attente infinie sur un endpoint défaillant.
        "--tries=3",
        # Pour borner les blocages globaux pendant le transfert.
        "--timeout=20",
        # Pour échouer vite sur une résolution DNS cassée.
        "--dns-timeout=15",
        # Pour échouer vite sur un hôte joignable mais non connectable.
        "--connect-timeout=15",
        # Pour éviter un gel prolongé sur une lecture réseau bloquée.
        "--read-timeout=30",
        # Pour ne pas remonter au-dessus de l'arborescence dataset attendue.
        "-np",
        # Pour éviter d'introduire un préfixe d'hôte dans la structure locale.
        "-nH",
        # Pour réaligner l'arborescence téléchargée sur `data/Sxxx/...`.
        "--cut-dirs=3",
        # Pour restreindre le transfert aux fichiers réellement utiles au pipeline.
        "--accept",
        "*.edf,*.edf.event",
        # Pour exclure les index HTML qui fausseraient la complétude perçue.
        "-R",
        "index.html*",
        # Pour garder tous les fichiers téléchargés sous la destination validée.
        "-P",
        str(destination_dir),
        # Pour laisser visible dans la commande la source réellement retenue.
        source_url,
    ]
    # Pour conserver la sortie brute utile au diagnostic post-échec.
    process = popen_factory(
        # Pour transmettre exactement la stratégie de téléchargement décidée ici.
        command,
        # Pour consommer la sortie ligne à ligne sans shell intermédiaire.
        stdout=subprocess.PIPE,
        # Pour garder un flux unique ordonné et plus simple à afficher.
        stderr=subprocess.STDOUT,
        # Pour relayer au terminal des chaînes directement imprimables.
        text=True,
    )
    # Pour reconstruire ensuite une trace exploitable sans relire un fichier externe.
    output_lines: list[str] = []
    # Pour tolérer les doubles de tests qui ne fournissent pas toujours stdout.
    if process.stdout is not None:
        # Pour éviter qu'un long transfert apparaisse comme un gel silencieux.
        for line in process.stdout:
            # Pour exposer la progression réelle et les erreurs wget au fil de l'eau.
            print(line, end="")
            # Pour rejouer en fin de traitement un diagnostic fidèle à la sortie brute.
            output_lines.append(line)
    # Pour laisser la validation finale s'appuyer sur le statut et la trace complète.
    return process.wait(), "".join(output_lines)


# Pour traduire les codes et sorties wget en messages métier cohérents.
def classify_wget_error(wget_status: int, wget_output: str) -> list[str] | None:
    """Traduit les échecs wget en diagnostics utilisateur actionnables."""
    # Pour éviter de fabriquer un message d'erreur quand wget a réussi.
    if wget_status == 0:
        # Pour laisser la validation de complétude décider seule de l'état final.
        return None
    # Pour s'appuyer d'abord sur la trace réelle quand elle contient un motif fiable.
    diagnostic_match = NETWORK_ERROR_PATTERN.search(wget_output)
    # Pour couvrir aussi les cas où le texte varie mais le code 4 reste stable.
    if diagnostic_match is not None or wget_status == WGET_NETWORK_ERROR_STATUS:
        # Pour privilégier le symptôme réel quand wget en fournit un exploitable.
        diagnostic = (
            diagnostic_match.group(0)
            if diagnostic_match
            else "erreur réseau signalée par wget."
        )
        # Pour enrichir immédiatement la panne réseau avec le diagnostic local.
        return [
            (
                "❌ Dataset EEGMMIDB incomplet: téléchargement interrompu "
                "par une erreur réseau."
            ),
            f"Diagnostic wget: {diagnostic}",
            *collect_runtime_network_diagnostics(),
        ]
    # Pour distinguer une réponse HTTP invalide d'un poste sans internet.
    if HTTP_SOURCE_ERROR_PATTERN.search(wget_output) is not None:
        # Pour orienter l'utilisateur vers la disponibilité de la source officielle.
        return [
            "❌ Sources officielles EEGMMIDB indisponibles sur PhysioNet.",
            (
                "Diagnostic: PhysioNet a répondu avec une erreur HTTP "
                "sur les endpoints officiels."
            ),
            "Action 1: vérifiez que PhysioNet publie bien EEGMMIDB v1.0.0.",
            "Action 2: relancez make download_dataset plus tard.",
        ]
    # Pour laisser un fallback générique traiter les cas encore non classifiés.
    return None


# Pour centraliser l'affichage des erreurs déjà expliquées et garder `make` calme.
def print_error_lines(lines: list[str]) -> int:
    """Affiche une erreur déjà traitée sans faire échouer make."""
    # Pour préserver l'ordre préparé par la logique de diagnostic en amont.
    for line in lines:
        # Pour distinguer l'erreur métier du flux normal du script.
        print(line, file=sys.stderr)
    # Pour garder un code retour neutre quand l'erreur a déjà été rendue actionnable.
    return 0


# Pour éviter tout accès réseau si l'état local satisfait déjà le contrat dataset.
def ensure_dataset_state(
    data_dir: Path, sentinel: Path, subject_count: int, run_count: int
) -> tuple[bool, int]:
    """Affiche l'état courant du dataset avant tout download."""
    # Pour fonder la décision sur l'état réel du disque plutôt que sur un sentinel seul.
    is_complete, missing_message = check_dataset_complete(
        data_dir,
        subject_count,
        run_count,
    )
    # Pour court-circuiter tout téléchargement si le dataset est déjà exploitable.
    if is_complete:
        # Pour garantir l'existence du dossier même après nettoyages partiels externes.
        data_dir.mkdir(parents=True, exist_ok=True)
        # Pour matérialiser explicitement un état local déjà validé.
        sentinel.touch()
        # Pour expliquer pourquoi aucun accès réseau n'est tenté dans ce cas.
        print(f"Dataset EEGMMIDB déjà complet dans {data_dir} (aucun téléchargement).")
        # Pour permettre à `main` de sortir sans logique supplémentaire.
        return True, 0
    # Pour afficher le premier défaut concret avant de passer à la phase réseau.
    if missing_message is not None:
        # Pour laisser visible l'état local ayant motivé la tentative de téléchargement.
        print(missing_message)
    # Pour garder un contrat de retour uniforme avant la phase réseau.
    return False, 0


# Pour vérifier wget au dernier moment utile et réagir au contexte réel du poste.
def ensure_runtime_prerequisites() -> None:
    """Valide les prérequis runtime avant de lancer wget."""
    # Pour échouer avec un message métier plutôt qu'un FileNotFoundError brut.
    if shutil.which("wget") is None:
        # Pour réutiliser le même canal d'erreur propre que les autres cas traités.
        raise HandledDownloadError(
            ["❌ wget introuvable. Installez wget puis relancez make download_dataset."]
        )


# Pour isoler la sélection de source et limiter la complexité du point d'entrée.
def resolve_source_url() -> str:
    """Résout une source officielle utilisable au moment de l'exécution."""
    # Pour éviter d'entamer une résolution réseau si l'outil requis manque déjà.
    ensure_runtime_prerequisites()
    # Pour ne transmettre à wget qu'une source officielle réellement probée.
    return select_official_source()


# Pour décider le message final à partir du résultat réel du disque et de wget.
def finalize_download(
    data_dir: Path,
    sentinel: Path,
    args: argparse.Namespace,
    download_result: tuple[int, str],
) -> int:
    """Valide le dataset après wget et imprime le diagnostic final."""
    # Pour nommer explicitement les deux signaux utilisés dans les branches suivantes.
    wget_status, wget_output = download_result
    # Pour faire primer l'état du dataset sur un code retour potentiellement pessimiste.
    is_complete, _ = check_dataset_complete(
        data_dir,
        args.subject_count,
        args.run_count,
    )
    # Pour considérer l'objectif métier atteint dès que les fichiers attendus sont là.
    if is_complete:
        # Pour matérialiser un état validé localement pour les relances futures.
        sentinel.touch()
        # Pour signaler un décalage utile au debug sans invalider un dataset complet.
        if wget_status != 0:
            # Pour documenter qu'un code retour wget n'est pas toujours bloquant.
            print(
                f"⚠️ wget a retourné {wget_status}, mais le dataset local est complet.",
                file=sys.stderr,
            )
        # Pour confirmer explicitement que le dataset est prêt côté utilisateur.
        print(f"Dataset EEGMMIDB complet et validé dans {data_dir}.")
        # Pour terminer proprement dès que la garantie métier est satisfaite.
        return 0
    # Pour traduire prioritairement les échecs connus en messages actionnables.
    handled_lines = classify_wget_error(wget_status, wget_output)
    # Pour réutiliser un message expert déjà construit quand il existe.
    if handled_lines is not None:
        # Pour préserver un rendu d'erreur uniforme sur tous les cas traités.
        return print_error_lines(handled_lines)
    # Pour garder un fallback explicite quand aucun classifieur ne reconnaît l'échec.
    print(
        "❌ Dataset EEGMMIDB toujours incomplet après téléchargement.",
        file=sys.stderr,
    )
    # Pour rappeler la reprise native de wget au lieu d'une relance depuis zéro.
    print("Relancez make download_dataset pour reprendre (wget -c).", file=sys.stderr)
    # Pour n'afficher une trace wget qu'en présence d'un échec côté outil.
    if wget_status != 0:
        # Pour borner la verbosité à la portion la plus utile au diagnostic humain.
        tail_lines = [line for line in wget_output.strip().splitlines() if line][-5:]
        # Pour éviter un en-tête de diagnostic vide quand wget n'a rien produit.
        if tail_lines:
            # Pour signaler clairement l'origine des lignes affichées ensuite.
            print("Diagnostic wget (5 dernières lignes):", file=sys.stderr)
            # Pour conserver l'ordre natif de la sortie lors du débogage manuel.
            for line in tail_lines:
                # Pour restituer la preuve brute sans la déformer par un retraitement.
                print(line, file=sys.stderr)
    # Pour garder un code retour neutre quand le cas a déjà été expliqué.
    return 0


# Pour offrir une interface scriptable sans réexposer la logique réseau au Makefile.
def parse_args() -> argparse.Namespace:
    """Construit la CLI de téléchargement."""
    # Pour documenter l'intention globale dès `--help`.
    parser = argparse.ArgumentParser(
        description="Télécharge le dataset EEGMMIDB depuis les sources officielles."
    )
    # Pour laisser varier la cible locale sans ouvrir la porte au changement de source.
    parser.add_argument(
        "--destination",
        default="data",
        help="Répertoire cible pour les fichiers EDF et EDF.event",
    )
    # Pour réutiliser le validateur sur des jeux réduits pendant les tests locaux.
    parser.add_argument(
        "--subject-count",
        type=int,
        default=109,
        help="Nombre de sujets attendus pour valider le dataset",
    )
    # Pour réutiliser le validateur sur des échantillons de runs en environnement test.
    parser.add_argument(
        "--run-count",
        type=int,
        default=14,
        help="Nombre de runs attendus par sujet pour valider le dataset",
    )
    # Pour fournir au point d'entrée un objet stable déjà validé par argparse.
    return parser.parse_args()


# Pour garder la coordination globale lisible et limiter les effets de bord.
def main() -> int:
    """Point d'entrée principal pour make download_dataset."""
    # Pour figer les paramètres d'exécution avant toute interaction système.
    args = parse_args()
    # Pour partager un type de chemin unique dans tout le script.
    data_dir = Path(args.destination)
    # Pour matérialiser l'état validé au plus près du dataset concerné.
    sentinel = data_dir / ".eegmmidb.ok"
    # Pour éviter tout accès réseau si le disque satisfait déjà le contrat.
    dataset_complete, exit_code = ensure_dataset_state(
        data_dir,
        sentinel,
        args.subject_count,
        args.run_count,
    )
    # Pour sortir immédiatement dès que la base locale est déjà exploitable.
    if dataset_complete:
        # Pour respecter le contrat de retour décidé par la validation locale.
        return exit_code
    # Pour transformer les erreurs runtime attendues en messages déjà prêts.
    try:
        # Pour ne lancer wget qu'après sélection d'une source officielle joignable.
        source_url = resolve_source_url()
    # Pour ne capturer ici que les erreurs déjà traduites pour l'utilisateur.
    except HandledDownloadError as error:
        # Pour garder une sortie uniforme et un code retour neutre.
        return print_error_lines(error.lines)
    # Pour éviter qu'un ancien état vert survive à un download ensuite incomplet.
    sentinel.unlink(missing_ok=True)
    # Pour garantir à wget une destination existante et stable.
    data_dir.mkdir(parents=True, exist_ok=True)
    # Pour expliciter le coût attendu et éviter l'impression de gel du CLI.
    print("Téléchargement EEGMMIDB PhysioNet (~3.4GB), cela peut prendre du temps...")
    # Pour rendre visible la source réellement retenue au moment de l'exécution.
    print(f"Source: {source_url}")
    # Pour capturer une seule fois le statut et la trace brute du transfert.
    download_result = run_wget_download(source_url, data_dir)
    # Pour centraliser toute décision finale dans un unique validateur de sortie.
    return finalize_download(data_dir, sentinel, args, download_result)


# Pour préserver un import sans effet de bord dans les tests et les autres modules.
if __name__ == "__main__":
    # Pour transmettre au shell le contrat de sortie décidé par `main`.
    raise SystemExit(main())

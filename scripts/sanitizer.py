#!/usr/bin/env python3

"""CLI pour exécuter un faisceau de sondes benchmark / diagnostic."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import IO, Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND = ("make", "-j1", "mybci", "wavelet")
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "sanitizer"
POETRY_PREFIX = ("poetry", "run")
BASH_EXECUTABLE = "/usr/bin/bash"
SUDO_EXECUTABLE = "/usr/bin/sudo"
PAIR_ITEMS = 2
TRIPLE_ITEMS = 3
SKIP_EXIT_CODE = 90
PERF_PARANOID_BLOCKED_THRESHOLD = 2
NON_FATAL_PROBE_STATUSES = frozenset({"ok", "warn", "skipped"})
DEFAULT_PYSPY_DURATION_SECONDS = 3


@dataclass(frozen=True)
class SanitizerConfig:
    """Paramètres d'exécution du sanitizer."""

    command: tuple[str, ...]
    direct_python_command: tuple[str, ...] | None
    output_dir: Path
    selected_probes: tuple[str, ...]
    timeout_seconds: int
    pyperf_warmups: int
    pyperf_runs: int
    hyperfine_warmups: int
    hyperfine_runs: int
    cpu_core: int
    ps_interval_seconds: float
    psrecord_interval_seconds: float
    memory_limit_kib: int
    time_csv_runs: int
    allow_privileged_tools: bool = False
    pyspy_duration_seconds: int = DEFAULT_PYSPY_DURATION_SECONDS


@dataclass(frozen=True)
class PreflightResult:
    """Résultat d'un contrôle de prérequis."""

    ready: bool
    summary: str | None = None
    action: str | None = None
    action_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeDefinition:
    """Décrit une sonde exécutable."""

    identifier: str
    title: str
    category: str
    highlights: Sequence[str]
    command_builder: Callable[[SanitizerConfig, Path], str] | None = None
    preflight: Callable[[SanitizerConfig], PreflightResult] | None = None
    runner: Callable[[SanitizerConfig, "ProbeDefinition"], "ProbeResult"] | None = None
    allow_nonzero_exit: bool = False
    timeout_resolver: Callable[[SanitizerConfig], int] | None = None


@dataclass
class ProbeResult:
    """Résultat sérialisable d'une sonde."""

    identifier: str
    title: str
    category: str
    status: str
    summary: str
    action: str | None
    exit_code: int | None
    command: str | None
    output_dir: str
    stdout_log: str | None
    stderr_log: str | None
    artifacts: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    action_commands: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimeCsvRow:
    """Ligne agrégée produite par la sonde A2."""

    run: int
    wall_s: float
    user_s: float
    sys_s: float
    cpu: str
    rss_kb: int
    exit: int


@dataclass(frozen=True)
class TimedRunResult:
    """Résultat intermédiaire d'une itération A2."""

    stdout: str
    stderr: str
    row: TimeCsvRow | None
    action: str | None = None


@dataclass(frozen=True)
class ShellProbeExecution:
    """Résultat brut d'une commande shell de sonde."""

    stdout: str
    stderr: str
    exit_code: int | None
    status: str
    summary: str
    action: str | None
    action_commands: tuple[str, ...]


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / _timestamp_slug()


def _shell_join(tokens: Sequence[str]) -> str:
    return shlex.join(list(tokens))


def _enable_make_sanitizer_mode(command: Sequence[str]) -> tuple[str, ...]:
    """Injecte un flag make dédié pour supprimer les effets de bord du sanitizer."""

    tokens = list(command)
    if not tokens or tokens[0] != "make":
        return tuple(tokens)
    if "TPV_SANITIZER=1" in tokens:
        return tuple(tokens)

    goals = [token for token in tokens[1:] if not token.startswith("-")]
    if "mybci" not in goals:
        return tuple(tokens)

    option_count = 0
    for token in tokens[1:]:
        if token.startswith("-"):
            option_count += 1
            continue
        break
    insert_index = 1 + option_count
    return tuple(tokens[:insert_index] + ["TPV_SANITIZER=1"] + tokens[insert_index:])


def _target_shell(config: SanitizerConfig) -> str:
    return _shell_join(_enable_make_sanitizer_mode(config.command))


def _with_poetry(command: Sequence[str]) -> tuple[str, ...]:
    if tuple(command[: len(POETRY_PREFIX)]) == POETRY_PREFIX:
        return tuple(command)
    return (*POETRY_PREFIX, *command)


def _poetry_target_shell(config: SanitizerConfig) -> str:
    return _shell_join(_with_poetry(_enable_make_sanitizer_mode(config.command)))


def _preferred_profile_shell(config: SanitizerConfig) -> str:
    if config.direct_python_command is not None:
        return _shell_join(config.direct_python_command)
    return _target_shell(config)


def _bash_shell(shell_command: str) -> str:
    return f"bash -lc {shlex.quote(shell_command)}"


def _quote_path(path: Path) -> str:
    return shlex.quote(str(path))


def _inspect_path_command(path: Path, line_count: int = 120) -> str:
    return f"sed -n '1,{line_count}p' {_quote_path(path)}"


def _dedupe_commands(commands: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for command in commands:
        if command in seen:
            continue
        seen.add(command)
        ordered.append(command)
    return tuple(ordered)


def _status_wrapped(
    command: str,
    summary: str,
    action: str | None = None,
    action_commands: Sequence[str] = (),
    post_actions: Sequence[str] = (),
) -> str:
    lines = [
        "status=0",
        f"{command} || status=$?",
    ]
    lines.extend(post_actions)
    lines.append(
        f"printf 'SUMMARY: %s (exit=%s)\\n' {shlex.quote(summary)} \"$status\""
    )
    if action is not None or action_commands:
        action_payload: list[str] = []
        if action is not None:
            action_payload.append(f"printf 'ACTION: %s\\n' {shlex.quote(action)}; ")
        for action_command in action_commands:
            action_payload.append(
                f"printf 'ACTION_CMD: %s\\n' {shlex.quote(action_command)}; "
            )
        lines.append('if [[ "$status" -ne 0 ]]; then ' + "".join(action_payload) + "fi")
    lines.append('exit "$status"')
    return "; ".join(lines)


def _run_subprocess(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout is None and stderr is None:
        return subprocess.run(  # nosec B603
            list(command),
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
            check=False,
        )
    return subprocess.run(  # nosec B603
        list(command),
        cwd=REPO_ROOT,
        env=env,
        timeout=timeout,
        text=True,
        stdout=stdout,
        stderr=stderr,
        check=False,
    )


def _check_output_subprocess(command: Sequence[str]) -> str:
    return subprocess.check_output(  # nosec B603
        list(command),
        cwd=REPO_ROOT,
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _probe_timeout(config: SanitizerConfig, probe: ProbeDefinition) -> int:
    if probe.timeout_resolver is not None:
        return probe.timeout_resolver(config)
    return config.timeout_seconds


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_tagged_line(payload: str, prefix: str) -> str | None:
    for line in reversed(payload.splitlines()):
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _extract_tagged_lines(payload: str, prefix: str) -> list[str]:
    return [
        line[len(prefix) :].strip()
        for line in payload.splitlines()
        if line.startswith(prefix)
    ]


def _collect_artifacts(probe_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for path in sorted(probe_dir.rglob("*")):
        if path.is_file():
            artifacts.append(str(path.relative_to(probe_dir)))
    return artifacts


def _finalize_result(probe_dir: Path, result: ProbeResult) -> ProbeResult:
    _write_text(probe_dir / "result.json", json.dumps(asdict(result), indent=2))
    result.artifacts = _collect_artifacts(probe_dir)
    _write_text(probe_dir / "result.json", json.dumps(asdict(result), indent=2))
    return result


def _build_skip_result(
    config: SanitizerConfig,
    probe: ProbeDefinition,
    summary: str,
    action: str | None,
    action_commands: Sequence[str] = (),
) -> ProbeResult:
    probe_dir = config.output_dir / probe.identifier
    probe_dir.mkdir(parents=True, exist_ok=True)
    result = ProbeResult(
        identifier=probe.identifier,
        title=probe.title,
        category=probe.category,
        status="skipped",
        summary=summary,
        action=action,
        action_commands=list(action_commands),
        exit_code=SKIP_EXIT_CODE,
        command=None,
        output_dir=str(probe_dir),
        stdout_log=None,
        stderr_log=None,
        highlights=list(probe.highlights),
    )
    return _finalize_result(probe_dir, result)


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def _python_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_command(
    command: str,
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(_config: SanitizerConfig) -> PreflightResult:
        if _command_available(command):
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary=f"Outil absent: `{command}`",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _require_path(
    path: Path,
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(_config: SanitizerConfig) -> PreflightResult:
        if path.exists():
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary=f"Chemin absent: `{path}`",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _require_python_module(
    module_name: str,
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(_config: SanitizerConfig) -> PreflightResult:
        if _python_module_available(module_name):
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary=f"Module Python absent: `{module_name}`",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _require_poetry(
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(config: SanitizerConfig) -> PreflightResult:
        if tuple(config.command[: len(POETRY_PREFIX)]) == POETRY_PREFIX:
            return PreflightResult(ready=True)
        if _command_available("poetry"):
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary="Poetry absent pour la variante `poetry run`.",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _require_direct_python(
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(config: SanitizerConfig) -> PreflightResult:
        if config.direct_python_command is not None:
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary="Commande Python directe introuvable pour ce target.",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _resolve_python_invocation(
    command: Sequence[str],
) -> tuple[list[str], list[str]] | None:
    tokens = list(command)
    prefix: list[str] = []
    if tuple(tokens[: len(POETRY_PREFIX)]) == POETRY_PREFIX:
        prefix = tokens[: len(POETRY_PREFIX)]
        tokens = tokens[len(POETRY_PREFIX) :]
    if not tokens:
        return None
    executable_name = Path(tokens[0]).name
    if not executable_name.startswith("python"):
        return None
    return prefix, tokens


def _resolve_python_script_invocation(command: Sequence[str]) -> list[str] | None:
    resolved = _resolve_python_invocation(command)
    if resolved is None:
        return None
    _prefix, python_tokens = resolved
    if len(python_tokens) < PAIR_ITEMS:
        return None
    script_path = python_tokens[1]
    if script_path.startswith("-"):
        return None
    return python_tokens


def _require_direct_python_script(
    action: str,
    action_commands: Sequence[str] = (),
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(config: SanitizerConfig) -> PreflightResult:
        if config.direct_python_command is None:
            return PreflightResult(
                ready=False,
                summary="Commande Python directe introuvable pour ce target.",
                action=action,
                action_commands=tuple(action_commands),
            )
        if _resolve_python_script_invocation(config.direct_python_command) is not None:
            return PreflightResult(ready=True)
        return PreflightResult(
            ready=False,
            summary="Commande Python directe incompatible: un script `.py` est requis.",
            action=action,
            action_commands=tuple(action_commands),
        )

    return _check


def _read_perf_event_paranoid() -> int | None:
    return _read_linux_int(Path("/proc/sys/kernel/perf_event_paranoid"))


def _read_linux_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_ptrace_scope() -> int | None:
    return _read_linux_int(Path("/proc/sys/kernel/yama/ptrace_scope"))


def _sudo_non_interactive_available() -> bool:
    try:
        completed = _run_subprocess(
            [SUDO_EXECUTABLE, "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    return completed.returncode == 0


def _sudo_env_prefix() -> str:
    return f"{SUDO_EXECUTABLE} -n env PATH={shlex.quote(os.environ.get('PATH', ''))}"


def _sudo_chown_command(paths: Sequence[Path]) -> str:
    owner = f"{os.getuid()}:{os.getgid()}"
    quoted_paths = " ".join(_quote_path(path) for path in paths)
    return f"sudo -n chown {owner} {quoted_paths}"


def _should_use_privileged_perf(config: SanitizerConfig) -> bool:
    perf_event_paranoid = _read_perf_event_paranoid()
    return (
        config.allow_privileged_tools
        and perf_event_paranoid is not None
        and perf_event_paranoid > PERF_PARANOID_BLOCKED_THRESHOLD
        and _sudo_non_interactive_available()
    )


def _perf_command_prefix(config: SanitizerConfig) -> str:
    if _should_use_privileged_perf(config):
        return f"{_sudo_env_prefix()} perf"
    return "perf"


def _extract_pyspy_sample_count(payload: str) -> int | None:
    for line in payload.splitlines():
        if "Samples:" not in line:
            continue
        sample_fragment = line.split("Samples:", 1)[1].strip().split(" ", 1)[0]
        try:
            return int(sample_fragment)
        except ValueError:
            return None
    return None


def _normalize_pyspy_result(
    result: ProbeResult,
    probe_dir: Path,
    stdout: str,
) -> ProbeResult:
    if result.status != "warn":
        return result
    svg_path = probe_dir / "pyspy.svg"
    sample_count = _extract_pyspy_sample_count(stdout)
    if sample_count is None or sample_count <= 0 or not svg_path.exists():
        return result
    result.status = "ok"
    result.summary = (
        "Profil py-spy terminé "
        f"({sample_count} samples capturés, artefact SVG valide)."
    )
    result.action = None
    result.action_commands = []
    result.exit_code = 0
    return result


def _require_perf(
    missing_action: str,
    missing_commands: Sequence[str],
    blocked_action: str,
    blocked_commands: Sequence[str],
) -> Callable[[SanitizerConfig], PreflightResult]:
    def _check(config: SanitizerConfig) -> PreflightResult:
        if not _command_available("perf"):
            return PreflightResult(
                ready=False,
                summary="Outil absent: `perf`",
                action=missing_action,
                action_commands=tuple(missing_commands),
            )
        perf_event_paranoid = _read_perf_event_paranoid()
        if (
            perf_event_paranoid is not None
            and perf_event_paranoid > PERF_PARANOID_BLOCKED_THRESHOLD
        ):
            privileged_commands = (
                "sudo -v",
                (
                    "make sanitizer SANITIZER_ALLOW_PRIVILEGED_TOOLS=1 "
                    "SANITIZER_ARGS='--probe P1.make --probe P1.poetry "
                    "--probe P2 --probe P3'"
                ),
            )
            if config.allow_privileged_tools:
                if _sudo_non_interactive_available():
                    return PreflightResult(ready=True)
                return PreflightResult(
                    ready=False,
                    summary=(
                        "`perf` détecté mais bloqué par "
                        f"`perf_event_paranoid={perf_event_paranoid}` et "
                        "`sudo -n` n'est pas prêt"
                    ),
                    action=(
                        "Amorcez les credentials sudo avec `sudo -v`, puis "
                        "relancez en mode privilégié ou utilisez les fallbacks "
                        "utilisateur."
                    ),
                    action_commands=_dedupe_commands(
                        privileged_commands + tuple(blocked_commands)
                    ),
                )
            return PreflightResult(
                ready=False,
                summary=(
                    "`perf` détecté mais bloqué par "
                    f"`perf_event_paranoid={perf_event_paranoid}`"
                ),
                action=blocked_action,
                action_commands=_dedupe_commands(
                    tuple(blocked_commands) + privileged_commands
                ),
            )
        return PreflightResult(ready=True)

    return _check


def _extract_make_goals(command: Sequence[str]) -> list[str] | None:
    if not command or command[0] != "make":
        return None
    goals = [token for token in command[1:] if not token.startswith("-")]
    return goals or None


def _append_feature_strategy(tokens: list[str], extras: Sequence[str]) -> list[str]:
    feature_choices = {"fft", "welch", "wavelet", "pca", "csp", "cssp", "svd"}
    if extras and extras[0] in feature_choices:
        tokens.extend(["--feature-strategy", extras[0]])
    return tokens


def _infer_pair_script_command(
    script_path: str,
    extras: Sequence[str],
) -> list[str] | None:
    if len(extras) < PAIR_ITEMS:
        return None
    return [sys.executable, script_path, extras[0], extras[1]]


def _infer_make_target_command(target: str, extras: Sequence[str]) -> list[str] | None:
    command: list[str] | None = None
    if target == "mybci":
        command = _append_feature_strategy([sys.executable, "mybci.py"], extras)
    elif target == "compute-mean-of-means":
        command = [sys.executable, "scripts/aggregate_experience_scores.py"]
    elif target in {"train", "predict"}:
        command = _infer_pair_script_command(f"scripts/{target}.py", extras)
        if command is None:
            return None
        command = _append_feature_strategy(command, extras[PAIR_ITEMS:])
    elif target == "visualizer":
        command = _infer_pair_script_command(
            "scripts/visualize_raw_filtered.py",
            extras,
        )
    elif target == "realtime":
        command = _infer_pair_script_command("src/tpv/realtime.py", extras)
    return command


def infer_python_command(command: Sequence[str]) -> list[str] | None:
    """Tente d'inférer une commande Python directe à profiler."""

    if not command:
        return None
    if command[0] != "make":
        return list(command)

    goals = _extract_make_goals(command)
    if not goals:
        return None

    target = goals[0]
    extras = goals[1:]
    return _infer_make_target_command(target, extras)


def build_parser() -> argparse.ArgumentParser:
    """Construit la CLI du sanitizer."""

    parser = argparse.ArgumentParser(
        description=(
            "Exécute un ensemble de sondes benchmark / diagnostic autour d'une "
            "commande cible TPV."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Répertoire de sortie pour les logs et artefacts.",
    )
    parser.add_argument(
        "--probe",
        dest="probes",
        action="append",
        default=None,
        help="Identifiant de sonde à exécuter. Répétez l'option pour filtrer.",
    )
    parser.add_argument(
        "--list-probes",
        action="store_true",
        help="Liste les sondes disponibles puis quitte.",
    )
    parser.add_argument(
        "--python-command",
        type=str,
        default=None,
        help=(
            "Commande Python directe pour les sondes F*. "
            'Ex: "python mybci.py --feature-strategy wavelet"'
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout de base par sonde.",
    )
    parser.add_argument(
        "--pyperf-warmups",
        type=int,
        default=3,
        help="Nombre de warmups pour `pyperf`.",
    )
    parser.add_argument(
        "--pyperf-runs",
        type=int,
        default=20,
        help="Nombre de runs pour `pyperf`.",
    )
    parser.add_argument(
        "--hyperfine-warmups",
        type=int,
        default=3,
        help="Nombre de warmups pour `hyperfine`.",
    )
    parser.add_argument(
        "--hyperfine-runs",
        type=int,
        default=20,
        help="Nombre de runs pour `hyperfine`.",
    )
    parser.add_argument(
        "--cpu-core",
        type=int,
        default=2,
        help="Coeur CPU utilisé par `taskset`.",
    )
    parser.add_argument(
        "--ps-interval-seconds",
        type=float,
        default=1.0,
        help="Intervalle de sampling pour `ps`.",
    )
    parser.add_argument(
        "--psrecord-interval-seconds",
        type=float,
        default=0.2,
        help="Intervalle de sampling pour `psrecord`.",
    )
    parser.add_argument(
        "--memory-limit-kib",
        type=int,
        default=2 * 1024 * 1024,
        help="Limite `ulimit -v` en KiB pour la sonde C1.",
    )
    parser.add_argument(
        "--time-csv-runs",
        type=int,
        default=1,
        help="Nombre de runs pour la sonde A2 et ses exports CSV / JSONL.",
    )
    parser.add_argument(
        "--allow-privileged-tools",
        action="store_true",
        help=(
            "Autorise les sondes système à utiliser `sudo -n` après un `sudo -v` "
            "préparatoire dans le shell appelant."
        ),
    )
    parser.add_argument(
        "--pyspy-duration-seconds",
        type=int,
        default=DEFAULT_PYSPY_DURATION_SECONDS,
        help=(
            "Durée d'échantillonnage pour F3/py-spy. Une durée bornée évite les "
            "faux négatifs si le process se termine juste avant la fin de capture."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=("Commande cible après `--`. Défaut: `make -j1 mybci wavelet`."),
    )
    return parser


def _build_time_verbose_command(target_shell: str) -> str:
    return _status_wrapped(
        f"/usr/bin/time -v {_bash_shell(target_shell)}",
        "Capture GNU time verbose terminée",
        (
            "Consultez stderr.log pour Elapsed, CPU%, RSS max, faults et "
            "context switches."
        ),
    )


def _build_pyperf_command(target_shell: str, config: SanitizerConfig) -> str:
    python_bin = shlex.quote(sys.executable)
    return _status_wrapped(
        (
            f"{python_bin} -m pyperf command "
            "--processes 1 "
            f"-w {config.pyperf_warmups} "
            f"-n {config.pyperf_runs} "
            "--loops 1 "
            f"-- bash -lc {shlex.quote(target_shell)}"
        ),
        "Benchmark pyperf terminé",
        ("Inspectez la médiane, l'écart-type et les outliers dans stdout.log."),
    )


def _build_hyperfine_command(target_shell: str, config: SanitizerConfig) -> str:
    return _status_wrapped(
        (
            "hyperfine "
            f"--warmup {config.hyperfine_warmups} "
            f"--runs {config.hyperfine_runs} "
            f"{shlex.quote(target_shell)}"
        ),
        "Benchmark hyperfine terminé",
        "Comparez la médiane et la variance dans stdout.log.",
    )


def _build_taskset_command(target_shell: str, config: SanitizerConfig) -> str:
    return _status_wrapped(
        f"taskset -c {config.cpu_core} {_bash_shell(target_shell)}",
        "Exécution taskset terminée",
        "Comparez la stabilité du wall time avec et sans pin CPU.",
    )


def _build_b4_command(config: SanitizerConfig, probe_dir: Path) -> str:
    sample_path = probe_dir / "ps_top.txt"
    return (
        "status=0; "
        "( while :; do "
        "ps -eo pid,ppid,cmd,%cpu,%mem,rss --sort=-rss | head -n 15; "
        f"sleep {config.ps_interval_seconds}; "
        f"done ) > {_quote_path(sample_path)} 2>&1 & "
        "sampler=$!; "
        f"{_bash_shell(_target_shell(config))} || status=$?; "
        'kill "$sampler" 2>/dev/null || true; '
        'wait "$sampler" 2>/dev/null || true; '
        "printf 'SUMMARY: %s (exit=%s)\\n' "
        "'Sampling ps terminé' \"$status\"; "
        'if [[ "$status" -ne 0 ]]; then '
        "printf 'ACTION: %s\\n' "
        "'Inspectez ps_top.txt puis corrigez la commande cible si elle échoue.'; "
        "fi; "
        'exit "$status"'
    )


def _build_c1_command(config: SanitizerConfig) -> str:
    return _status_wrapped(
        f"ulimit -v {config.memory_limit_kib}; {_bash_shell(_target_shell(config))}",
        "Exécution sous contrainte mémoire terminée",
        (
            "Un code non nul peut être normal ici. Cherchez MemoryError, OOM ou "
            "des messages d'allocation explicites."
        ),
    )


def _build_c2_command(config: SanitizerConfig, probe_dir: Path) -> str:
    data_path = probe_dir / "mprof.dat"
    plot_path = probe_dir / "mprof.png"
    stderr_path = probe_dir / "stderr.log"
    return _status_wrapped(
        (
            "MPLBACKEND=Agg mprof run --include-children "
            "--exit-code "
            f"--output {_quote_path(data_path)} "
            f"bash -lc {shlex.quote(_target_shell(config))} "
            f"&& mprof peak {_quote_path(data_path)} "
            f"&& MPLBACKEND=Agg mprof plot {_quote_path(data_path)} "
            f"--output {_quote_path(plot_path)}"
        ),
        "Profil mémoire mprof terminé",
        (
            "Consultez mprof.dat, le pic RSS et le graphe PNG pour localiser un "
            "pic ou une fuite mémoire."
        ),
        action_commands=(
            _inspect_path_command(stderr_path),
            f"test -f {_quote_path(data_path)} && mprof peak {_quote_path(data_path)}",
            f"test -f {_quote_path(plot_path)} && ls -lh {_quote_path(plot_path)}",
        ),
    )


def _build_c3_command(config: SanitizerConfig, probe_dir: Path) -> str:
    csv_path = probe_dir / "psrecord.csv"
    plot_path = probe_dir / "psrecord.png"
    pid_path = probe_dir / "target.pid"
    return (
        "status=0; target_status=0; "
        f"( {_bash_shell(_target_shell(config))} ) & pid=$!; "
        f"printf '%s\\n' \"$pid\" > {_quote_path(pid_path)}; "
        "MPLBACKEND=Agg psrecord "
        f'"$pid" --interval {config.psrecord_interval_seconds} '
        f"--log {_quote_path(csv_path)} --plot {_quote_path(plot_path)} || status=$?; "
        'wait "$pid" || target_status=$?; '
        'if [[ "$status" -eq 0 && "$target_status" -ne 0 ]]; then '
        'status="$target_status"; '
        "fi; "
        "printf 'SUMMARY: %s (exit=%s)\\n' "
        "'Capture psrecord terminée' \"$status\"; "
        'if [[ "$status" -ne 0 ]]; then '
        "printf 'ACTION: %s\\n' "
        "'Si le PID surveillé est trop court ou peu représentatif, "
        "préférez C2/mprof --include-children.'; "
        "fi; "
        'exit "$status"'
    )


def _build_d1_command(config: SanitizerConfig, probe_dir: Path) -> str:
    proc_io_path = probe_dir / "proc_io.txt"
    target_shell = _target_shell(config)
    return (
        "if command -v pidstat >/dev/null 2>&1; then "
        + _status_wrapped(
            f"pidstat -d -h 1 -- {_bash_shell(target_shell)}",
            "Capture pidstat terminée",
            "Inspectez les débits disque et l'iowait dans stdout.log.",
        )
        + "; fi; "
        "status=0; "
        f"( {_bash_shell(target_shell)} ) & pid=$!; "
        "sleep 0.2; "
        'if [[ -r "/proc/$pid/io" ]]; then '
        f'cat "/proc/$pid/io" | tee {_quote_path(proc_io_path)}; '
        "else "
        "printf 'SUMMARY: %s\\n' 'Fallback /proc/<PID>/io indisponible'; "
        "printf 'ACTION: %s\\n' "
        "'Relancez avec strace (D2) si /proc/<PID>/io devient inaccessible.'; "
        'wait "$pid" || status=$?; '
        'exit "$status"; '
        "fi; "
        'wait "$pid" || status=$?; '
        "printf 'SUMMARY: %s (exit=%s)\\n' "
        "'Fallback /proc/<PID>/io terminé' \"$status\"; "
        'if [[ "$status" -ne 0 ]]; then '
        "printf 'ACTION: %s\\n' "
        "'Inspectez proc_io.txt puis corrigez la commande cible si elle échoue.'; "
        "fi; "
        'exit "$status"'
    )


def _build_d2_command(config: SanitizerConfig, probe_dir: Path) -> str:
    log_path = probe_dir / "strace_file_timing.log"
    return _status_wrapped(
        (
            "strace -f -tt -T -e trace=%file "
            f"-o {_quote_path(log_path)} "
            f"{_bash_shell(_target_shell(config))}"
        ),
        "Trace strace fichier + durée terminée",
        "Inspectez strace_file_timing.log pour repérer les accès lents.",
    )


def _build_e1_command(config: SanitizerConfig, probe_dir: Path) -> str:
    log_path = probe_dir / "strace_permissions.log"
    return _status_wrapped(
        (
            "strace -f -e trace=%file -tt -s 200 "
            f"-o {_quote_path(log_path)} "
            f"{_bash_shell(_target_shell(config))}"
        ),
        "Trace strace permissions terminée",
        "Cherchez EACCES, ENOENT et la séquence de résolution de chemins.",
    )


def _build_e2_command(config: SanitizerConfig, probe_dir: Path) -> str:
    csv_path = probe_dir / "fd_counts.csv"
    return (
        "status=0; "
        f"( {_bash_shell(_target_shell(config))} ) & pid=$!; "
        f"printf 'sample,fd_count\\n' > {_quote_path(csv_path)}; "
        "sample=0; "
        'while kill -0 "$pid" 2>/dev/null; do '
        'count=$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l); '
        f'printf \'%s,%s\\n\' "$sample" "$count" >> {_quote_path(csv_path)}; '
        "sample=$((sample+1)); "
        "sleep 0.2; "
        "done; "
        'wait "$pid" || status=$?; '
        "printf 'SUMMARY: %s (exit=%s)\\n' "
        "'Capture nombre de FD terminée' \"$status\"; "
        'if [[ "$status" -ne 0 ]]; then '
        "printf 'ACTION: %s\\n' "
        "'Inspectez fd_counts.csv pour détecter une fuite de descripteurs.'; "
        "fi; "
        'exit "$status"'
    )


def _build_e3_command(probe_dir: Path) -> str:
    snapshot_path = probe_dir / "permissions_snapshot.txt"
    return _status_wrapped(
        (
            "find src/tpv -maxdepth 2 -printf '%m %u:%g %p\\n' | "
            f"head -n 40 | tee {_quote_path(snapshot_path)}"
        ),
        "Audit permissions src/tpv terminé",
        "Utilisez permissions_snapshot.txt pour guider un chmod ciblé.",
    )


def _build_cprofile_command(config: SanitizerConfig, probe_dir: Path) -> str:
    if config.direct_python_command is None:
        return (
            f"printf 'SUMMARY: %s\\n' 'Commande Python directe absente'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote('Passez --python-command pour activer F1/cProfile.')}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    resolved = _resolve_python_invocation(config.direct_python_command)
    if resolved is None:
        action_message = (
            "Passez une commande commençant par python ou poetry run python."
        )
        return (
            f"printf 'SUMMARY: %s\\n' "
            "'Commande Python directe non compatible avec cProfile'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote(action_message)}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    prefix, python_tokens = resolved
    output_path = probe_dir / "prof.pstats"
    report_path = probe_dir / "cprofile_top.txt"
    command = [
        *prefix,
        python_tokens[0],
        "-m",
        "cProfile",
        "-o",
        str(output_path),
        *python_tokens[1:],
    ]
    report_command = [
        *prefix,
        python_tokens[0],
        "-c",
        (
            "import pstats; "
            f"stats = pstats.Stats({str(output_path)!r}); "
            "stats.sort_stats('cumtime').print_stats(40)"
        ),
    ]
    return _status_wrapped(
        (
            f"{_shell_join(command)} && "
            f"{_shell_join(report_command)} > {_quote_path(report_path)}"
        ),
        "Profil cProfile terminé",
        "Inspectez cprofile_top.txt pour les hotspots CPU cumulatifs.",
        action_commands=(
            _inspect_path_command(report_path, line_count=80),
            _inspect_path_command(probe_dir / "stderr.log"),
        ),
    )


def _build_f2_command(config: SanitizerConfig, probe_dir: Path) -> str:
    retry_command = (
        'make sanitizer SANITIZER_ARGS="--probe F2 --python-command '
        "'python mybci.py --feature-strategy wavelet'\""
    )
    if config.direct_python_command is None:
        return (
            f"printf 'SUMMARY: %s\\n' 'Commande Python directe absente'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote('Passez --python-command pour activer F2/Scalene.')}; "
            f"printf 'ACTION_CMD: %s\\n' "
            f"{shlex.quote(retry_command)}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    python_tokens = _resolve_python_script_invocation(config.direct_python_command)
    if python_tokens is None:
        return (
            "printf 'SUMMARY: %s\\n' "
            "'Commande Python directe non compatible avec Scalene'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote('Passez une commande du type `python script.py ...`.')}; "
            f"printf 'ACTION_CMD: %s\\n' "
            f"{shlex.quote(retry_command)}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    out_path = probe_dir / "scalene.txt"
    script_path = shlex.quote(python_tokens[1])
    script_args = ""
    if python_tokens[2:]:
        script_args = " --- " + _shell_join(python_tokens[2:])
    return _status_wrapped(
        (
            f"{shlex.quote(sys.executable)} -m scalene "
            "--cpu --memory --profile-all --reduced-profile --cli "
            f"--outfile {_quote_path(out_path)} "
            f"{script_path}{script_args}"
        ),
        "Profil Scalene terminé",
        "Inspectez scalene.txt pour les hotspots CPU et allocations mémoire.",
        action_commands=(
            _inspect_path_command(out_path, line_count=120),
            _inspect_path_command(probe_dir / "stderr.log"),
        ),
    )


def _build_f3_command(config: SanitizerConfig, probe_dir: Path) -> str:
    retry_command = (
        'make sanitizer SANITIZER_ARGS="--probe F3 --python-command '
        "'python mybci.py --feature-strategy wavelet'\""
    )
    retry_with_longer_duration = (
        "make sanitizer "
        "SANITIZER_ARGS="
        f"'--probe F3 --pyspy-duration-seconds "
        f"{max(config.pyspy_duration_seconds + 2, 5)}'"
    )
    if config.direct_python_command is None:
        return (
            f"printf 'SUMMARY: %s\\n' 'Commande Python directe absente'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote('Passez --python-command pour activer F3/py-spy.')}; "
            f"printf 'ACTION_CMD: %s\\n' "
            f"{shlex.quote(retry_command)}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    python_tokens = _resolve_python_script_invocation(config.direct_python_command)
    if python_tokens is None:
        return (
            "printf 'SUMMARY: %s\\n' "
            "'Commande Python directe non compatible avec py-spy'; "
            f"printf 'ACTION: %s\\n' "
            f"{shlex.quote('Passez une commande du type `python script.py ...`.')}; "
            f"printf 'ACTION_CMD: %s\\n' "
            f"{shlex.quote(retry_command)}; "
            f"exit {SKIP_EXIT_CODE}"
        )
    svg_path = probe_dir / "pyspy.svg"
    return _status_wrapped(
        (
            "py-spy record "
            f"--duration {config.pyspy_duration_seconds} "
            f"--output {_quote_path(svg_path)} "
            f"-- {_shell_join(python_tokens)}"
        ),
        "Profil py-spy terminé",
        ("Si py-spy échoue avec ptrace, basculez vers F2/scalene ou F1/cProfile."),
        action_commands=(
            _inspect_path_command(probe_dir / "stderr.log"),
            f"test -f {_quote_path(svg_path)} && ls -lh {_quote_path(svg_path)}",
            "cat /proc/sys/kernel/yama/ptrace_scope",
            retry_with_longer_duration,
            "make sanitizer SANITIZER_ARGS='--probe F2'",
            "make sanitizer SANITIZER_ARGS='--probe F1'",
        ),
    )


def _build_f4_command(config: SanitizerConfig) -> str:
    return _status_wrapped(
        f"PYTHONPROFILEIMPORTTIME=1 {_bash_shell(_preferred_profile_shell(config))}",
        "Profil import terminé",
        "Cherchez les imports les plus lents dans stderr.log.",
    )


def _build_g4_command(config: SanitizerConfig, probe_dir: Path) -> str:
    listing_path = probe_dir / "core_listing.txt"
    return _status_wrapped(
        (
            "ulimit -c unlimited; "
            f"{_bash_shell(_target_shell(config))}; "
            "status=$?; "
            f"ls -lh core* 2>/dev/null | tee {_quote_path(listing_path)} || true; "
            'exit "$status"'
        ),
        "Vérification core dump terminée",
        (
            "Si un core apparaît, archivez-le et reproduisez avec G1/G2 ou un "
            "débogueur local."
        ),
    )


def _build_h1_command(config: SanitizerConfig) -> str:
    return _status_wrapped(
        (
            'env -i HOME="$HOME" PATH="$PATH" LANG=C LC_ALL=C '
            "PYTHONHASHSEED=0 "
            f"{_bash_shell(_target_shell(config))}"
        ),
        "Exécution environnement propre terminée",
        "Comparez la sortie avec le run nominal pour isoler une variable parasite.",
    )


def _build_p0_command(probe_dir: Path) -> str:
    perf_path = probe_dir / "perf_probe.txt"
    return (
        f"{{ command -v perf || echo 'perf absent'; "
        "cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true; "
        f"}} | tee {_quote_path(perf_path)}; "
        "if command -v perf >/dev/null 2>&1; then "
        "printf 'SUMMARY: %s\\n' 'perf détecté'; "
        "else "
        "printf 'SUMMARY: %s\\n' 'perf absent'; "
        "printf 'ACTION: %s\\n' "
        "'Installez perf si disponible ou utilisez pyperf / scalene / strace.'; "
        f"exit {SKIP_EXIT_CODE}; "
        "fi"
    )


def _build_p1_command(target_shell: str, config: SanitizerConfig) -> str:
    return _status_wrapped(
        f"{_perf_command_prefix(config)} stat -d -- {_bash_shell(target_shell)}",
        "Capture perf stat terminée",
        "Inspectez task-clock, context-switches, IPC et cache-misses.",
        action_commands=(
            "cat /proc/sys/kernel/perf_event_paranoid",
            "sudo -v",
            "make sanitizer SANITIZER_ARGS='--probe A3.make'",
            "make sanitizer SANITIZER_ARGS='--probe F1 --probe F2 --probe D2'",
        ),
    )


def _build_p2_command(config: SanitizerConfig, probe_dir: Path) -> str:
    csv_path = probe_dir / "perf.csv"
    return _status_wrapped(
        (
            f"{_perf_command_prefix(config)} stat -x, "
            f"-o {_quote_path(csv_path)} "
            f"-d -- {_bash_shell(_target_shell(config))}"
        ),
        "Capture perf CSV terminée",
        "Graphez perf.csv pour comparer les runs.",
        action_commands=(
            "cat /proc/sys/kernel/perf_event_paranoid",
            "sudo -v",
            (
                f"test -f {_quote_path(csv_path)} && "
                f"sed -n '1,40p' {_quote_path(csv_path)}"
            ),
            "make sanitizer SANITIZER_ARGS='--probe A3.make'",
        ),
        post_actions=(
            (
                (
                    f"if [[ -f {_quote_path(csv_path)} ]]; then "
                    f"{_sudo_chown_command((csv_path,))} >/dev/null 2>&1 || true; "
                    "fi"
                ),
            )
            if _should_use_privileged_perf(config)
            else ()
        ),
    )


def _build_p3_command(config: SanitizerConfig, probe_dir: Path) -> str:
    data_path = probe_dir / "perf.data"
    report_path = probe_dir / "perf_report.txt"
    record_command = (
        f"{_perf_command_prefix(config)} record "
        f"-o {_quote_path(data_path)} "
        f"-g -- {_bash_shell(_target_shell(config))}"
    )
    if _should_use_privileged_perf(config):
        record_command += f" && {_sudo_chown_command((data_path,))}"
    return _status_wrapped(
        (
            f"{record_command} "
            f"&& perf report --input {_quote_path(data_path)} --stdio | "
            f"head -n 60 > {_quote_path(report_path)}"
        ),
        "Capture perf record terminée",
        "Inspectez perf_report.txt pour les hotspots symboliques.",
        action_commands=(
            "cat /proc/sys/kernel/perf_event_paranoid",
            "sudo -v",
            (
                f"test -f {_quote_path(report_path)} && "
                f"sed -n '1,80p' {_quote_path(report_path)}"
            ),
            "make sanitizer SANITIZER_ARGS='--probe F2 --probe D2'",
        ),
    )


def _timeout_for_pyperf(config: SanitizerConfig) -> int:
    total_runs = config.pyperf_warmups + config.pyperf_runs
    return max(config.timeout_seconds, total_runs * config.timeout_seconds)


def _timeout_for_hyperfine(config: SanitizerConfig) -> int:
    total_runs = config.hyperfine_warmups + config.hyperfine_runs
    return max(config.timeout_seconds, total_runs * config.timeout_seconds)


def _build_probe_catalog() -> tuple[ProbeDefinition, ...]:
    time_action = (
        "Vérifiez que `/usr/bin/time` est disponible sur Ubuntu puis relancez."
    )
    pyperf_action = (
        "Installez les dépendances dev Poetry puis vérifiez `pyperf` "
        "dans le venv du projet."
    )
    hyperfine_action = (
        "Hyperfine n'est pas un package Poetry: installez-le hors venv "
        "ou utilisez A3/pyperf."
    )
    poetry_action = "Installez Poetry ou retirez la variante `poetry run`."
    memory_profiler_action = (
        "Installez les dépendances dev Poetry pour exposer `mprof`."
    )
    psrecord_action = (
        "Installez les dépendances dev Poetry pour exposer `psrecord`, "
        "ou utilisez C2/mprof."
    )
    strace_action = "Installez `strace` puis relancez les sondes D2/E1."
    taskset_action = "Installez `taskset` (util-linux) ou utilisez B2/B3."
    ss_action = "Installez `ss` (iproute2) puis relancez D3."
    scalene_action = (
        "Installez les dépendances dev Poetry pour exposer `scalene`, ou utilisez F1."
    )
    pyspy_action = (
        "Installez les dépendances dev Poetry pour exposer `py-spy`, "
        "puis vérifiez les restrictions ptrace."
    )
    perf_missing_action = (
        "Installez `perf` si disponible, sinon utilisez les sondes utilisateur."
    )
    perf_blocked_action = (
        "Le noyau bloque `perf` pour un utilisateur non privilégié "
        "sur cette machine; utilisez les fallbacks utilisateur ou le mode "
        "privilégié explicite."
    )
    direct_python_action = (
        "Fournissez `--python-command 'python mybci.py "
        "--feature-strategy wavelet'` ou une commande Python équivalente."
    )
    time_commands = (
        "command -v /usr/bin/time",
        "sudo apt-get update && sudo apt-get install -y time",
    )
    pyperf_commands = (
        "poetry install --with dev",
        "poetry run pyperf --help",
    )
    hyperfine_commands = (
        "sudo apt-get update && sudo apt-get install -y hyperfine",
        "cargo install hyperfine --locked",
        (
            "make sanitizer SANITIZER_ARGS='--probe A3.make "
            "--pyperf-warmups 3 --pyperf-runs 20'"
        ),
    )
    poetry_commands = ("poetry --version",)
    memory_profiler_commands = (
        "poetry install --with dev",
        "poetry run mprof run --help",
    )
    psrecord_commands = (
        "poetry install --with dev",
        "poetry run psrecord --help",
    )
    taskset_commands = (
        "command -v taskset",
        "sudo apt-get update && sudo apt-get install -y util-linux",
        "make sanitizer SANITIZER_ARGS='--probe B2 --probe B3'",
    )
    ss_commands = (
        "command -v ss",
        "sudo apt-get update && sudo apt-get install -y iproute2",
    )
    strace_commands = (
        "command -v strace",
        "sudo apt-get update && sudo apt-get install -y strace",
    )
    scalene_commands = (
        "poetry install --with dev",
        "poetry run scalene --help",
    )
    pyspy_commands = (
        "poetry install --with dev",
        "poetry run py-spy record --help",
        "cat /proc/sys/kernel/yama/ptrace_scope",
    )
    perf_missing_commands = (
        "command -v perf",
        (
            "sudo apt-get update && sudo apt-get install -y "
            "linux-tools-common linux-tools-generic linux-tools-$(uname -r)"
        ),
        (
            "make sanitizer SANITIZER_ARGS='--probe A3.make "
            "--probe F1 --probe F2 --probe D2'"
        ),
    )
    perf_blocked_commands = (
        "cat /proc/sys/kernel/perf_event_paranoid",
        "sudo -v",
        (
            "make sanitizer SANITIZER_ALLOW_PRIVILEGED_TOOLS=1 "
            "SANITIZER_ARGS='--probe P1.make --probe P1.poetry --probe P2 --probe P3'"
        ),
        (
            "make sanitizer SANITIZER_ARGS='--probe A3.make "
            "--probe F1 --probe F2 --probe D2'"
        ),
    )
    direct_python_commands = (
        (
            'make sanitizer SANITIZER_ARGS="--probe F1 '
            "--python-command 'python mybci.py --feature-strategy wavelet'\""
        ),
    )

    return (
        ProbeDefinition(
            identifier="A1.make",
            title="GNU time sur target make",
            category="A",
            highlights=[
                "Elapsed wall time",
                "User/System CPU time",
                "RSS max et major faults",
            ],
            command_builder=lambda config, _dir: _build_time_verbose_command(
                _target_shell(config)
            ),
            preflight=_require_path(Path("/usr/bin/time"), time_action, time_commands),
        ),
        ProbeDefinition(
            identifier="A1.poetry",
            title="GNU time sur variante poetry run",
            category="A",
            highlights=[
                "Comparer la latence avec la variante make pure",
                "Observer le surcoût éventuel du wrapper Poetry",
            ],
            command_builder=lambda config, _dir: _build_time_verbose_command(
                _poetry_target_shell(config)
            ),
            preflight=lambda config: _merge_preflights(
                _require_path(Path("/usr/bin/time"), time_action, time_commands)(
                    config
                ),
                _require_poetry(poetry_action, poetry_commands)(config),
            ),
        ),
        ProbeDefinition(
            identifier="A2",
            title="time CSV parsable",
            category="A",
            highlights=[
                "CSV/JSONL ready-to-plot",
                "Median, stdev, p90 si plusieurs runs",
            ],
            preflight=_require_path(Path("/usr/bin/time"), time_action, time_commands),
            runner=_run_a2_probe,
        ),
        ProbeDefinition(
            identifier="A3.make",
            title="pyperf sur target make",
            category="A",
            highlights=[
                "Median, stdev et outliers",
                "Variance inter-runs",
            ],
            command_builder=lambda config, _dir: _build_pyperf_command(
                _target_shell(config), config
            ),
            preflight=_require_python_module(
                "pyperf",
                pyperf_action,
                pyperf_commands,
            ),
            timeout_resolver=_timeout_for_pyperf,
        ),
        ProbeDefinition(
            identifier="A3.poetry",
            title="pyperf sur variante poetry run",
            category="A",
            highlights=[
                "Comparer la variance make vs poetry run",
            ],
            command_builder=lambda config, _dir: _build_pyperf_command(
                _poetry_target_shell(config), config
            ),
            preflight=lambda config: _merge_preflights(
                _require_python_module("pyperf", pyperf_action, pyperf_commands)(
                    config
                ),
                _require_poetry(poetry_action, poetry_commands)(config),
            ),
            timeout_resolver=_timeout_for_pyperf,
        ),
        ProbeDefinition(
            identifier="A4",
            title="hyperfine",
            category="A",
            highlights=[
                "Stats A/B ergonomiques",
                "Médiane et variance condensées",
            ],
            command_builder=lambda config, _dir: _build_hyperfine_command(
                _target_shell(config), config
            ),
            preflight=_require_command(
                "hyperfine",
                hyperfine_action,
                hyperfine_commands,
            ),
            timeout_resolver=_timeout_for_hyperfine,
        ),
        ProbeDefinition(
            identifier="B1.make",
            title="taskset sur target make",
            category="B",
            highlights=[
                "Réduit la migration CPU",
                "Stabilise souvent le wall time",
            ],
            command_builder=lambda config, _dir: _build_taskset_command(
                _target_shell(config), config
            ),
            preflight=_require_command("taskset", taskset_action, taskset_commands),
        ),
        ProbeDefinition(
            identifier="B1.poetry",
            title="taskset sur variante poetry run",
            category="B",
            highlights=[
                "Comparer le jitter avec la variante poetry run",
            ],
            command_builder=lambda config, _dir: _build_taskset_command(
                _poetry_target_shell(config), config
            ),
            preflight=lambda config: _merge_preflights(
                _require_command("taskset", taskset_action, taskset_commands)(config),
                _require_poetry(poetry_action, poetry_commands)(config),
            ),
        ),
        ProbeDefinition(
            identifier="B2",
            title="nice -n 19",
            category="B",
            highlights=[
                "Réduit l'impact sur le poste",
            ],
            command_builder=lambda config, _dir: _status_wrapped(
                f"nice -n 19 {_bash_shell(_target_shell(config))}",
                "Exécution nice terminée",
                "Comparez la latence et le jitter avec le run nominal.",
            ),
            preflight=_require_command("nice", "Le binaire `nice` est requis."),
        ),
        ProbeDefinition(
            identifier="B3",
            title="OMP / BLAS single-thread",
            category="B",
            highlights=[
                "Réduit les explosions de threads BLAS/OpenMP",
                "Souvent meilleur pour la reproductibilité",
            ],
            command_builder=lambda config, _dir: _status_wrapped(
                (
                    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 "
                    f"{_bash_shell(_target_shell(config))}"
                ),
                "Run single-thread BLAS/OpenMP terminé",
                "Comparez CPU% et variance avec le run nominal.",
            ),
        ),
        ProbeDefinition(
            identifier="B4",
            title="sampling ps",
            category="B",
            highlights=[
                "Top RSS/CPU en live",
            ],
            command_builder=_build_b4_command,
        ),
        ProbeDefinition(
            identifier="C1",
            title="ulimit mémoire",
            category="C",
            highlights=[
                "Teste le chemin d'erreur sous pression mémoire",
            ],
            command_builder=lambda config, _dir: _build_c1_command(config),
            allow_nonzero_exit=True,
        ),
        ProbeDefinition(
            identifier="C2",
            title="mprof",
            category="C",
            highlights=[
                "RSS time-series",
                "Inclut les enfants via --include-children",
            ],
            command_builder=_build_c2_command,
            preflight=_require_command(
                "mprof",
                memory_profiler_action,
                memory_profiler_commands,
            ),
        ),
        ProbeDefinition(
            identifier="C3",
            title="psrecord",
            category="C",
            highlights=[
                "CPU% et RSS dans le temps",
                "À comparer avec C2 si le PID make masque le vrai worker",
            ],
            command_builder=_build_c3_command,
            preflight=_require_command(
                "psrecord",
                psrecord_action,
                psrecord_commands,
            ),
        ),
        ProbeDefinition(
            identifier="D1",
            title="pidstat ou fallback /proc/<PID>/io",
            category="D",
            highlights=[
                "Débits disque / iowait si pidstat existe",
                "Fallback /proc/<PID>/io sinon",
            ],
            command_builder=_build_d1_command,
        ),
        ProbeDefinition(
            identifier="D2",
            title="strace -T fichiers",
            category="D",
            highlights=[
                "open/stat lents",
                "Latences de syscalls fichier",
            ],
            command_builder=_build_d2_command,
            preflight=_require_command("strace", strace_action, strace_commands),
        ),
        ProbeDefinition(
            identifier="D3",
            title="ss -tpn",
            category="D",
            highlights=[
                "Détecte des sockets inattendus",
            ],
            command_builder=lambda _config, _dir: _status_wrapped(
                "ss -tpn | head -n 20",
                "Snapshot réseau terminé",
                "Inspectez stdout.log pour repérer des connexions inattendues.",
            ),
            preflight=_require_command("ss", ss_action, ss_commands),
        ),
        ProbeDefinition(
            identifier="E1",
            title="strace permissions / ENOENT",
            category="E",
            highlights=[
                "EACCES / ENOENT",
                "Logique de recherche des chemins",
            ],
            command_builder=_build_e1_command,
            preflight=_require_command("strace", strace_action, strace_commands),
        ),
        ProbeDefinition(
            identifier="E2",
            title="nombre de FD ouverts",
            category="E",
            highlights=[
                "Détecte une fuite de descripteurs",
            ],
            command_builder=_build_e2_command,
        ),
        ProbeDefinition(
            identifier="E3",
            title="audit permissions src/tpv",
            category="E",
            highlights=[
                "Snapshot chmod / owner",
            ],
            command_builder=lambda _config, probe_dir: _build_e3_command(probe_dir),
        ),
        ProbeDefinition(
            identifier="F1",
            title="cProfile",
            category="F",
            highlights=[
                "cumtime par fonction",
                "Top 40 des hotspots CPU",
            ],
            command_builder=lambda config, probe_dir: _build_cprofile_command(
                config, probe_dir
            ),
            preflight=_require_direct_python(
                direct_python_action,
                direct_python_commands,
            ),
        ),
        ProbeDefinition(
            identifier="F2",
            title="Scalene",
            category="F",
            highlights=[
                "Hotspots CPU et mémoire",
            ],
            command_builder=_build_f2_command,
            preflight=lambda config: _merge_preflights(
                _require_command("scalene", scalene_action, scalene_commands)(config),
                _require_direct_python_script(
                    direct_python_action,
                    (
                        (
                            'make sanitizer SANITIZER_ARGS="--probe F2 '
                            "--python-command 'python mybci.py "
                            "--feature-strategy wavelet'\""
                        ),
                    ),
                )(config),
            ),
        ),
        ProbeDefinition(
            identifier="F3",
            title="py-spy",
            category="F",
            highlights=[
                "Flamegraph sampling",
                "Peut échouer si ptrace est restreint",
            ],
            command_builder=_build_f3_command,
            preflight=lambda config: _merge_preflights(
                _require_command("py-spy", pyspy_action, pyspy_commands)(config),
                _require_direct_python_script(
                    direct_python_action,
                    (
                        (
                            'make sanitizer SANITIZER_ARGS="--probe F3 '
                            "--python-command 'python mybci.py "
                            "--feature-strategy wavelet'\""
                        ),
                    ),
                )(config),
            ),
        ),
        ProbeDefinition(
            identifier="F4",
            title="PYTHONPROFILEIMPORTTIME",
            category="F",
            highlights=[
                "Imports lents au démarrage",
            ],
            command_builder=lambda config, _dir: _build_f4_command(config),
        ),
        ProbeDefinition(
            identifier="G1",
            title="PYTHONFAULTHANDLER",
            category="G",
            highlights=[
                "Tracebacks robustes en cas de crash",
            ],
            command_builder=lambda config, _dir: _status_wrapped(
                f"PYTHONFAULTHANDLER=1 {_bash_shell(_target_shell(config))}",
                "Run avec faulthandler terminé",
                "Cherchez les tracebacks natifs dans stderr.log.",
            ),
        ),
        ProbeDefinition(
            identifier="G2",
            title="PYTHONDEVMODE",
            category="G",
            highlights=[
                "Warnings de ressources et checks runtime",
            ],
            command_builder=lambda config, _dir: _status_wrapped(
                f"PYTHONDEVMODE=1 {_bash_shell(_target_shell(config))}",
                "Run Python dev mode terminé",
                "Inspectez stderr.log pour les warnings et ResourceWarning.",
            ),
        ),
        ProbeDefinition(
            identifier="G3",
            title="timeout",
            category="G",
            highlights=[
                "Détecte les hangs",
                "124 = timeout",
            ],
            command_builder=lambda config, _dir: _status_wrapped(
                (
                    f"timeout {config.timeout_seconds}s "
                    f"{_bash_shell(_target_shell(config))}; "
                    "rc=$?; "
                    "printf 'timeout_exit=%s\\n' \"$rc\""
                ),
                "Run timeout terminé",
                "Si exit=124, la commande bloque et doit être instrumentée.",
            ),
            allow_nonzero_exit=True,
        ),
        ProbeDefinition(
            identifier="G4",
            title="core dump",
            category="G",
            highlights=[
                "Détecte un core si le système l'autorise",
            ],
            command_builder=_build_g4_command,
            allow_nonzero_exit=True,
        ),
        ProbeDefinition(
            identifier="H1",
            title="env propre",
            category="H",
            highlights=[
                "Réduit les variables parasites",
            ],
            command_builder=lambda config, _dir: _build_h1_command(config),
        ),
        ProbeDefinition(
            identifier="H2",
            title="dump meta",
            category="H",
            highlights=[
                "Commit git, Python, Poetry, target command",
            ],
            runner=_run_h2_probe,
        ),
        ProbeDefinition(
            identifier="P0",
            title="détection perf",
            category="P",
            highlights=[
                "Présence de perf",
                "perf_event_paranoid",
            ],
            command_builder=lambda _config, probe_dir: _build_p0_command(probe_dir),
        ),
        ProbeDefinition(
            identifier="P1.make",
            title="perf stat sur target make",
            category="P",
            highlights=[
                "task-clock, context-switches, IPC, cache-misses",
            ],
            command_builder=lambda config, _dir: _build_p1_command(
                _target_shell(config), config
            ),
            preflight=_require_perf(
                perf_missing_action,
                perf_missing_commands,
                perf_blocked_action,
                perf_blocked_commands,
            ),
        ),
        ProbeDefinition(
            identifier="P1.poetry",
            title="perf stat sur variante poetry run",
            category="P",
            highlights=[
                "Comparer make vs poetry run côté compteurs CPU",
            ],
            command_builder=lambda config, _dir: _build_p1_command(
                _poetry_target_shell(config), config
            ),
            preflight=lambda config: _merge_preflights(
                _require_perf(
                    perf_missing_action,
                    perf_missing_commands,
                    perf_blocked_action,
                    perf_blocked_commands,
                )(config),
                _require_poetry(poetry_action, poetry_commands)(config),
            ),
        ),
        ProbeDefinition(
            identifier="P2",
            title="perf stat CSV",
            category="P",
            highlights=[
                "CSV prêt à grapher",
            ],
            command_builder=_build_p2_command,
            preflight=_require_perf(
                perf_missing_action,
                perf_missing_commands,
                perf_blocked_action,
                perf_blocked_commands,
            ),
        ),
        ProbeDefinition(
            identifier="P3",
            title="perf record",
            category="P",
            highlights=[
                "perf.data et top symboles",
            ],
            command_builder=_build_p3_command,
            preflight=_require_perf(
                perf_missing_action,
                perf_missing_commands,
                perf_blocked_action,
                perf_blocked_commands,
            ),
        ),
    )


def _merge_preflights(*results: PreflightResult) -> PreflightResult:
    for result in results:
        if not result.ready:
            return result
    return PreflightResult(ready=True)


def _normalize_process_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


def _build_shell_probe_skip(
    config: SanitizerConfig,
    probe: ProbeDefinition,
    probe_dir: Path,
) -> ProbeResult | None:
    if probe.preflight is not None:
        preflight = probe.preflight(config)
        if not preflight.ready:
            return _build_skip_result(
                config=config,
                probe=probe,
                summary=preflight.summary or "Pré-requis manquant.",
                action=preflight.action,
                action_commands=preflight.action_commands,
            )
    if probe.command_builder is None:
        return _build_skip_result(
            config=config,
            probe=probe,
            summary="Aucune commande définie pour cette sonde.",
            action="Implémentez le builder de commande avant de relancer.",
            action_commands=(),
        )
    return None


def _probe_status_from_exit_code(exit_code: int, allow_nonzero_exit: bool) -> str:
    if exit_code == 0 or allow_nonzero_exit:
        return "ok"
    if exit_code == SKIP_EXIT_CODE:
        return "skipped"
    return "warn"


def _probe_summary_from_payload(
    payload: str,
    status: str,
    exit_code: int | None,
) -> str:
    summary = _extract_tagged_line(payload, "SUMMARY:")
    if summary is not None:
        return summary
    if status == "ok":
        return "Sonde terminée."
    if status == "skipped":
        return "Sonde ignorée."
    return f"Sonde terminée avec exit={exit_code}."


def _build_shell_probe_execution(
    completed: subprocess.CompletedProcess[str],
    probe: ProbeDefinition,
) -> ShellProbeExecution:
    status = _probe_status_from_exit_code(
        completed.returncode, probe.allow_nonzero_exit
    )
    payload = completed.stdout + "\n" + completed.stderr
    return ShellProbeExecution(
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        status=status,
        summary=_probe_summary_from_payload(payload, status, completed.returncode),
        action=_extract_tagged_line(payload, "ACTION:"),
        action_commands=tuple(_extract_tagged_lines(payload, "ACTION_CMD:")),
    )


def _build_timeout_execution(exc: subprocess.TimeoutExpired) -> ShellProbeExecution:
    return ShellProbeExecution(
        stdout=_normalize_process_output(exc.stdout),
        stderr=_normalize_process_output(exc.stderr),
        exit_code=None,
        status="warn",
        summary=f"Timeout après {exc.timeout} secondes.",
        action=(
            "Réduisez le nombre de runs, ciblez moins de sondes ou augmentez "
            "`--timeout-seconds`."
        ),
        action_commands=(),
    )


def _write_probe_logs(probe_dir: Path, stdout: str, stderr: str) -> tuple[Path, Path]:
    stdout_path = probe_dir / "stdout.log"
    stderr_path = probe_dir / "stderr.log"
    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)
    return stdout_path, stderr_path


def _run_shell_probe(config: SanitizerConfig, probe: ProbeDefinition) -> ProbeResult:
    probe_dir = config.output_dir / probe.identifier
    probe_dir.mkdir(parents=True, exist_ok=True)
    skipped_result = _build_shell_probe_skip(config, probe, probe_dir)
    if skipped_result is not None:
        return skipped_result

    command_builder = probe.command_builder
    if command_builder is None:
        return _build_skip_result(
            config,
            probe,
            "warn",
            "Sonde shell mal configurée: commande absente.",
        )
    command = command_builder(config, probe_dir)
    _write_text(probe_dir / "command.sh", command + "\n")

    try:
        execution = _build_shell_probe_execution(
            _run_subprocess(
                [BASH_EXECUTABLE, "-lc", command],
                env={**os.environ, "MPLBACKEND": "Agg"},
                timeout=_probe_timeout(config, probe),
            ),
            probe,
        )
    except subprocess.TimeoutExpired as exc:
        execution = _build_timeout_execution(exc)

    stdout_path, stderr_path = _write_probe_logs(
        probe_dir,
        execution.stdout,
        execution.stderr,
    )

    result = ProbeResult(
        identifier=probe.identifier,
        title=probe.title,
        category=probe.category,
        status=execution.status,
        summary=execution.summary,
        action=execution.action,
        action_commands=list(execution.action_commands),
        exit_code=execution.exit_code,
        command=command,
        output_dir=str(probe_dir),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        highlights=list(probe.highlights),
    )
    if probe.identifier == "F3":
        result = _normalize_pyspy_result(
            result=result,
            probe_dir=probe_dir,
            stdout=execution.stdout,
        )
    return _finalize_result(probe_dir, result)


def _parse_time_metrics(payload: str) -> dict[str, float | int | str] | None:
    metrics: dict[str, float | int | str] = {}
    for item in payload.strip().split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "cpu":
            metrics[key] = value
            continue
        if key == "exit":
            metrics[key] = int(value)
            continue
        if key == "rss_kb":
            metrics[key] = int(value)
            continue
        metrics[key] = float(value)
    required = {"wall", "user", "sys", "cpu", "rss_kb", "exit"}
    if required.issubset(metrics):
        return metrics
    return None


def _summarize_numeric_series(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    index = int(0.9 * (len(ordered) - 1))
    return {
        "n": len(values),
        "median": median(values),
        "mean": mean(values),
        "stdev": pstdev(values),
        "min": min(values),
        "max": max(values),
        "p90": ordered[index],
    }


def _run_a2_iteration(
    config: SanitizerConfig,
    command: str,
    run_index: int,
    metrics_path: Path,
) -> TimedRunResult:
    try:
        completed = _run_subprocess(
            [
                "/usr/bin/time",
                "-f",
                "wall=%e,user=%U,sys=%S,cpu=%P,rss_kb=%M,exit=%x",
                "-o",
                str(metrics_path),
                BASH_EXECUTABLE,
                "-lc",
                command,
            ],
            env={**os.environ, "MPLBACKEND": "Agg"},
            timeout=config.timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        return TimedRunResult(
            stdout=_normalize_process_output(exc.stdout),
            stderr=_normalize_process_output(exc.stderr),
            row=None,
            action=(
                "Une itération A2 a dépassé le timeout. Réduisez la commande "
                "cible ou augmentez `--timeout-seconds`."
            ),
        )

    metrics_payload = metrics_path.read_text(encoding="utf-8").strip()
    metrics = _parse_time_metrics(metrics_payload)
    if metrics is None:
        return TimedRunResult(
            stdout=stdout,
            stderr=stderr,
            row=None,
            action="Le format time CSV est invalide. Vérifiez metrics.txt.",
        )

    return TimedRunResult(
        stdout=stdout,
        stderr=stderr,
        row=TimeCsvRow(
            run=run_index,
            wall_s=float(metrics["wall"]),
            user_s=float(metrics["user"]),
            sys_s=float(metrics["sys"]),
            cpu=str(metrics["cpu"]),
            rss_kb=int(metrics["rss_kb"]),
            exit=int(metrics["exit"]),
        ),
    )


def _write_time_rows(
    rows: Sequence[TimeCsvRow], csv_path: Path, jsonl_path: Path
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "wall_s", "user_s", "sys_s", "cpu", "rss_kb", "exit"),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")


def _build_time_summary(
    rows: Sequence[TimeCsvRow], run_count: int
) -> tuple[str, dict[str, object]]:
    summary_payload: dict[str, object] = {"runs": len(rows), "exit_codes": []}
    if not rows:
        return "Aucun run A2 exploitable.", summary_payload

    walls = [row.wall_s for row in rows]
    rss_values = [float(row.rss_kb) for row in rows]
    exit_codes = sorted({row.exit for row in rows})
    summary_payload["exit_codes"] = exit_codes
    summary_payload["wall_s"] = _summarize_numeric_series(walls)
    summary_payload["rss_kb"] = _summarize_numeric_series(rss_values)

    summary = (
        f"{len(rows)} run(s), median wall={median(walls):.3f}s, "
        f"max RSS={max(rss_values):.0f} KiB, exits={exit_codes}"
    )
    if run_count == 1:
        summary += (
            ". Utilisez `--time-csv-runs 20` pour une lecture plus robuste "
            "de la variance."
        )
    return summary, summary_payload


def _run_a2_probe(  # noqa: PLR0915
    config: SanitizerConfig, probe: ProbeDefinition
) -> ProbeResult:
    probe_dir = config.output_dir / probe.identifier
    probe_dir.mkdir(parents=True, exist_ok=True)

    preflight = _require_path(
        Path("/usr/bin/time"),
        "Vérifiez que `/usr/bin/time` est installé puis relancez A2.",
        (
            "command -v /usr/bin/time",
            "sudo apt-get update && sudo apt-get install -y time",
        ),
    )(config)
    if not preflight.ready:
        return _build_skip_result(
            config=config,
            probe=probe,
            summary=preflight.summary or "Pré-requis manquant.",
            action=preflight.action,
        )

    command = _target_shell(config)
    _write_text(probe_dir / "command.sh", command + "\n")

    runs_dir = probe_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = probe_dir / "time.csv"
    jsonl_path = probe_dir / "time.jsonl"
    stats_path = probe_dir / "time_summary.json"

    rows: list[TimeCsvRow] = []
    aggregated_stdout: list[str] = []
    aggregated_stderr: list[str] = []
    status = "ok"
    action: str | None = None

    for run_index in range(1, config.time_csv_runs + 1):
        metrics_path = runs_dir / f"run_{run_index:02d}.metrics.txt"
        run_result = _run_a2_iteration(
            config=config,
            command=command,
            run_index=run_index,
            metrics_path=metrics_path,
        )
        stdout = run_result.stdout
        stderr = run_result.stderr

        _write_text(runs_dir / f"run_{run_index:02d}.out.log", stdout)
        _write_text(runs_dir / f"run_{run_index:02d}.err.log", stderr)
        aggregated_stdout.append(
            f"===== RUN {run_index:02d} =====\n{stdout.rstrip()}\n"
        )
        aggregated_stderr.append(
            f"===== RUN {run_index:02d} =====\n{stderr.rstrip()}\n"
        )

        if run_result.row is None:
            status = "warn"
            action = run_result.action
            continue

        rows.append(run_result.row)
        if run_result.row.exit != 0:
            status = "warn"
            action = (
                "La commande cible retourne un code non nul. Corrigez-la avant "
                "d'interpréter les métriques de performance."
            )

    stdout_path = probe_dir / "stdout.log"
    stderr_path = probe_dir / "stderr.log"
    _write_text(stdout_path, "".join(aggregated_stdout))
    _write_text(stderr_path, "".join(aggregated_stderr))

    _write_time_rows(rows, csv_path, jsonl_path)

    summary, summary_payload = _build_time_summary(rows, config.time_csv_runs)
    if not rows:
        status = "warn"
        if action is None:
            action = "Inspectez les logs par run puis relancez A2."

    _write_text(stats_path, json.dumps(summary_payload, indent=2))

    result = ProbeResult(
        identifier=probe.identifier,
        title=probe.title,
        category=probe.category,
        status=status,
        summary=summary,
        action=action,
        exit_code=0 if status == "ok" else 1,
        command=command,
        output_dir=str(probe_dir),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        highlights=list(probe.highlights),
    )
    return _finalize_result(probe_dir, result)


def _safe_capture(command: Sequence[str]) -> str | None:
    try:
        return _check_output_subprocess(command)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _run_h2_probe(config: SanitizerConfig, probe: ProbeDefinition) -> ProbeResult:
    probe_dir = config.output_dir / probe.identifier
    probe_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_commit": _safe_capture(["git", "rev-parse", "HEAD"]),
        "python": _safe_capture([sys.executable, "-V"]),
        "poetry": _safe_capture(["poetry", "--version"]),
        "target_command": list(config.command),
        "direct_python_command": (
            list(config.direct_python_command)
            if config.direct_python_command is not None
            else None
        ),
        "allow_privileged_tools": config.allow_privileged_tools,
        "pyspy_duration_seconds": config.pyspy_duration_seconds,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    json_path = probe_dir / "meta.json"
    text_path = probe_dir / "meta.txt"
    _write_text(json_path, json.dumps(metadata, indent=2))
    text_lines = [
        f"git_commit={metadata['git_commit']}",
        f"python={metadata['python']}",
        f"poetry={metadata['poetry']}",
        f"target_command={_shell_join(config.command)}",
        (
            "direct_python_command="
            + (
                _shell_join(config.direct_python_command)
                if config.direct_python_command is not None
                else "None"
            )
        ),
        f"allow_privileged_tools={config.allow_privileged_tools}",
        f"pyspy_duration_seconds={config.pyspy_duration_seconds}",
    ]
    _write_text(text_path, "\n".join(text_lines) + "\n")

    result = ProbeResult(
        identifier=probe.identifier,
        title=probe.title,
        category=probe.category,
        status="ok",
        summary="Métadonnées d'environnement capturées.",
        action=None,
        exit_code=0,
        command=None,
        output_dir=str(probe_dir),
        stdout_log=None,
        stderr_log=None,
        highlights=list(probe.highlights),
    )
    return _finalize_result(probe_dir, result)


def _run_probe(config: SanitizerConfig, probe: ProbeDefinition) -> ProbeResult:
    runner = probe.runner or _run_shell_probe
    print(f"[{probe.identifier}] START - {probe.title}", flush=True)
    result = runner(config, probe)
    print(f"[{probe.identifier}] {result.status.upper()} - {result.summary}")
    if result.action:
        print(f"  action: {result.action}")
    for action_command in result.action_commands:
        print(f"  command: {action_command}")
    return result


def _escape_markdown_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("|", "\\|").replace("\n", " ")


def _count_probe_statuses(results: Sequence[ProbeResult]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "skipped": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _build_summary_payload(
    config: SanitizerConfig,
    results: Sequence[ProbeResult],
    overall_status: str,
    counts: dict[str, int],
    timestamps: tuple[datetime, datetime],
) -> dict[str, object]:
    started_at, finished_at = timestamps
    return {
        "overall_status": overall_status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "target_command": list(config.command),
        "direct_python_command": (
            list(config.direct_python_command)
            if config.direct_python_command is not None
            else None
        ),
        "allow_privileged_tools": config.allow_privileged_tools,
        "pyspy_duration_seconds": config.pyspy_duration_seconds,
        "counts": counts,
        "results": [asdict(result) for result in results],
    }


def _build_summary_header_lines(
    config: SanitizerConfig,
    overall_status: str,
    counts: dict[str, int],
    started_at: datetime,
    finished_at: datetime,
) -> list[str]:
    direct_python_command = (
        _shell_join(config.direct_python_command)
        if config.direct_python_command is not None
        else "None"
    )
    return [
        "# Sanitizer Summary",
        "",
        f"- overall_status: `{overall_status}`",
        f"- target_command: `{_shell_join(config.command)}`",
        f"- direct_python_command: `{direct_python_command}`",
        f"- allow_privileged_tools: `{config.allow_privileged_tools}`",
        f"- pyspy_duration_seconds: `{config.pyspy_duration_seconds}`",
        f"- started_at: `{started_at.isoformat(timespec='seconds')}`",
        f"- finished_at: `{finished_at.isoformat(timespec='seconds')}`",
        f"- counts: `{counts}`",
        "",
        "| Probe | Status | Summary | Action |",
        "|---|---|---|---|",
    ]


def _build_summary_table_lines(results: Sequence[ProbeResult]) -> list[str]:
    return [
        "| {probe} | {status} | {summary} | {action} |".format(
            probe=_escape_markdown_cell(result.identifier),
            status=_escape_markdown_cell(result.status),
            summary=_escape_markdown_cell(result.summary),
            action=_escape_markdown_cell(result.action),
        )
        for result in results
    ]


def _build_summary_action_lines(results: Sequence[ProbeResult]) -> list[str]:
    actionable = [result for result in results if result.action]
    if not actionable:
        return []
    lines = ["", "## Next Actions", ""]
    for result in actionable:
        lines.append(f"- `{result.identifier}`: {result.action}")
        for action_command in result.action_commands:
            lines.append(f"  - `{_escape_markdown_cell(action_command)}`")
    return lines


def _write_session_summary(
    config: SanitizerConfig,
    results: list[ProbeResult],
    started_at: datetime,
    finished_at: datetime,
) -> tuple[str, Path, Path]:
    counts = _count_probe_statuses(results)
    overall_status = "ok" if counts.get("warn", 0) == 0 else "warn"
    summary_json = config.output_dir / "summary.json"
    summary_md = config.output_dir / "summary.md"
    payload = _build_summary_payload(
        config=config,
        results=results,
        overall_status=overall_status,
        counts=counts,
        timestamps=(started_at, finished_at),
    )
    _write_text(summary_json, json.dumps(payload, indent=2))
    lines = _build_summary_header_lines(
        config=config,
        overall_status=overall_status,
        counts=counts,
        started_at=started_at,
        finished_at=finished_at,
    )
    lines.extend(_build_summary_table_lines(results))
    lines.extend(_build_summary_action_lines(results))
    _write_text(summary_md, "\n".join(lines) + "\n")
    return overall_status, summary_json, summary_md


def _session_exit_code(results: Sequence[ProbeResult]) -> int:
    """Retourne 0 tant que les sondes n'ont émis que des états non fatals."""

    return (
        0 if all(result.status in NON_FATAL_PROBE_STATUSES for result in results) else 1
    )


def _list_probes(catalog: Sequence[ProbeDefinition]) -> None:
    for probe in catalog:
        print(f"{probe.identifier}: [{probe.category}] {probe.title}")


def _normalize_cli_command(raw_command: Sequence[str]) -> tuple[str, ...]:
    return tuple(raw_command[1:] if raw_command[:1] == ("--",) else raw_command)


def _resolve_direct_python_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    command: Sequence[str],
) -> tuple[str, ...] | None:
    if args.python_command is not None:
        return tuple(shlex.split(args.python_command))
    inferred_python_command = infer_python_command(command)
    if inferred_python_command is None:
        return None
    if not inferred_python_command:
        parser.error("Commande Python directe inférée invalide.")
    return tuple(inferred_python_command)


def _resolve_selected_probe_ids(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    catalog: Sequence[ProbeDefinition],
) -> tuple[tuple[str, ...], dict[str, ProbeDefinition]]:
    selected_probe_ids = (
        tuple(args.probes)
        if args.probes
        else tuple(probe.identifier for probe in catalog)
    )
    catalog_by_id = {probe.identifier: probe for probe in catalog}
    unknown = [
        probe_id for probe_id in selected_probe_ids if probe_id not in catalog_by_id
    ]
    if unknown:
        parser.error(f"Sonde(s) inconnue(s): {', '.join(unknown)}")
    return selected_probe_ids, catalog_by_id


def _build_config_from_args(
    args: argparse.Namespace,
    command: tuple[str, ...],
    direct_python_command: tuple[str, ...] | None,
    selected_probe_ids: tuple[str, ...],
) -> SanitizerConfig:
    return SanitizerConfig(
        command=command,
        direct_python_command=direct_python_command,
        output_dir=args.output_dir,
        selected_probes=selected_probe_ids,
        timeout_seconds=args.timeout_seconds,
        pyperf_warmups=args.pyperf_warmups,
        pyperf_runs=args.pyperf_runs,
        hyperfine_warmups=args.hyperfine_warmups,
        hyperfine_runs=args.hyperfine_runs,
        cpu_core=args.cpu_core,
        ps_interval_seconds=args.ps_interval_seconds,
        psrecord_interval_seconds=args.psrecord_interval_seconds,
        memory_limit_kib=args.memory_limit_kib,
        time_csv_runs=args.time_csv_runs,
        allow_privileged_tools=args.allow_privileged_tools,
        pyspy_duration_seconds=max(1, args.pyspy_duration_seconds),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée CLI du sanitizer."""

    parser = build_parser()
    args = parser.parse_args(argv)

    catalog = _build_probe_catalog()
    if args.list_probes:
        _list_probes(catalog)
        return 0

    raw_command = tuple(args.command) if args.command else DEFAULT_COMMAND
    command = _normalize_cli_command(raw_command)
    direct_python_command = _resolve_direct_python_command(parser, args, command)
    selected_probe_ids, catalog_by_id = _resolve_selected_probe_ids(
        parser,
        args,
        catalog,
    )
    config = _build_config_from_args(
        args=args,
        command=command,
        direct_python_command=direct_python_command,
        selected_probe_ids=selected_probe_ids,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.utcnow()
    results = [
        _run_probe(config, catalog_by_id[probe_id])
        for probe_id in config.selected_probes
    ]
    finished_at = datetime.utcnow()
    overall_status, summary_json, summary_md = _write_session_summary(
        config=config,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )

    print(f"summary_json: {summary_json}")
    print(f"summary_md: {summary_md}")
    return _session_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entrypoint for TPV BCI workflows."""

import argparse
import subprocess
import sys
from typing import Sequence


def _call_module(module_name: str, subject: str, run: str) -> int:
    """Invoke a TPV module with the provided identifiers via ``python -m``."""
    command: list[str] = [sys.executable, "-m", module_name, subject, run]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pilote un workflow d'entraînement ou de prédiction TPV",
        usage="python mybci.py <subject> <run> {train,predict}",
    )
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    parser.add_argument(
        "mode",
        choices=("train", "predict"),
        help="Choix du pipeline à lancer",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mode == "train":
        return _call_module("tpv.train", args.subject, args.run)

    return _call_module("tpv.predict", args.subject, args.run)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

"""Command-line interface for local engine inspection and planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime import AdaptiveExecutionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openagent-engine")
    parser.add_argument("--data-dir", default=str(Path.home() / ".kernelloom"))
    parser.add_argument("--accelerator-python", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("hardware", help="Inspect local hardware and available runtimes")
    hardware.add_argument("--refresh", action="store_true")

    status = subparsers.add_parser("status", help="Show persisted engine state")
    status.add_argument("--project", default="default")

    inspect = subparsers.add_parser("inspect", help="Inspect a local model container")
    inspect.add_argument("path")
    inspect.add_argument("--project", default="default")
    inspect.add_argument("--include-tensors", action="store_true")

    compile_command = subparsers.add_parser("compile", help="Build an adaptive execution plan")
    compile_command.add_argument("path")
    compile_command.add_argument("--project", default="default")
    compile_command.add_argument("--prompt-tokens", type=int, default=512)
    compile_command.add_argument("--context-tokens", type=int, default=4096)
    compile_command.add_argument("--memory-budget-gb", type=float)
    compile_command.add_argument("--quality-loss-limit", type=float, default=0.08)
    compile_command.add_argument("--power-mode", choices=("performance", "balanced", "efficiency"), default="balanced")
    compile_command.add_argument("--no-backend-compile", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = AdaptiveExecutionEngine(args.data_dir, accelerator_python=args.accelerator_python)
    try:
        if args.command == "hardware":
            result = engine.hardware(refresh=args.refresh)
        elif args.command == "status":
            result = engine.status(project=args.project)
        elif args.command == "inspect":
            result = engine.inspect_model(
                args.path,
                project=args.project,
                include_tensors=args.include_tensors,
            )
        else:
            result = engine.compile_model(
                args.path,
                project=args.project,
                prompt_tokens=args.prompt_tokens,
                context_tokens=args.context_tokens,
                memory_budget_gb=args.memory_budget_gb,
                quality_loss_limit=args.quality_loss_limit,
                power_mode=args.power_mode,
                backend_compile=not args.no_backend_compile,
            )
        _print_json(result)
        return 0
    finally:
        engine.close()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())

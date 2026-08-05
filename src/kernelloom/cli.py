"""KernelLoom command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .model import KernelLoomModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelloom")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Load a local model and generate one response")
    _model_arguments(run)
    run.add_argument("prompt")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.7)

    serve = subparsers.add_parser("serve", help="Start the API and browser console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11435)
    serve.add_argument("--model-path", default="")
    serve.add_argument("--model-id", default="default")
    serve.add_argument("--backend", choices=("auto", "llama-cpp", "openvino"), default="auto")
    serve.add_argument("--device", default="CPU")
    return parser


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--backend", choices=("auto", "llama-cpp", "openvino"), default="auto")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = ModelConfig(
            model_path=args.model_path,
            model_id=args.model_id,
            backend=args.backend,
            device=args.device,
            context_length=args.context_length,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
        )
        with KernelLoomModel(config) as model:
            result = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
        print(result.text)
        return 0

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the server dependencies with: pip install kernelloom[server]") from exc
    initial = None
    if args.model_path:
        initial = ModelConfig(args.model_path, model_id=args.model_id, backend=args.backend, device=args.device)
    from .server import create_app

    uvicorn.run(create_app(initial_model=initial), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

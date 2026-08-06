"""KernelLoom command-line interface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from .config import ModelConfig
from .model import KernelLoomModel
from .settings import load_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelloom")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Load a local model and generate one response")
    _model_arguments(run)
    run.add_argument("prompt")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.7)

    chat = subparsers.add_parser("chat", help="Open an interactive chat with one local model")
    _model_arguments(chat)
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.7)

    benchmark = subparsers.add_parser("benchmark", help="Measure end-to-end local generation latency")
    _model_arguments(benchmark)
    benchmark.add_argument("prompt")
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--max-tokens", type=int, default=64)

    embed = subparsers.add_parser("embed", help="Create a vector with a local GGUF embedding model")
    _model_arguments(embed)
    embed.add_argument("text")

    inspect = subparsers.add_parser("inspect", help="Inspect a local model without loading its weights")
    inspect.add_argument("model_path")
    inspect.add_argument("--data-dir", default="")

    hardware = subparsers.add_parser("hardware", help="Show detected local inference hardware")
    hardware.add_argument("--data-dir", default="")
    hardware.add_argument("--refresh", action="store_true")

    subparsers.add_parser("doctor", help="Check the local KernelLoom installation")

    serve = subparsers.add_parser("serve", help="Start the API and browser console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11435)
    serve.add_argument("--model-path", default="")
    serve.add_argument("--model-id", default="default")
    serve.add_argument("--backend", choices=("auto", "llama-cpp", "openvino"), default="auto")
    serve.add_argument("--device", default="CPU")
    serve.add_argument("--config", default="", help="JSON file containing server and model settings")
    serve.add_argument("--max-models", type=int, default=4)
    return parser


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--backend", choices=("auto", "llama-cpp", "openvino"), default="auto")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--batch-threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--mlock", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        _doctor()
        return 0
    if args.command in {"hardware", "inspect"}:
        from openagent_engine import AdaptiveExecutionEngine, HardwareProfiler

        data_dir = args.data_dir or str(Path.home() / ".kernelloom")
        if args.command == "hardware":
            print(json.dumps(HardwareProfiler(data_dir).profile(force=args.refresh).to_dict(), indent=2))
            return 0
        engine = AdaptiveExecutionEngine(data_dir)
        try:
            print(json.dumps(engine.inspect_model(args.model_path), indent=2))
        finally:
            engine.close()
        return 0
    if args.command in {"run", "chat", "benchmark", "embed"}:
        config = _config_from_args(args)
        if args.command == "embed":
            config.embedding = True
        with KernelLoomModel(config) as model:
            if args.command == "run":
                result = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
                print(result.text)
            elif args.command == "chat":
                _interactive_chat(model, args.max_tokens, args.temperature)
            elif args.command == "benchmark":
                _benchmark(model, args.prompt, args.runs, args.max_tokens)
            else:
                vector = model.embed(args.text)
                print(json.dumps({"model": model.config.model_id, "dimensions": len(vector), "embedding": vector}))
        return 0

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the server dependencies with: pip install kernelloom[server]") from exc
    initial = None
    configured: list[ModelConfig] = []
    max_models = args.max_models
    host, port = args.host, args.port
    if args.config:
        runtime = load_runtime_config(args.config)
        configured = runtime.models
        max_models = runtime.max_models
        host, port = runtime.host, runtime.port
    if args.model_path:
        initial = ModelConfig(args.model_path, model_id=args.model_id, backend=args.backend, device=args.device)
    from .server import create_app

    uvicorn.run(
        create_app(initial_model=initial, initial_models=configured, max_models=max_models),
        host=host,
        port=port,
    )
    return 0


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model_path=args.model_path,
        model_id=args.model_id,
        backend=args.backend,
        device=args.device,
        context_length=args.context_length,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        threads=args.threads,
        batch_threads=args.batch_threads,
        gpu_layers=args.gpu_layers,
        flash_attention=args.flash_attention,
        use_mlock=args.mlock,
    )


def _interactive_chat(model: KernelLoomModel, max_tokens: int, temperature: float) -> None:
    messages: list[dict[str, str]] = []
    print("KernelLoom local chat. Type /exit to quit, /clear to reset.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages.clear()
            print("history cleared")
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        print("model> ", end="", flush=True)
        fragments: list[str] = []
        for fragment in model.stream(messages, max_new_tokens=max_tokens, temperature=temperature):
            fragments.append(fragment)
            print(fragment, end="", flush=True)
        print()
        messages.append({"role": "assistant", "content": "".join(fragments)})


def _benchmark(model: KernelLoomModel, prompt: str, runs: int, max_tokens: int) -> None:
    if runs < 1:
        raise SystemExit("--runs must be positive")
    timings: list[float] = []
    characters = 0
    for _ in range(runs):
        started = time.perf_counter()
        result = model.generate(prompt, max_new_tokens=max_tokens, temperature=0)
        timings.append((time.perf_counter() - started) * 1000)
        characters += len(result.text)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(json.dumps({
        "model": model.config.model_id,
        "runs": runs,
        "mean_ms": round(statistics.mean(timings), 3),
        "p50_ms": round(statistics.median(timings), 3),
        "p95_ms": round(p95, 3),
        "characters_per_second": round(characters / (sum(timings) / 1000), 2),
    }, indent=2))


def _doctor() -> None:
    packages = ("llama_cpp", "openvino", "openvino_genai", "fastapi", "langchain_core")
    print(json.dumps({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_threads": __import__("os").cpu_count(),
        "dependencies": {name: importlib.util.find_spec(name) is not None for name in packages},
        "accelerator_python": __import__("os").environ.get("KERNELLOOM_ACCELERATOR_PYTHON", ""),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

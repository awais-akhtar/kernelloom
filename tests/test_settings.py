from pathlib import Path
import json
import tempfile

from kernelloom import load_runtime_config


def test_runtime_config_resolves_relative_model_paths() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
        root = Path(temp)
        path = root / "kernelloom.json"
        path.write_text(json.dumps({
            "server": {"host": "0.0.0.0", "port": 9000, "max_models": 2},
            "models": [{"model_path": "models/chat.gguf", "model_id": "chat"}],
        }), encoding="utf-8")
        config = load_runtime_config(path)
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.max_models == 2
        assert config.models[0].model_path == str((root / "models/chat.gguf").resolve())

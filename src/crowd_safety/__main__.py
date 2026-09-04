import argparse
import json
from pathlib import Path

from .config import ConfigError, load_config
from .runner import benchmark_video, process_video
from .replay import STRATEGIES, replay_run
from .api import create_app
from .persistence import configured_store, import_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline crowd-safety video tools")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate-config", help="validate a TOML pipeline config")
    validate.add_argument("--config", required=True)
    process = commands.add_parser("process-video", help="process a local video offline")
    process.add_argument("--config", required=True)
    process.add_argument("--input")
    benchmark = commands.add_parser("benchmark", help="record offline processing metrics")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--input")
    replay = commands.add_parser("replay", help="replay stored crowd/violence signals")
    replay.add_argument("--run-directory", required=True)
    replay.add_argument("--config", required=True)
    replay.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="all")
    import_command = commands.add_parser("import-run", help="import an offline run into configured storage")
    import_command.add_argument("--config", required=True)
    import_command.add_argument("--run-directory", required=True)
    serve = commands.add_parser("serve-api", help="serve the human-review API")
    serve.add_argument("--config", required=True)
    serve.add_argument("--run-directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--ephemeral", action="store_true", help="use in-memory storage for a local demo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        load_config(args.config)
        print(f"valid config: {args.config}")
    elif args.command == "process-video":
        result = process_video(load_config(args.config), args.input)
        print(result.run_directory)
    elif args.command == "benchmark":
        print(benchmark_video(load_config(args.config), args.input))
    elif args.command == "replay":
        strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
        replay_run(args.run_directory, load_config(args.config), strategies)
        print(Path(args.run_directory).resolve() / "replay")
    elif args.command == "import-run":
        config = load_config(args.config)
        store = configured_store(config)
        imported = import_run(args.run_directory, store, config.m5.evidence_root)
        print(json.dumps({"run_directory": str(Path(args.run_directory).resolve()), "imported": imported}))
    elif args.command == "serve-api":
        import uvicorn

        config = load_config(args.config)
        store = configured_store(config, allow_ephemeral=args.ephemeral)
        if args.run_directory:
            import_run(args.run_directory, store, config.m5.evidence_root)
        uvicorn.run(create_app(store, config.m5.evidence_root), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

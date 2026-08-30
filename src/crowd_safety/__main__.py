import argparse

from .config import ConfigError, load_config
from .runner import benchmark_video, process_video


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface for all benchmark lifecycle operations."""
from __future__ import annotations
import argparse


def get_parser() -> argparse.ArgumentParser:
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument(
        "command",
        nargs="?",
        choices=("validate", "synthetic", "prepare", "evaluate"),
    )
    tmp_args, _ = preliminary_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        prog="mdbench",
        description="MDBench toolkit",
        epilog="Run 'mdbench <command> --help' for command-specific options.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    if tmp_args.command == "validate":
        from ..validate_problem import get_parser as update_parser
        update_parser(commands.add_parser("validate"))
    elif tmp_args.command == "synthetic":
        from ..synthetic_data import get_parser as update_parser
        update_parser(commands.add_parser("synthetic"))
    elif tmp_args.command == "prepare":
        from ..prepare_problem import get_parser as update_parser
        update_parser(commands.add_parser("prepare"))
    elif tmp_args.command == "evaluate":
        from ..evaluate_result import get_parser as update_parser
        update_parser(commands.add_parser("evaluate"))
    else:
        commands.add_parser("validate", help="Validate benchmark problem files")
        commands.add_parser("synthetic", help="Generate synthetic benchmark data")
        commands.add_parser("prepare", help="Prepare benchmark task packages")
        commands.add_parser("evaluate", help="Evaluate benchmark submissions")
    return parser


def main(args) -> int:
    if args.command == "validate":
        from ..validate_problem import main as command_main
        return command_main(args)
    elif args.command == "synthetic":
        from ..synthetic_data import main as command_main
        return command_main(args)
    elif args.command == "prepare":
        from ..prepare_problem import main as command_main
        return command_main(args)
    elif args.command == "evaluate":
        from ..evaluate_result import main as command_main
        return command_main(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def cli() -> int:
    """Parse command-line arguments and dispatch the selected command."""
    from ..utils.logger import config_logger

    config_logger()
    parser = get_parser()
    args = parser.parse_args()
    return main(args)


if __name__ == "__main__":
    raise SystemExit(cli())

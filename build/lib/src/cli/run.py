"""CLI for the proposal benchmark. Legacy evaluation modes are not exposed."""
import argparse
import importlib

COMMANDS = {'validate': 'validate_problem', 'synthetic': 'synthetic_data',
            'export': 'export_problems', 'feedback': 'feedback_server', 'evaluate': 'evaluate'}


def get_parser():
    parser = argparse.ArgumentParser(prog='mdbench')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, module in COMMANDS.items():
        importlib.import_module(f'src.{module}').get_parser(commands.add_parser(name))
    return parser


def main(args):
    return importlib.import_module(f'src.{COMMANDS[args.command]}').main(args)


def cli():
    return main(get_parser().parse_args())

if __name__ == '__main__':
    raise SystemExit(cli())

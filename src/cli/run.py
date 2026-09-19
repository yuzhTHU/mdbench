"""CLI for the proposal benchmark. Legacy evaluation modes are not exposed."""
import argparse
import importlib
import sys

COMMANDS = {
    'validate': 'validate_problem', 
    'synthetic': 'synthetic_data',
    'export': 'export_problems', 
    'feedback': 'feedback_server',
    'run': 'run_experiment',
}

def get_parser(argv=None):
    parser = argparse.ArgumentParser(prog='mdbench')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, module in COMMANDS.items():
        kwargs = {'argv': argv} if name == 'run' else {}
        importlib.import_module(f'src.{module}').get_parser(commands.add_parser(name), **kwargs)
    return parser


def main(args):
    return importlib.import_module(f'src.{COMMANDS[args.command]}').main(args)


def cli():
    argv = sys.argv[1:]
    return main(get_parser(argv).parse_args(argv))

if __name__ == '__main__':
    raise SystemExit(cli())

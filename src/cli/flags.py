"""Small argparse compatibility helpers shared by benchmark entry points."""
import argparse


def add_minus_flags(parser: argparse.ArgumentParser):
    """Add ``--some-name`` aliases for options written as ``--some_name``."""
    for action in parser._actions:
        if not action.option_strings:
            continue
        aliases = []
        for option in action.option_strings:
            if option.startswith('--') and '_' in option:
                alias = option.replace('_', '-')
                if alias not in action.option_strings:
                    aliases.append(alias)
        for alias in aliases:
            action.option_strings.append(alias)
            parser._option_string_actions[alias] = action
    return parser


def add_negation_flags(parser: argparse.ArgumentParser):
    """Add ``--no-...`` counterparts for every ``store_true`` option."""
    existing = {option for action in parser._actions for option in action.option_strings}
    additions = []
    for action in parser._actions:
        if not isinstance(action, argparse._StoreTrueAction):
            continue
        options = []
        for option in action.option_strings:
            if option.startswith('--'):
                negated = '--no-' + option.removeprefix('--')
                if negated not in existing and negated not in options:
                    options.append(negated)
        if options:
            additions.append((options, action))
            existing.update(options)
    for options, action in additions:
        parser.add_argument(*options, dest=action.dest, action='store_false',
                            default=action.default,
                            help=f'Disable {action.option_strings[0].removeprefix("--")}')
    return parser

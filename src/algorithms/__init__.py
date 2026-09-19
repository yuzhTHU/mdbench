"""Algorithm registry. Every algorithm owns run, get_ask and parser methods."""
from importlib import import_module
from pathlib import Path


def list_algorithms():
    return sorted(p.stem for p in Path(__file__).parent.glob('*.py') if not p.stem.startswith('_'))


def _module(name):
    if name not in list_algorithms(): raise ValueError(f'Unknown algorithm: {name}')
    return import_module(f'{__name__}.{name}')


def get_algorithm(name):
    return _module(name).run


def get_ask(name):
    factory = getattr(_module(name), 'get_ask', None)
    if not callable(factory): raise ValueError(f'Algorithm has no get_ask(): {name}')
    return factory


def get_update_parser(name):
    return getattr(_module(name), 'update_parser', None)

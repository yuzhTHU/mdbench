# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import sys
import importlib
from typing import TYPE_CHECKING, Dict, Tuple

__all__ = ["setup_lazy_imports", "TYPE_CHECKING"]

def setup_lazy_imports(module_name: str, import_mapping: Dict[str, Tuple[str, str]]):
    def __getattr__(name: str):
        if name in import_mapping:
            module_path, requires = import_mapping[name]
            try:
                module = importlib.import_module(module_path, package=module_name)
                # Return a matching attribute, or the submodule itself.
                return getattr(module, name) if hasattr(module, name) else module
            except ImportError as e:
                msg = f"Failed to import '{name}' from '{module_path}' in module '{module_name}' since missing optional dependency."
                if not requires:
                    pass
                elif requires == 'all':
                    msg += f"Try to run `pip install .[all]` to install the required dependencies."
                else:
                    msg += f"Try to run `pip install .[{requires}]` or `pip install .[all]` to install the required dependencies."
                raise ImportError(msg) from e
                
        raise AttributeError(f"Module {module_name!r} has no attribute {name!r}")

    def __dir__():
        return list(import_mapping.keys())

    # Build the public namespace from the caller's globals.
    caller_globals = sys._getframe(1).f_globals
    
    __all__ = [name for name in caller_globals.keys() if not name.startswith('_')]
    __all__.extend(import_mapping.keys())

    return __getattr__, __dir__, __all__

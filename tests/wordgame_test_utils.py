"""Load hosted driver definitions without importing the unrelated local-model stack.

Like author_execution_harness.py, this executes production AST definitions.
Public registry/result tests separately exercise normal package imports.
"""

import ast
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    namespace = {**globals(), 'AUTHOR_IMPLEMENTATIONS': ROOT / 'adversariallm/attacks/author_implementations'}
    runtime = ROOT / 'adversariallm/attacks/utils/author_runtime.py'
    names = {'normalize_config', 'load_symbols', 'OpenRouterAuthorClient', 'validate_routes', 'execute'}
    nodes = [node for node in ast.parse(runtime.read_bytes()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runtime), 'exec'), namespace)
    adapter = ROOT / 'adversariallm/attacks/wordgame.py'
    tree = ast.parse(adapter.read_bytes())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in {'WordGameConfig', '_BaseLLM', '_query_client'}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'WordGameAttack')
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'_execute', 'validate_config'}:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(adapter), 'exec'), namespace)
    return SimpleNamespace(config=namespace['WordGameConfig'], execute=namespace['execute'],
                           driver=namespace['_execute'], validate_config=namespace['validate_config'])


CORE = load_core()

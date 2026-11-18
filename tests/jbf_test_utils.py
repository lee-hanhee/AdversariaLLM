"""Load JBF-backed hosted driver definitions without importing the unrelated local-model stack."""

import ast
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def load_jbf_core(module_name, config_name, attack_name):
    namespace = {**globals(), 'AUTHOR_IMPLEMENTATIONS': ROOT / 'adversariallm/attacks/author_implementations'}
    for rel, wanted in (
        ('adversariallm/attacks/utils/author_runtime.py',
         {'normalize_config', 'load_symbols', 'LiteLLMClient', 'OpenRouterAuthorClient', 'validate_routes', 'execute'}),
        ('adversariallm/attacks/utils/jbf_runtime.py',
         {'_BaseLLM', 'make_jbf_query_client', 'run_jbf_universal_attempts'}),
    ):
        path = ROOT / rel
        nodes = [node for node in ast.parse(path.read_bytes()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

    adapter = ROOT / 'adversariallm/attacks' / f'{module_name}.py'
    tree = ast.parse(adapter.read_bytes())
    nodes = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == config_name
        or isinstance(node, ast.FunctionDef)
    )]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == attack_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'_execute', 'validate_config'}:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(adapter), 'exec'), namespace)
    return SimpleNamespace(config=namespace[config_name], execute=namespace['execute'],
                           driver=namespace['_execute'], validate_config=namespace['validate_config'])

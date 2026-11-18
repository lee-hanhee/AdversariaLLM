"""Shared Jailbreak Foundry LLMLiteLLM transport binding for hosted attacks."""

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from .author_runtime import load_symbols


class _BaseLLM:
    """Supply only the framework state used by the preserved LiteLLM wrapper."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.last_query = None


def make_jbf_query_client(transport, role, source_profile, sleep, **llm_kwargs):
    """Bind JBF's LLMLiteLLM wrapper to an OpenRouterAuthorClient role."""
    # Disable nested OpenAI SDK retries externally; JBF's 5-attempt retry loop owns retries.
    def completion(**kwargs):
        kwargs.pop('api_key', None)
        kwargs.pop('api_base', None)
        return transport.create_for(role, **kwargs)

    bindings = {
        'BaseLLM': _BaseLLM, 'CostTracker': Any,
        'Dict': Dict, 'List': List, 'Optional': Optional, 'Union': Union, 'Any': Any, 'Tuple': Tuple,
        'os': os, 'json': json, 'logger': logging.getLogger(__name__),
        'time': SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=sleep),
        'completion': completion,
    }
    source = load_symbols('wordgame/wordgame_llm.py', ['RichResponse', 'RichResponseList', 'LLMLiteLLM'], bindings)
    llm = source['LLMLiteLLM'](
        model_identifier=f'openai/{source_profile}', model_name=source_profile,
        provider='openai', real_provider='openai', log_dir=None, **llm_kwargs)
    # Repository traces replace JBF cost tracking and approximate reasoning-token accounting.
    llm._resolve_cost_tracker = lambda _tracker: None
    llm.get_token_count = lambda text: int(len(text.split()) * 1.33)
    return llm


def run_jbf_universal_attempts(author, target, question, reference, attempts_per_query=1, example_id=0):
    """Execute JBF universal_attack.py's outer attempts_per_query loop."""
    attempts = []
    previous_attempt_failed = None
    for attempt_index in range(1, max(1, int(attempts_per_query)) + 1):
        prompt = author.generate_attack(
            prompt=question,
            goal=question,
            target=reference,
            attempt_index=attempt_index,
            attempts_per_query=attempts_per_query,
            attempt_success_threshold=attempts_per_query,
            previous_attempt_failed=previous_attempt_failed,
            example_id=example_id,
        )
        response = str(target.query(prompt))
        attempts.append({'attempt_index': attempt_index, 'prompt': prompt, 'response': response})
        previous_attempt_failed = True
    return attempts


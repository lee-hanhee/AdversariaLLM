"""Shared I/O and result containers for the preserved author implementations."""

import ast
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import version
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI
from omegaconf import OmegaConf

from ..attack import AttackResult, AttackStepResult, SingleAttackRunResult


def normalize_config(config):
    # Resolve only the selected configuration, including required model and provider fields.
    if is_dataclass(config):
        return asdict(config)
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return dict(config)


@dataclass(kw_only=True)
class AuthorStepResult(AttackStepResult):
    # Keep the source step record and the calls that produced it.
    api_metadata: list[dict] | None = None
    author_result: dict = field(default_factory=dict)
    author_requests: list[dict] = field(default_factory=list)


@dataclass
class AuthorRunResult(SingleAttackRunResult):
    """Keep the full author record even when a run produces no target steps."""

    author_result: dict = field(default_factory=dict)
    author_requests: list[dict] = field(default_factory=list)

    def add_step(self, request, *, messages, completions, raw_completions,
                 author_result, author_requests, scores=None, time_taken=0.0):
        # Share repository bookkeeping; attack branches supply the original values.
        self.steps.append(AuthorStepResult(
            step=len(self.steps), model_input=messages,
            model_completions=completions, model_completions_raw=raw_completions,
            api_metadata=[request['metadata']] if 'metadata' in request else [],
            scores=scores if scores is not None else {}, time_taken=time_taken,
            author_result=author_result, author_requests=author_requests))


@dataclass
class AuthorAttackResult(AttackResult):
    author_summary: dict = field(default_factory=dict)
    author_environment: dict = field(default_factory=dict)


AUTHOR_IMPLEMENTATIONS = Path(__file__).resolve().parents[1] / 'author_implementations'


class OpenRouterTarget:
    """The configuration and SDK client required by the author attacks."""

    def __init__(self, config):
        self.config = normalize_config(config)
        validate_routes(self.config, {}, ())
        load_dotenv(Path(__file__).resolve().parents[3] / '.env', override=False)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError('OPENROUTER_API_KEY is required for hosted attacks.')
        self.client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)


def load_symbols(filename, names, bindings):
    # Execute only declared author definitions; bind imports and storage I/O explicitly.
    path = AUTHOR_IMPLEMENTATIONS / filename
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]

    # Execute the selected source definitions with explicitly supplied I/O globals.
    namespace = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class OpenRouterAuthorClient:
    """Bind author client interfaces to one SDK client and retain per-example traces."""

    def __init__(self, client, model, judge):
        # Share the existing SDK client and keep separate routes and traces for each role.
        self.client = client
        self.routes = {'target': model, 'judge': judge}
        self.requests, self.durations = [], []
        self.index, self.phase = -1, 'target'

        # These are the only OpenAI/Anthropic client interfaces used by the author code.
        self.chat = SimpleNamespace(completions=self)
        self.messages = SimpleNamespace(create=lambda **kwargs: self.create_for(self.phase, _anthropic=True, **kwargs))

    def __call__(self, **_options):
        # Original constructor credentials/endpoints are replaced by the existing hosted client.
        return self

    def create(self, **kwargs):
        # Follow the target/judge phase expected by sequential author clients.
        return self.create_for(self.phase, **kwargs)

    def for_role(self, role):
        # Bind either author SDK interface to a fixed role across repeated attempts.
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: self.create_for(role, **kwargs))),
            messages=SimpleNamespace(create=lambda **kwargs: self.create_for(role, _anthropic=True, **kwargs)))

    def create_for(self, role, _anthropic=False, **kwargs):
        # Explicit roles allow multi-agent attacks to interleave target/attacker/judge calls.
        route = self.routes[role]
        record = {'example': self.index, 'role': role, 'endpoint': str(self.client.base_url),
                  'request': copy.deepcopy(kwargs)}
        self.requests[self.index].append(record)

        # Claude has a separate system field and expects a different response shape.
        routed = copy.deepcopy(kwargs)
        anthropic = _anthropic or 'system' in routed
        if 'system' in routed:
            routed['messages'] = [{'role': 'system', 'content': routed.pop('system')}, *routed['messages']]

        # Pin the configured OpenRouter model and provider while retaining source request controls.
        routed['model'] = route['id']
        routed['extra_body'] = {'provider': {'only': [route['provider']],
                                             'allow_fallbacks': False, 'require_parameters': True}}
        record['routed_request'] = copy.deepcopy(routed)
        try:
            # Keep SDK retries; all application retries remain in the original author methods.
            response = self.client.chat.completions.create(**routed)

            # Capture text and API metadata for the repository result adapter.
            usage = getattr(response, 'usage', None)
            record['metadata'] = {'id': response.id, 'model': response.model,
                                  'usage': usage.model_dump() if usage else None,
                                  'finish_reasons': [choice.finish_reason for choice in response.choices],
                                  'provider': getattr(response, 'provider', None)}
            record['raw_text'] = response.choices[0].message.content if response.choices else None
            self.phase = 'judge'

            # Return the response shape expected by the original author SDK wrapper.
            if anthropic:
                content = [SimpleNamespace(text=record['raw_text'])] if response.choices else []
                return SimpleNamespace(content=content)
            return response
        except Exception as error:
            # Retain the failed call in the trace and propagate the original SDK error.
            record['error_type'] = type(error).__name__
            raise

    def examples(self, values):
        # Group traces as requests happen, avoiding a later scan of the full request history.
        for self.index, value in enumerate(values):
            self.phase = 'target'
            self.requests.append([])
            started = time.perf_counter()
            yield value

            # Finish timing when the author driver completes this example.
            self.durations.append(time.perf_counter() - started)


def validate_routes(model, config, roles):
    """Require explicit hosted routes before any requests are made."""
    if model.get('backend') != 'openrouter':
        raise ValueError('Author attacks require backend=openrouter.')
    for role, route in [('target', model), *((role, config.get(role, {})) for role in roles)]:
        for field in ('id', 'provider'):
            value = route.get(field)
            if not isinstance(value, str) or not value.strip() or value == '???':
                raise ValueError(f'{role} requires an explicit OpenRouter {field}.')
    if config.get('seed') is not None:
        raise ValueError('Author execution does not seed RNG.')
    if model.get('base_url'):
        raise ValueError('Author transport uses only https://openrouter.ai/api/v1.')


def execute(driver, model, config, dataset, *, client, sleep=None):
    """Run an attack's driver with shared request tracing and timing."""
    transport = OpenRouterAuthorClient(client, model, config.get('judge', {}))
    questions = [(conv[0]['content'], conv[1]['content']) for conv in dataset]
    try:
        records, summary = driver(transport, model, config, questions, sleep)
    except Exception as error:
        error.author_requests = [request for trace in transport.requests for request in trace]
        raise
    return {'records': records, 'requests': transport.requests, 'durations': transport.durations,
            'author_summary': summary, 'environment': {'openai': version('openai')}}


def run_attack(attack, target, dataset):
    """Share hosted setup and bookkeeping; each attack owns its driver and steps."""
    if not isinstance(target, OpenRouterTarget):
        raise ValueError('This attack requires the OpenRouter author-driver target.')
    config = attack.config
    model = normalize_config(target.config)
    attack.validate_config(model, config)
    originals = list(dataset)
    for conv in originals:
        if len(conv) != 2 or [message['role'] for message in conv] != ['user', 'assistant']:
            raise ValueError('Author attacks require a user request and an assistant reference answer.')
    result = execute(attack._execute, model, config, originals, client=target.client)
    state = random.getstate()
    try:
        runs = []
        for original, record, trace, duration in zip(
                originals, result['records'], result['requests'], result['durations']):
            run = AuthorRunResult(original_prompt=original, total_time=duration,
                                  author_result=record, author_requests=trace)
            attack._add_steps(run, record, trace, duration)
            runs.append(run)
        return AuthorAttackResult(runs=runs, author_summary=result['author_summary'],
                                  author_environment=result['environment'])
    finally:
        random.setstate(state)

"""Map author execution into the repository's hosted target and result interfaces."""

from dataclasses import asdict, dataclass, field, is_dataclass
import random

from omegaconf import OmegaConf

from ..defenses.api_target import APITarget
from .attack import Attack, AttackResult, AttackStepResult, SingleAttackRunResult
from .author_execution import execute


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
    # Retain driver-wide counters and the SDK environment alongside example results.
    author_summary: dict = field(default_factory=dict)
    author_environment: dict = field(default_factory=dict)


class AuthorAttack(Attack):
    """Shared repository entry point; subclasses select the preserved author driver."""

    def __init__(self, config):
        # The author drivers do not seed RNG, so bypass Attack.__init__.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        # Require the hosted client interface used by the original-driver bindings.
        if not isinstance(target, APITarget):
            raise ValueError(f'{self.AUTHOR_METHOD} requires the OpenRouter author-driver target.')

        # Delegate execution and result conversion through the common author path.
        return run_author(self.AUTHOR_METHOD, target, self.config, dataset)


def run_author(method, target, config, dataset):
    # Reuse the repository's hosted client, while the original driver owns calls and retries.
    model = normalize_config(target.config)
    originals = list(dataset)

    # Validate each dataset item before the author driver starts making requests.
    for conv in originals:
        if len(conv) != 2 or [message['role'] for message in conv] != ['user', 'assistant']:
            raise ValueError('Author attacks require a user request and an assistant reference answer.')

    # Run the selected author driver with the target's existing SDK client.
    result = execute(method, model, config, originals, client=target.text_generator.client)

    # Repository dataclass validation must not consume the author's random stream.
    state = random.getstate()
    try:
        return adapt_results(method, originals, result)
    finally:
        random.setstate(state)


def adapt_results(method, originals, result):
    # Adapt source artifacts to the existing result and logging interface only.
    runs = []
    for original, record, trace, duration in zip(originals, result['records'], result['requests'], result['durations']):
        # Keep the complete author evidence at example level, including runs with no steps.
        run = AuthorRunResult(original_prompt=original, total_time=duration,
                              author_result=record, author_requests=trace)

        # Convert each attack's records into the same repository result hierarchy.
        if method == 'flip_attack':
            target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
            scores = {}

            # Copy optional author evaluation fields without adding another evaluator.
            if 'judge_score_gpt4' in record:
                scores['flip_attack_author'] = {key: [float(record[key])] for key in (
                    'judge_success_dict', 'judge_score_gpt4', 'judge_success_gpt4')}

            # Pair the processed answer with its raw API response and original call trace.
            run.add_step(target_request, messages=record['all_prompt'],
                         completions=[record['output']], raw_completions=[target_request['raw_text']],
                         scores=scores, time_taken=duration, author_result=record, author_requests=trace)

        elif method == 'equacode':
            target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
            run.add_step(target_request, messages=target_request['request']['messages'],
                         completions=[record['response']], raw_completions=[target_request['raw_text']],
                         time_taken=duration, author_result=record, author_requests=trace)

        elif method == 'gta':
            # A GTA run has multiple judged turns, or just backgrounds for zero rounds.
            start = 0
            for judgement in record['judgements']:
                # Slice the calls belonging to this suspect turn and its author judge.
                turn = trace[start:judgement['request_end']]
                target_request = next(request for request in turn if request['role'] == judgement['role'])
                response = judgement['response']
                raw = target_request.get('raw_text')
                run.add_step(target_request, messages=target_request['request']['messages'],
                             completions=[] if response is None else [response],
                             raw_completions=[] if raw is None else [raw],
                             scores={'gta_author': {'rating': [float(judgement['rating'])]}},
                             author_result=judgement, author_requests=turn)

                # Move the boundary forward so later turns keep their own request histories.
                start = judgement['request_end']

        elif method == 'mousetrap':
            # Each original attempt makes one target call followed by one judge call.
            for index, attempt in enumerate(record['attempts']):
                turn = trace[2 * index:2 * index + 2]
                target_request = turn[0]
                response = attempt['response']
                raw = target_request.get('raw_text')
                run.add_step(target_request, messages=target_request['request']['messages'],
                             completions=[] if response is None else [response],
                             raw_completions=[] if raw is None else [raw],
                             scores={'mousetrap_author': {'ats': [float(attempt['ats'])], 'aru': [float(attempt['aru'])]}},
                             author_result=attempt, author_requests=turn)

        elif method == 'query_attack':
            # Physical input lines remain separate target calls within their originating example.
            targets = [request for request in trace if request['role'] == 'target']
            if len(targets) != len(record['lines']):
                raise ValueError('QueryAttack target trace does not match its source-line records.')
            for line, request in zip(record['lines'], targets):
                run.add_step(request, messages=request['request']['messages'],
                             completions=[line['response']], raw_completions=[request['raw_text']],
                             author_result=line, author_requests=[request])

        else:
            raise ValueError(f'Unknown author method: {method}')

        # Collect the completed example after its attack-specific steps have been mapped.
        runs.append(run)

    # Attach the original run summary and SDK environment to the full dataset result.
    return AuthorAttackResult(runs=runs, author_summary=result['author_summary'],
                              author_environment=result['environment'])

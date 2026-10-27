"""Map author execution into the repository's hosted target and result interfaces."""

from dataclasses import asdict, dataclass, field, is_dataclass

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


class AuthorAttack(Attack):
    """Shared repository entry point; subclasses select the preserved author driver."""

    def __init__(self, config):
        # The author drivers do not seed RNG, so bypass Attack.__init__.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        if not isinstance(target, APITarget):
            raise ValueError(f'{self.AUTHOR_METHOD} requires the OpenRouter author-driver target.')
        return run_author(self.AUTHOR_METHOD, target, self.config, dataset)


def run_author(method, target, config, dataset):
    # Reuse the repository's hosted client, while the original driver owns calls and retries.
    model = normalize_config(target.config)
    originals = list(dataset)
    for conv in originals:
        if len(conv) != 2 or [message['role'] for message in conv] != ['user', 'assistant']:
            raise ValueError('Author attacks require a user request and an assistant reference answer.')
    result = execute(method, model, config, originals, client=target.text_generator.client)

    # Adapt source artifacts to the existing result and logging interface only.
    runs = []
    for original, record, trace, duration in zip(originals, result['records'], result['requests'], result['durations']):
        run = AuthorRunResult(original_prompt=original, total_time=duration,
                              author_result=record, author_requests=trace)

        if method == 'flip_attack':
            target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
            scores = {}
            if 'judge_score_gpt4' in record:
                scores['flip_attack_author'] = {key: [float(record[key])] for key in (
                    'judge_success_dict', 'judge_score_gpt4', 'judge_success_gpt4')}
            run.add_step(target_request, messages=record['all_prompt'],
                         completions=[record['output']], raw_completions=[target_request['raw_text']],
                         scores=scores, time_taken=duration, author_result=record, author_requests=trace)

        elif method == 'equacode':
            target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
            run.add_step(target_request, messages=target_request['request']['messages'],
                         completions=[record['response']], raw_completions=[target_request['raw_text']],
                         time_taken=duration, author_result=record, author_requests=trace)

        else:
            raise ValueError(f'Unknown author method: {method}')

        runs.append(run)
    return AuthorAttackResult(runs=runs, author_summary=result['author_summary'],
                              author_environment=result['environment'])

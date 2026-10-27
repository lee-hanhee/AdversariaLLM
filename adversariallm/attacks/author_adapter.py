"""Map author execution into the repository's hosted target and result interfaces."""

from dataclasses import asdict, dataclass, field, is_dataclass

from omegaconf import OmegaConf

from .attack import AttackResult, AttackStepResult, SingleAttackRunResult
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
class AuthorAttackResult(AttackResult):
    author_summary: dict = field(default_factory=dict)
    author_environment: dict = field(default_factory=dict)


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
        target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
        scores = {}
        if method == 'flip_attack':
            messages = record['all_prompt']
            response = record['output']
            if 'judge_score_gpt4' in record:
                scores['flip_attack_author'] = {key: [float(record[key])] for key in (
                    'judge_success_dict', 'judge_score_gpt4', 'judge_success_gpt4')}

        elif method == 'equacode':
            messages = target_request['request']['messages']
            response = record['response']

        else:
            raise ValueError(f'Unknown author method: {method}')

        step = AuthorStepResult(step=0, scores=scores, model_input=messages, model_completions=[response],
                                model_completions_raw=[target_request['raw_text']],
                                api_metadata=[target_request['metadata']], time_taken=duration,
                                author_result=record, author_requests=trace)
        runs.append(SingleAttackRunResult(original_prompt=original, steps=[step], total_time=duration))
    return AuthorAttackResult(runs=runs, author_summary=result['author_summary'],
                              author_environment=result['environment'])

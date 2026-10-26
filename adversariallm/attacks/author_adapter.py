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
    model, config = normalize_config(target.config), normalize_config(config)
    originals = list(dataset)
    for conv in originals:
        if len(conv) != 2 or [message['role'] for message in conv] != ['user', 'assistant']:
            raise ValueError('Author attacks require a user request and an assistant reference answer.')
    result = execute({'method': method, 'model': model, 'config': config, 'dataset': originals},
                     sdk_factory=lambda provider, options: target.text_generator.client)

    # Adapt source artifacts to the existing result and logging interface only.
    runs = []
    for original, item in zip(originals, result['runs']):
        record = item['author_result']
        scores = {}
        if 'judge_score_gpt4' in record:
            scores['flip_attack_author'] = {key: [float(record[key])] for key in (
                'judge_success_dict', 'judge_score_gpt4', 'judge_success_gpt4')}
        step = AuthorStepResult(step=0, scores=scores, model_input=item['messages'], model_completions=[item['response']],
                                model_completions_raw=[item['raw_response']], api_metadata=[item['api_metadata']],
                                time_taken=item['time_taken'], author_result=item['author_result'],
                                author_requests=item['requests'])
        runs.append(SingleAttackRunResult(original_prompt=original, steps=[step], total_time=item['time_taken']))
    return AuthorAttackResult(runs=runs, author_summary=result['author_summary'],
                              author_environment=result['environment'])

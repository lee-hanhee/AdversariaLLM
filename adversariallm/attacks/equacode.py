"""Repository binding for the original EquaCode driver and mandatory judge flow."""

from dataclasses import dataclass, field

from .attack import Attack
from .utils.author_runtime import (
    load_symbols, normalize_config, run_attack, validate_routes,
)


@dataclass
class EquaCodeConfig:
    # Use repository identification without adding author-absent RNG seeding.
    name: str = 'equacode'
    type: str = 'discrete'
    version: str = '0.2.0'
    seed: None = None

    # Select original model handling independently of the hosted model ID.
    source_profile: str = 'gpt-3.5-turbo'

    # Retain the main-script template selection and literal judge ID.
    variant: str = 'equacode'
    judge_model: str = 'GPT-4'

    # Require an explicit OpenRouter model and provider for the author judge.
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class EquaCodeAttack(Attack):
    """Run the preserved equacode algorithm through the hosted target."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        # Preserve the source algorithm's unseeded random stream.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('judge',))
        if config.get('source_profile') != 'gpt-3.5-turbo':
            raise ValueError('EquaCode source_profile must retain gpt-3.5-turbo.')
        if config.get('judge_model') != 'GPT-4':
            raise ValueError('EquaCode judge_model must retain the source label GPT-4.')
        if config['variant'] not in {'equacode', 'equation', 'code', 'origin'}:
            raise ValueError('Unknown EquaCode variant.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
        run.add_step(target_request, messages=target_request['request']['messages'],
                     completions=[record['response']], raw_completions=[target_request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        questions = [question for question, _ in questions]
        # Load the unchanged prompt variants and message-construction helper.
        records = []
        templates = load_symbols('equacode/equacode_prompts.py', ['equacoder', 'equa', 'coder', 'origin'], {})
        helper = load_symbols('equacode/equacode_utils.py', ['synthesis_message'], {})

        # Capture the author's spreadsheet row unchanged.
        def save(row, fields, filename):
            records.append({'question': fields[0], 'prompt': fields[1], 'response': fields[2],
                            'evaluation': fields[3], 'row': row})

        # Bind the original driver's API constructors and spreadsheet callback to repository I/O.
        driver = load_symbols('equacode/equacode_driver.py', ['targetLLM'], {
            **templates, **helper, 'os': __import__('os'), 'OpenAI': transport,
            'openai_key': transport.client.api_key, 'base_url': str(transport.client.base_url),
            'assessmodel': config['judge_model'], 'assess_base_url': str(transport.client.base_url), 'save_data': save,
        })['targetLLM']

        # Run the original generation/judging flow without creating spreadsheet files.
        author = driver(questions=transport.examples(questions), target_model=config['source_profile'], text=config['variant'])
        author.filecheck = lambda: None
        author.gernerate()

        return records, {}

"""Bind the preserved EquaCode driver to repository I/O."""

from ...author_execution import load_symbols


def execute(transport, model, config, questions, sleep):
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
    author = driver(questions=transport.examples(questions), target_model=model['author_model'], text=config['variant'])
    author.filecheck = lambda: None
    author.gernerate()

    return records, {}

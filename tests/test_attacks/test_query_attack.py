"""Compare complete official QueryAttack stages and the hosted binding on benign inputs."""

import ast
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from adversariallm.attacks import Attack
from adversariallm.attacks.author_adapter import AuthorRunResult, AuthorStepResult
from adversariallm.attacks.author_execution import validate_author_config
from adversariallm.attacks.query_attack import QueryAttack, QueryAttackConfig
from adversariallm.defenses.api_target import APITarget
from adversariallm.defenses.registry import validate_hosted_target
from tests.author_test_utils import ROOT, author_payload, compare_author


LANGUAGES = ['C++', 'C', 'C#', 'Python', 'Go', 'SQL', 'Java', 'JavaScript', 'URL']


def extracted(content='explain rainbow method', source='Science Handbook', item='rainbow'):
    # Provide every field required by the unchanged author parser.
    return {'text': json.dumps({'Request Content': content, 'Content Source': source,
                               'Key Object': item, 'Risk Level': 'Safe', 'Violation Type': 'None'})}


def payload(**settings):
    # Keep ordinary comparisons single-line; dedicated cases exercise physical-line parsing.
    value = author_payload('query_attack', **settings)
    value['dataset'][0][0]['content'] = 'Explain a rainbow.'
    value['replies'] = [extracted(), {'text': '  A benign answer.\nSecond line.  '}]
    return value


@pytest.mark.parametrize('language', LANGUAGES)
def test_all_nine_languages_preserve_files_prompts_controls_and_raw_text(language):
    # Translate once into all original languages, then run the selected target CLI profile.
    value = payload(target_language=language)
    original, adapted = compare_author(value)
    row = adapted['records'][0]['lines'][0]

    # Keep the writer/reader delimiter mismatch and newline-escaped target output.
    assert row['translation'].endswith('\n----')
    assert row['response'] == '  A benign answer.\\nSecond line.  '
    assert row['phrases'][0]['sentence_index'] == '0'
    assert len(adapted['author_summary']['files']) == 11
    assert len(original['wire']) == len(adapted['wire']) == 2
    helper, target = [request['body'] for request in adapted['wire']]
    assert not set(helper) & {'temperature', 'top_p', 'seed', 'max_tokens'}
    assert {key: target[key] for key in ('stream', 'max_tokens', 'temperature', 'top_p', 'seed')} == {
        'stream': False, 'max_tokens': 512, 'temperature': 0, 'top_p': 1, 'seed': 0}
    assert [message['role'] for message in target['messages']] == ['system', 'user', 'assistant', 'user']


@pytest.mark.parametrize('language', LANGUAGES)
def test_single_language_translation_preserves_cpp_missing_output(language):
    # Keep the author's single-language path, including its missing C++ store_data call.
    _, result = compare_author(payload(translation_language=language, target_language=language))
    if language == 'C++':
        assert result['error']['type'] == 'FileNotFoundError'
        assert len(result['wire']) == 1
    else:
        assert len(result['author_summary']['files']) == 3
        assert len(result['records'][0]['lines']) == 1


def test_split_threshold_retries_batch_order_and_phrase_csv_round_trip():
    # The second request exceeds the original 13-token gate and splits into two phrases.
    value = payload()
    value['dataset'] = [[{'role': 'user', 'content': text}, {'role': 'assistant', 'content': ''}]
                        for text in ('Explain rainbows.', ' '.join(['rainbow'] * 14))]
    value['replies'] = [extracted(), {'text': 'not an array'},
                        {'text': '["Explain rainbows.", "Describe colors."]'},
                        {'text': 'missing fields'}, extracted('describe colors'), extracted('explain light'),
                        {'text': 'first\nanswer'}, {'text': 'second\nanswer'}]
    _, result = compare_author(value)

    # All splitting/extraction finishes before either target request starts.
    assert [request['body']['model'] for request in result['wire']] == ['test/translator'] * 6 + ['test/target'] * 2
    assert [len(record['lines'][0]['phrases']) for record in result['records']] == [1, 2]
    assert result['records'][1]['lines'][0]['phrases'][1]['sentence_index'] == '1'


def test_multiline_crlf_blank_lines_and_whitespace_follow_original_input_reader():
    # Preserve physical-line splitting instead of treating multiline messages as one source request.
    value = payload()
    value['dataset'] = [[{'role': 'user', 'content': '  Explain rainbows.\r\n\nDescribe colors.  '},
                         {'role': 'assistant', 'content': ''}],
                        [{'role': 'user', 'content': 'Explain light.'}, {'role': 'assistant', 'content': ''}]]
    value['replies'] = [extracted()] * 4 + [{'text': f'answer {index}\n'} for index in range(4)]
    _, result = compare_author(value)
    assert [len(record['lines']) for record in result['records']] == [3, 1]
    assert [row['question'] for row in result['records'][0]['lines']] == ['Explain rainbows.', '', 'Describe colors.']
    assert [len(trace) for trace in result['requests']] == [6, 2]


@pytest.mark.parametrize('verify', [False, True])
def test_original_translation_verification_remains_optional(verify):
    # Verification uses the original helper route and does not add a post-run evaluator.
    value = payload(trans_verify=verify)
    value['replies'] = [extracted()] + ([{'text': 'no'}] * 3 if verify else []) + [{'text': 'Answer.'}]
    _, result = compare_author(value)
    assert len(result['wire']) == (5 if verify else 2)
    assert all(request['body']['model'] == 'test/translator' for request in result['wire'][:-1])


def test_original_unsafe_phrase_reversal_and_verification_control_flow():
    # Trigger the source retry/reversal branch and then satisfy its next safety check.
    value = payload(trans_verify=True)
    value['replies'] = [extracted(), {'text': 'yes'}, extracted('red green blue'),
                        {'text': 'yes'}, {'text': 'no'}, {'text': 'no'},
                        *[{'text': 'no'}] * 3, {'text': 'Answer.'}]
    _, result = compare_author(value)
    assert result['records'][0]['lines'][0]['phrases'][0]['Content'] == 'blue green red'
    assert len(result['wire']) == 10


def test_original_nltk_filter_does_not_rewrite_returned_fields():
    # Retain the source's temporary-list assignment quirk after actual tokenization and tagging.
    value = payload()
    value['replies'][0] = extracted('explain a rainbow and colors')
    _, result = compare_author(value)
    assert result['records'][0]['lines'][0]['phrases'][0]['Content'] == 'explain a rainbow and colors'


@pytest.mark.parametrize('item,error', [({'text': None}, 'TypeError'),
                                       ({'text': 'Answer', 'empty_choices': True}, 'IndexError'),
                                       ({'status': 404}, 'NotFoundError')])
@pytest.mark.parametrize('phase', ['helper', 'target'])
def test_original_api_and_response_shape_errors_are_not_repaired(item, error, phase):
    # Preserve errors from both original call sites without adding application retries.
    value = payload()
    value['replies'] = [item] if phase == 'helper' else [extracted(), item]
    _, result = compare_author(value)
    expected = 'AttributeError' if phase == 'target' and item.get('text', '') is None else error
    assert result['error']['type'] == expected
    assert len(result['wire']) == (1 if phase == 'helper' else 2)


@pytest.mark.parametrize('phase', ['helper', 'target'])
def test_sdk_retry_body_and_delay_match_original_baseline(phase):
    # Compare real author/shared SDK serializers and their existing transient-error retry layer.
    value = payload()
    value['replies'].insert(0 if phase == 'helper' else 1, {'status': 500})
    _, result = compare_author(value)
    assert len(result['wire']) == 3
    assert len(result['sdk_sleeps']) == 1
    assert result['sleeps'] == []


def test_empty_dataset_still_runs_original_empty_file_stages():
    # Preserve the original scripts' empty artifacts without issuing API calls.
    value = payload()
    value['dataset'] = []
    value['replies'] = []
    _, result = compare_author(value)
    assert result['records'] == [] and result['wire'] == []
    assert len(result['author_summary']['files']) == 11


@pytest.mark.parametrize('role,field', [('target', 'id'), ('target', 'provider'),
                                        ('translator', 'id'), ('translator', 'provider')])
def test_each_route_requires_explicit_model_and_provider(role, field):
    value = payload()
    route = value['model'] if role == 'target' else value['config']['translator']
    route[field] = None

    # Fail configuration before loading source, NLTK resources, datasets, or clients.
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        validate_author_config('query_attack', value['model'], value['config'])


@pytest.mark.parametrize('setting,value', [('translation_language', 'Rust'), ('target_language', 'ALL'),
    ('trans_verify', 'False'), ('request_seed', True), ('max_new_tokens', 1.5), ('temperate', '0'),
    ('top_p', False), ('seed', 0), ('judge', {}), ('generation_config', {})])
def test_unsupported_configuration_is_rejected(setting, value):
    config = payload()
    config['config'][setting] = value
    with pytest.raises(ValueError):
        validate_author_config('query_attack', config['model'], config['config'])


def test_registry_yaml_and_defaults_match_original_cli_declarations():
    # Compose the public configuration and compare both script defaults against official source.
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'models=query_attack_author', 'model=query_attack_author',
            'attack=query_attack', 'models.query_attack_author.id=test/target',
            'models.query_attack_author.provider=target-provider',
            'attacks.query_attack.translator.id=test/translator',
            'attacks.query_attack.translator.provider=translator-provider'])
    validate_hosted_target(Attack.from_name('query_attack'), cfg.models.query_attack_author,
                           None, cfg.attacks.query_attack)
    assert Attack.from_name('query_attack') is QueryAttack
    assert set(asdict(QueryAttackConfig())) == set(cfg.attacks.query_attack)

    # Read default literals directly from each original argparse call.
    def defaults(filename):
        tree = ast.parse((ROOT / 'verification/query_attack/reference' / filename).read_bytes())
        return {node.args[0].value: ast.literal_eval(keyword.value)
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                for keyword in node.keywords if keyword.arg == 'default'}
    original_translation, original_target = defaults('translator.py'), defaults('test.py')
    config = QueryAttackConfig()
    assert config.translation_language == original_translation['--target_language']
    assert config.target_language == original_target['--target_language']
    assert config.trans_verify == original_translation['--trans_verify']
    for field in ('temperate', 'top_p', 'max_new_tokens'):
        assert getattr(config, field) == original_target['--' + field]
    assert config.request_seed == original_target['--seed']


def test_repository_entrypoint_keeps_common_results_and_only_openrouter_credentials(monkeypatch):
    import httpx
    from openai import OpenAI

    # Use the actual shared SDK with entirely in-memory HTTP replies.
    value = payload()
    value['dataset'][0][0]['content'] = 'Explain rainbows.\nDescribe colors.'
    replies = iter([extracted(), extracted(), {'text': 'first\nanswer'}, {'text': 'second\nanswer'}])
    wire = []
    def http(request):
        wire.append(json.loads(request.content))
        assert request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={
            'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': wire[-1]['model'],
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': next(replies)['text']},
                         'finish_reason': 'stop'}]}, request=request)
    client = OpenAI(api_key='test-key', base_url='https://openrouter.ai/api/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(http)))
    target = object.__new__(APITarget)
    target.config = value['model']
    target.text_generator = SimpleNamespace(client=client)
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(key, raising=False)

    # The public attack uses the common hierarchy and retains helper calls at run level.
    result = QueryAttack(value['config']).run(target, value['dataset'])
    run = result.runs[0]
    assert isinstance(run, AuthorRunResult) and len(run.steps) == 2
    assert all(isinstance(step, AuthorStepResult) for step in run.steps)
    assert [request['role'] for request in run.author_requests] == ['translator', 'translator', 'target', 'target']
    assert run.steps[0].model_completions == ['first\\nanswer']
    assert run.steps[0].model_completions_raw == ['first\nanswer']
    assert run.steps[0].scores == {}
    assert len(asdict(result)['runs'][0]['author_result']['lines']) == 2


def test_missing_nltk_data_fails_before_any_request(monkeypatch):
    import nltk
    from adversariallm.attacks.author_execution import OpenRouterAuthorClient
    from adversariallm.attacks.utils.query_attack import binding

    # Keep missing-resource failures explicit, without silently substituting tokenizers or downloading.
    value = payload()
    def missing(*args, **kwargs):
        raise LookupError('Missing NLTK resource')
    monkeypatch.setattr(nltk, 'word_tokenize', missing)
    client = OpenRouterAuthorClient(None, value['model'], {})
    with pytest.raises(LookupError, match='Missing NLTK resource'):
        binding.execute(client, value['model'], value['config'], ['Explain rainbows.'], None)
    assert client.requests == []


@pytest.mark.parametrize('tokens,helper_calls', [(13, 1), (14, 2)])
def test_original_split_gate_uses_nltk_token_count(tokens, helper_calls):
    # Exercise both sides of the author's exact token-count threshold.
    value = payload()
    value['dataset'][0][0]['content'] = ' '.join(['rainbow'] * tokens)
    value['replies'] = ([{'text': '["Explain rainbows."]'}] if tokens > 13 else []) + [extracted(), {'text': 'Answer.'}]
    _, result = compare_author(value)
    assert len(result['wire']) == helper_calls + 1


@pytest.mark.parametrize('escaped', [False, True])
def test_unicode_braces_and_csv_punctuation_preserve_original_serialization(escaped):
    # Keep Unicode and punctuation through the original regex parser, CSV, and language templates.
    value = payload()
    value['dataset'][0][0]['content'] = 'Décris les couleurs {du ciel}.'
    value['replies'][0] = extracted('décrire café, couleurs {du ciel}', 'Science, {Guide}', "Children's rainbow")
    value['replies'][0]['text'] = json.dumps(json.loads(value['replies'][0]['text']), ensure_ascii=escaped)
    _, result = compare_author(value)
    row = result['records'][0]['lines'][0]
    expected = r'd\u00e9crire caf\u00e9, couleurs {du ciel}' if escaped else 'décrire café, couleurs {du ciel}'
    assert row['phrases'][0]['Content'] == expected
    assert "Children's rainbow" in row['translation']


def test_original_target_cli_overrides_do_not_change_helper_defaults():
    # Keep target request overrides separate from the helper's omitted generation settings.
    _, result = compare_author(payload(temperate=0.4, top_p=0.7, max_new_tokens=123, request_seed=42))
    helper, target = [request['body'] for request in result['wire']]
    assert {key: target[key] for key in ('temperature', 'top_p', 'max_tokens', 'seed')} == {
        'temperature': 0.4, 'top_p': 0.7, 'max_tokens': 123, 'seed': 42}
    assert not set(helper) & {'temperature', 'top_p', 'max_tokens', 'seed'}


def test_delimiters_inside_fields_preserve_extra_calls_and_source_ownership():
    # Let the original parser split an extracted field, followed by a separate ordinary input line.
    value = payload()
    value['dataset'].append([{'role': 'user', 'content': 'Describe colors.'},
                             {'role': 'assistant', 'content': ''}])
    broken = extracted('explain\n---\ncolors')['text'].replace('\\n', '\n')
    value['replies'] = [{'text': broken}, extracted(),
                        {'text': 'fragment one'}, {'text': 'fragment two'}, {'text': 'second input'}]
    _, result = compare_author(value)

    # Preserve all three target calls and retain distinct source and target indices.
    assert [len(record['lines']) for record in result['records']] == [2, 1]
    rows = [row for record in result['records'] for row in record['lines']]
    assert [row['sentence_index'] for row in rows] == [0, 0, 1]
    assert [row['target_index'] for row in rows] == [0, 1, 2]
    assert [len(trace) for trace in result['requests']] == [3, 2]


def test_carriage_returns_in_target_output_follow_original_print_and_file_behavior():
    # The original replaces newlines only; captured output files additionally use universal newlines.
    value = payload()
    value['replies'][-1] = {'text': 'first\r\nsecond\rthird'}
    _, result = compare_author(value)
    assert result['records'][0]['lines'][0]['response'] == 'first\r\\nsecond\rthird'

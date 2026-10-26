"""Compare full author behavior using FlipAttack's pinned native SDKs offline."""

from itertools import product

import pytest

from adversariallm.attacks.author_execution import FLIP_MODELS
from tests.author_test_utils import author_payload, compare_author


@pytest.mark.parametrize('mode', ['FWO', 'FCW', 'FCS', 'FMM'])
@pytest.mark.parametrize('cot,lang_gpt,few_shot', list(product([False, True], repeat=3)))
def test_all_guidance_combinations(mode, cot, lang_gpt, few_shot):
    payload = author_payload('flip_attack', flip_mode=mode, cot=cot, lang_gpt=lang_gpt, few_shot=few_shot)
    _, result = compare_author(payload)
    assert result['runs'][0]['response'] == 'A benign answer.'
    assert result['sleeps'] == [1]
    assert result['environment'] == {'openai': '1.109.1'}


@pytest.mark.parametrize('model,provider', list(FLIP_MODELS.items()))
def test_original_models_endpoints_and_serialization(model, provider):
    payload = author_payload('flip_attack')
    payload['model']['author_model'] = model
    _, result = compare_author(payload)
    request = result['wire'][0]
    if provider == 'anthropic':
        assert request['body']['max_tokens'] == 3584
        assert request['body']['messages'][0]['role'] == 'system'
        assert request['url'].startswith('https://openrouter.ai/')
        assert result['runs'][0]['response'] == '  A benign answer.\n'
        assert result['runs'][0]['messages'][0]['role'] == 'system'
    else:
        assert result['runs'][0]['response'] == 'A benign answer.'
        assert 'max_tokens' not in request['body']
        assert request['url'].startswith('https://openrouter.ai/')


@pytest.mark.parametrize('judge_reply,expected', [('Rating: [[10]]', 10), ('unparseable', 1), ('[[99]]', 99)])
def test_optional_original_evaluator_and_parser(judge_reply, expected):
    payload = author_payload('flip_attack', evaluate=True)
    payload['replies'].append({'text': judge_reply})
    _, result = compare_author(payload)
    record = result['runs'][0]['author_result']
    assert record['judge_score_gpt4'] == expected
    assert record['judge_success_gpt4'] == int(expected == 10)
    assert result['sleeps'] == [1, 1]
    assert [r['role'] for r in result['runs'][0]['requests']] == ['target', 'judge']


@pytest.mark.parametrize('text,success', [("I'm sorry, no.", 0), ("i'm sorry, no.", 1)])
def test_keyword_evaluator_keeps_case_sensitive_author_behavior(text, success):
    payload = author_payload('flip_attack', evaluate=True)
    payload['replies'] = [{'text': text}, {'text': '[[1]]'}]
    _, result = compare_author(payload)
    assert result['runs'][0]['author_result']['judge_success_dict'] == success


def test_retry_count_sleep_and_success_postprocessing():
    payload = author_payload('flip_attack', retry_time=2, failed_sleep_time=3, round_sleep_time=2)
    payload['replies'] = [{'status': 400}, {'status': 400}, {'text': '  Done.\n'}]
    _, result = compare_author(payload)
    assert result['sleeps'] == [3, 3, 2]
    assert len(result['wire']) == 3 and result['runs'][0]['response'] == 'Done.'


def test_author_retry_exhaustion_is_not_replaced_with_an_empty_response():
    payload = author_payload('flip_attack', retry_time=2)
    payload['replies'] = [{'status': 404}] * 3
    _, result = compare_author(payload)
    assert result['error']['type'] == 'UnboundLocalError'
    assert len(result['wire']) == 3 and result['sleeps'] == [1, 1]


def test_original_max_token_option_remains_ignored_by_gpt_client():
    payload = author_payload('flip_attack', max_token=12)
    _, result = compare_author(payload)
    assert not {'max_tokens', 'max_completion_tokens'}.intersection(result['wire'][0]['body'])


def test_repeated_examples_create_fresh_builders_and_align_results():
    payload = author_payload('flip_attack', cot=True, lang_gpt=True, few_shot=True)
    payload['dataset'] *= 2
    payload['replies'] = [{'text': 'First.'}, {'text': 'Second.'}]
    _, result = compare_author(payload)
    assert result['wire'][0]['body'] == result['wire'][1]['body']
    assert [run['response'] for run in result['runs']] == ['First.', 'Second.']


@pytest.mark.parametrize('text', ['Hi', ''])
def test_author_short_input_exceptions_are_preserved(text):
    payload = author_payload('flip_attack', few_shot=True)
    payload['dataset'][0][0]['content'] = text
    _, result = compare_author(payload)
    assert result['wire'] == [] and 'error' in result


def test_original_default_allows_one_initial_attempt_and_one_thousand_retries():
    payload = author_payload('flip_attack')
    payload['replies'] = [{'status': 404}] * 1001
    _, result = compare_author(payload)
    assert len(result['wire']) == 1001
    assert result['sleeps'] == [1] * 1000
    assert result['error']['type'] == 'UnboundLocalError'


def test_judge_keeps_its_original_client_defaults_instead_of_target_retry_overrides():
    payload = author_payload('flip_attack', evaluate=True, retry_time=0)
    payload['replies'] = [{'text': 'Answer.'}, {'status': 404}, {'text': 'Rating: [[10]]'}]
    _, result = compare_author(payload)
    assert [r['role'] for r in result['runs'][0]['requests']] == ['target', 'judge', 'judge']
    assert result['sleeps'] == [1, 1, 1]

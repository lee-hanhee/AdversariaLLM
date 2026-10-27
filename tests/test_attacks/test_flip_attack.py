"""Compare full author behavior using FlipAttack's pinned native SDKs offline."""

from itertools import product

import pytest

from adversariallm.attacks.author_execution import FLIP_MODELS
from tests.author_test_utils import author_payload, compare_author


@pytest.mark.parametrize('mode', ['FWO', 'FCW', 'FCS', 'FMM'])
@pytest.mark.parametrize('cot,lang_gpt,few_shot', list(product([False, True], repeat=3)))
def test_all_guidance_combinations(mode, cot, lang_gpt, few_shot):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', flip_mode=mode, cot=cot, lang_gpt=lang_gpt, few_shot=few_shot)

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Preserve prompt generation, response stripping, and source sleep behavior for each variant.
    assert result['records'][0]['output'] == 'A benign answer.'
    assert result['sleeps'] == [1]
    assert result['environment'] == {'openai': '1.109.1'}


@pytest.mark.parametrize('model,provider', list(FLIP_MODELS.items()))
def test_original_models_endpoints_and_serialization(model, provider):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack')
    payload['model']['author_model'] = model

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)
    request = result['wire'][0]

    # Preserve each source model's request fields and response processing after routing translation.
    if provider == 'anthropic':
        assert request['body']['max_tokens'] == 3584
        assert request['body']['messages'][0]['role'] == 'system'
        assert request['url'].startswith('https://openrouter.ai/')
        assert result['records'][0]['output'] == '  A benign answer.\n'
        assert result['records'][0]['all_prompt'][0]['role'] == 'system'
    else:
        assert result['records'][0]['output'] == 'A benign answer.'
        assert 'max_tokens' not in request['body']
        assert request['url'].startswith('https://openrouter.ai/')


@pytest.mark.parametrize('judge_reply,expected', [('Rating: [[10]]', 10), ('unparseable', 1), ('[[99]]', 99)])
def test_optional_original_evaluator_and_parser(judge_reply, expected):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', evaluate=True)
    payload['replies'].append({'text': judge_reply})

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)
    record = result['records'][0]

    # Retain the original judge parser, success rule, and target/judge call order.
    assert record['judge_score_gpt4'] == expected
    assert record['judge_success_gpt4'] == int(expected == 10)
    assert result['sleeps'] == [1, 1]
    assert [r['role'] for r in result['requests'][0]] == ['target', 'judge']


@pytest.mark.parametrize('text,success', [("I'm sorry, no.", 0), ("i'm sorry, no.", 1)])
def test_keyword_evaluator_keeps_case_sensitive_author_behavior(text, success):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', evaluate=True)
    payload['replies'] = [{'text': text}, {'text': '[[1]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the author keyword evaluator's case-sensitive result.
    assert result['records'][0]['judge_success_dict'] == success


def test_retry_count_sleep_and_success_postprocessing():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', retry_time=2, failed_sleep_time=3, round_sleep_time=2)
    payload['replies'] = [{'status': 400}, {'status': 400}, {'text': '  Done.\n'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Retain original application retry delays and successful-response stripping.
    assert result['sleeps'] == [3, 3, 2]
    assert len(result['wire']) == 3 and result['records'][0]['output'] == 'Done.'


def test_author_retry_exhaustion_is_not_replaced_with_an_empty_response():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', retry_time=2)
    payload['replies'] = [{'status': 404}] * 3

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Preserve the source exhaustion exception and its exact application attempt count.
    assert result['error']['type'] == 'UnboundLocalError'
    assert len(result['wire']) == 3 and result['sleeps'] == [1, 1]


def test_original_max_token_option_remains_ignored_by_gpt_client():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', max_token=12)

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Leave the ignored GPT output-cap option absent from the wire request.
    assert not {'max_tokens', 'max_completion_tokens'}.intersection(result['wire'][0]['body'])


def test_repeated_examples_create_fresh_builders_and_align_results():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', cot=True, lang_gpt=True, few_shot=True)
    payload['dataset'] *= 2
    payload['replies'] = [{'text': 'First.'}, {'text': 'Second.'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Rebuild prompts per example and preserve result order.
    assert result['wire'][0]['body'] == result['wire'][1]['body']
    assert [record['output'] for record in result['records']] == ['First.', 'Second.']


@pytest.mark.parametrize('text', ['Hi', ''])
def test_author_short_input_exceptions_are_preserved(text):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', few_shot=True)
    payload['dataset'][0][0]['content'] = text

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the original short-input failure before any API call.
    assert result['wire'] == [] and 'error' in result


def test_original_default_allows_one_initial_attempt_and_one_thousand_retries():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack')
    payload['replies'] = [{'status': 404}] * 1001

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Retain the original initial attempt plus one thousand application retries.
    assert len(result['wire']) == 1001
    assert result['sleeps'] == [1] * 1000
    assert result['error']['type'] == 'UnboundLocalError'


def test_judge_keeps_its_original_client_defaults_instead_of_target_retry_overrides():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', evaluate=True, retry_time=0)
    payload['replies'] = [{'text': 'Answer.'}, {'status': 404}, {'text': 'Rating: [[10]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep judge retries independent of target retry overrides.
    assert [r['role'] for r in result['requests'][0]] == ['target', 'judge', 'judge']
    assert result['sleeps'] == [1, 1, 1]


def test_retries_and_judges_stay_grouped_with_their_original_example():
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', evaluate=True, retry_time=1)
    payload['dataset'] *= 2
    payload['replies'] = [{'status': 404}, {'text': 'First.'}, {'status': 400}, {'text': '[[10]]'},
                          {'text': 'Second.'}, {'text': '[[1]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep retries and evaluations attached to their original dataset example.
    assert [record['output'] for record in result['records']] == ['First.', 'Second.']
    assert [[request['role'] for request in trace] for trace in result['requests']] == [
        ['target', 'target', 'judge', 'judge'], ['target', 'judge']]
    for index, trace in enumerate(result['requests']):
        assert all(request['example'] == index for request in trace)
    assert len(result['durations']) == 2
    assert result['author_summary'] == {'all_count': 2, 'dict_success_count': 2, 'gpt_success_count': 1}


@pytest.mark.parametrize('model', ['gpt-4-0613', 'claude-3-5-sonnet-20240620'])
def test_empty_api_choices_reach_original_response_processing_without_added_retries(model):
    # Prepare the source options and scripted target/judge replies for this case.
    payload = author_payload('flip_attack', retry_time=1)
    payload['model']['author_model'] = model
    payload['replies'] = [{'text': '', 'empty_choices': True}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Let the source response-processing exception surface without another application call.
    assert result['error']['type'] == 'IndexError'
    assert len(result['wire']) == 1 and result['sleeps'] == []

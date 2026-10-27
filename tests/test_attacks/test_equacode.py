"""Compare EquaCode's unchanged driver and mandatory evaluation flow offline."""

import pytest

from tests.author_test_utils import author_payload, compare_author


@pytest.mark.parametrize('variant', ['equacode', 'equation', 'code', 'origin'])
@pytest.mark.parametrize('text', ['Explain {key: value}.\nUse café examples.',
                                 '  Keep [INSERT PROMPT HERE] literally.\t🌈  ', ''])
def test_original_variants_placeholder_handling_and_default_requests(variant, text):
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode', variant=variant)
    payload['dataset'][0][0]['content'] = text
    payload['replies'].append({'text': 'Rating: [[10]]'})

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Preserve all template substitutions and omit extra generation controls.
    assert len(result['wire']) == 2
    assert result['records'][0]['response'] == '  A benign answer.\n'
    assert result['records'][0]['evaluation'] == 'Rating: [[10]]'
    assert result['environment'] == {'openai': '1.109.1'}
    for request in result['wire']:
        assert set(request['body']) == {'model', 'messages', 'provider'}
    assert result['wire'][1]['body']['model'] == 'test/judge'


@pytest.mark.parametrize('replies,calls', [
    (['bad', 'Rating: [[4]]'], 2),
    (['rating lowercase is not accepted'] * 6, 6),
])
def test_original_judge_retry_condition_and_six_attempt_limit(replies, calls):
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['replies'] += [{'text': text} for text in replies]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the case-sensitive judge stop condition and six-call limit.
    assert len(result['wire']) == 1 + calls
    assert result['records'][0]['evaluation'] == replies[-1]
    assert result['sleeps'] == []


def test_no_added_application_retries_or_model_fallbacks():
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['replies'] = [{'status': 404}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Propagate the original failed request without another application call.
    assert result['error']['type'] == 'NotFoundError' and len(result['wire']) == 1


def test_original_empty_completion_is_passed_to_judge():
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['replies'] = [{'text': ''}, {'text': 'Rating: [[1]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Pass an empty target answer to the mandatory source judge unchanged.
    assert result['records'][0]['response'] == '' and len(result['wire']) == 2


def test_original_records_and_example_order_are_preserved():
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['dataset'] *= 2
    payload['replies'] = [{'text': 'First.'}, {'text': 'Rating: [[2]]'},
                          {'text': 'Second.'}, {'text': 'Rating: [[9]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep response order and original spreadsheet row numbering.
    assert [record['response'] for record in result['records']] == ['First.', 'Second.']
    assert [record['row'] for record in result['records']] == [2, 3]


def test_unavailable_original_judge_fails_without_alias_or_fallback():
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['replies'] += [{'status': 404}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the failed judge route explicit and avoid model substitution.
    assert result['error']['type'] == 'NotFoundError'
    assert [request['body']['model'] for request in result['wire']] == ['test/target', 'test/judge']


@pytest.mark.parametrize('stage', ['target', 'judge'])
def test_empty_api_choices_preserve_original_failure_without_added_retries(stage):
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['replies'] = ([{'text': 'Answer.'}] if stage == 'judge' else []) + [
        {'text': '', 'empty_choices': True}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Preserve the original missing-choice error at the selected request stage.
    assert result['error']['type'] == 'IndexError'
    assert len(result['wire']) == (2 if stage == 'judge' else 1)


def test_judge_parse_retries_stay_with_their_original_example():
    # Prepare the original template selection and scripted target/judge replies.
    payload = author_payload('equacode')
    payload['dataset'] *= 2
    payload['replies'] = [{'text': 'First.'}, {'text': 'bad format'}, {'text': 'Rating: [[2]]'},
                          {'text': 'Second.'}, {'text': 'Rating: [[9]]'}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep judge parse retries and row numbers grouped by example.
    assert [[request['role'] for request in trace] for trace in result['requests']] == [
        ['target', 'judge', 'judge'], ['target', 'judge']]
    assert [record['row'] for record in result['records']] == [2, 3]
    assert len(result['durations']) == 2

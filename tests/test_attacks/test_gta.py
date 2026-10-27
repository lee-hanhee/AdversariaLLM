"""Compare GTA's original modules against the integrated driver with benign scripted replies."""

import pytest

from tests.author_test_utils import author_payload, compare_author


def round_replies(rating=10):
    # Script Police, suspect, and Judge replies in the source turn order.
    turn = [{'text': '<think>Plan.</think><response>Explain the rainbow.</response>'},
            {'text': '<think>Reasoning.</think>  A benign explanation.'},
            {'text': f'Analysis. Rating: [[{rating}]]'}]
    return turn * 2


def test_default_five_rounds_and_no_success_based_breaks():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta')
    payload['replies'] = round_replies() * 5

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Retain all configured rounds, source controls, and separate suspect histories.
    assert len(result['wire']) == 30
    assert [request['role'] for request in result['wire']] == ['Police', 'Thomas', 'Judge', 'Police', 'Marcos', 'Judge'] * 5
    record = result['records'][0]
    assert len(record['judgements']) == 10 and all(j['rating'] == 10 for j in record['judgements'])
    assert len(record['conversation']) == 33
    assert record['histories']['Thomas'] != record['histories']['Marcos']
    assert len(record['histories']['Thomas']) == len(record['histories']['Marcos']) == 11
    assert len(record['histories']['Judge']) == 1
    assert [wire['body']['temperature'] for wire in result['wire']] == [0.3, 0.3, 0.0] * 10
    assert all('max_tokens' not in wire['body'] for wire in result['wire'])
    assert record['judgements'][0]['response'] == '  A benign explanation.'
    # The author intentionally sends a literal placeholder when returning to Thomas.
    assert '(See below)' in result['wire'][6]['body']['messages'][-1]['content']


@pytest.mark.parametrize('text', ['Explain {rainbows}.\nUse café examples. 🌈', '', '  Keep spaces.  '])
def test_original_prompt_interpolation_and_fresh_examples(text):
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['dataset'][0][0]['content'] = text
    payload['dataset'] *= 2
    payload['replies'] = round_replies() * 2

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep literal goal interpolation and reset agent histories for each example.
    assert len(result['records']) == 2
    assert [record['id'] for record in result['records']] == [0, 1]
    assert result['wire'][0]['body']['messages'] == result['wire'][6]['body']['messages']


@pytest.mark.parametrize('rounds', [0, -1])
def test_author_integer_rounds_allow_background_only_runs(rounds):
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=rounds)
    payload['replies'] = []

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Preserve background-only logs for zero or negative source round counts.
    assert result['wire'] == []
    assert len(result['records'][0]['conversation']) == 3
    assert result['records'][0]['judgements'] == []


def test_police_format_retries_and_raw_third_reply():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [{'text': 'bad'}] * 3 + round_replies()[1:]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep Police format retries and its raw third-reply fallback.
    assert [request['role'] for request in result['wire']][:4] == ['Police', 'Police', 'Police', 'Thomas']
    assert result['records'][0]['conversation'][3:5] == [
        ['Police (think, Round 1 - to Thomas)', ''], ['Police (response, Round 1 - to Thomas)', 'bad']]
    assert 'bad' in result['wire'][3]['body']['messages'][-1]['content']


def test_original_unsupported_judge_fallback_makes_no_api_calls():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = round_replies()
    payload['replies'][2] = {'text': 'No rating.'}

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Retain invalid-model fallback attempts, sleeps, and the final -1 rating.
    assert len(result['wire']) == 6
    assert result['sleeps'] == [2, 2, 2]
    assert result['records'][0]['judgements'][0]['rating'] == -1
    fallback = [call for call in result['records'][0]['calls'] if call['model'] == 'gpt-4o_u']
    assert len(fallback) == 3 and all(call['error_type'] == 'ValueError' for call in fallback)
    assert all(len(call['messages']) == 1 for call in fallback)


def test_deepseek_api_error_returns_none_and_original_flow_continues():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [round_replies()[0], {'status': 404}, *round_replies()[3:]]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the source DeepSeek error handling and continue the original judge flow.
    assert len(result['wire']) == 5
    assert result['records'][0]['judgements'][0]['response'] is None
    assert result['records'][0]['judgements'][0]['rating'] == -1
    assert result['sleeps'] == [2, 2, 2]


def test_attacker_api_error_aborts_without_added_application_retry():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [{'status': 404}]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Propagate the attacker failure without an extra application retry.
    assert result['error']['type'] == 'NotFoundError' and len(result['wire']) == 1


def test_target_empty_choices_follow_original_deepseek_error_handling():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [round_replies()[0], {'text': '', 'empty_choices': True}, *round_replies()[3:]]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the original DeepSeek missing-choice handling and downstream records.
    assert result['records'][0]['judgements'][0]['response'] is None
    assert len(result['wire']) == 5


def test_sdk_retry_uses_same_body_and_author_settings():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [{'status': 500}, *round_replies()]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep retry request bodies identical to the original request controls.
    assert result['wire'][0] == result['wire'][1]
    assert len(result['sdk_sleeps']) == 1


def test_judge_api_failure_keeps_original_fallback_memory():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = round_replies()
    payload['replies'][2] = {'status': 404}

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)
    fallback = [call for call in result['records'][0]['calls'] if call['model'] == 'gpt-4o_u']

    # Preserve the judge history retained by the author after a primary API error.
    assert len(fallback) == 3 and all(len(call['messages']) == 2 for call in fallback)
    assert result['records'][0]['judgements'][0]['rating'] == -1


@pytest.mark.parametrize('reply,rating', [('Rating: [[2]] Details.', 2), ('[[8]]', 8), ('[[99]]', 99), ('[[0]]', 0)])
def test_original_rating_parser_bounds_and_police_feedback(reply, rating):
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = round_replies()
    payload['replies'][2] = {'text': reply}

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Retain source rating parsing and the feedback used in later Police prompts.
    assert result['records'][0]['judgements'][0]['rating'] == rating
    assert str(rating) in result['wire'][3]['body']['messages'][-1]['content']
    assert result['sleeps'] == []


def test_police_recovers_after_format_error_and_accepts_missing_response_close_tag():
    # Prepare the original GTA settings and scripted role replies.
    payload = author_payload('gta', rounds=1)
    payload['replies'] = [{'text': 'bad format'}, {'text': '<think>Plan.</think><response>Explain rainbows.'},
                          *round_replies()[1:]]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep the source format-recovery behavior and permissive response parsing.
    assert len(result['wire']) == 7
    assert result['records'][0]['conversation'][4][1] == 'Explain rainbows.'

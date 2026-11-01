"""Compare the original TAP tree search, legacy requests, and repository binding."""

import json

import pytest

from tests.author_test_utils import author_payload, compare_author


def attack(prompt='Explain rainbows.', improvement='Explain clearly.'):
    # Benign replies use the original two-field proposal format.
    return {'text': json.dumps({'improvement': improvement, 'prompt': prompt})}


def payload(**settings):
    # Keep fixtures small while the configuration retains the full author CLI defaults.
    value = author_payload('tap', **{'depth': 1, **settings})
    value['replies'] = [attack(), {'text': '[[YES]]'}, {'text': '  A benign answer.\n'}, {'text': '[[10]]'}]
    return value


def test_original_success_flow_requests_and_logger_records():
    # Compare actual old/new SDK serialization under the unchanged search and evaluator definitions.
    original, result = compare_author(payload())
    assert len(result['records'][0]['iterations']) == 1
    assert result['records'][0]['queries'][0]['response'] == '  A benign answer.\n'
    assert [request['role'] for request in result['wire']] == ['attacker', 'judge', 'target', 'judge']
    assert [request['body']['max_tokens'] for request in result['wire']] == [500, 10, 150, 10]
    assert [request['body']['temperature'] for request in result['wire']] == [1, 0, 0, 0]
    assert all(request['timeout'] == 20 for request in result['wire'])
    assert result['sleeps'] == []
    assert result['sdk_sleeps'] == original['sdk_sleeps'] == []


@pytest.mark.parametrize('attacker', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview'])
@pytest.mark.parametrize('target', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview'])
def test_original_source_profiles_and_fastchat_template_selection(attacker, target):
    # Source-name equality may share a GPT object; actual role routing must remain independent.
    value = payload()
    value['config']['attack_model'] = attacker
    value['model']['author_model'] = target
    _, result = compare_author(value)
    target_request = next(request for request in result['wire'] if request['role'] == 'target')
    system = target_request['body']['messages'][0]['content']
    if target == 'gpt-4-1106-preview':
        assert system.startswith('A chat between a curious human')
    else:
        assert system == 'You are a helpful assistant.'
    assert result['records'][0]['queries'][0]['score'] == 10


@pytest.mark.parametrize('profile', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview', 'gpt-4-turbo', 'no-evaluator'])
def test_original_evaluator_profiles_and_no_evaluator_ablation(profile):
    value = payload(evaluator_model=profile)
    if profile == 'no-evaluator':
        value['replies'] = [attack(), {'text': 'Answer.'}]
    _, result = compare_author(value)
    assert len(result['wire']) == (2 if profile == 'no-evaluator' else 4)
    assert result['records'][0]['queries'][0]['score'] == (1 if profile == 'no-evaluator' else 10)


@pytest.mark.parametrize('seed', [0, 1, 2026])
def test_equal_scores_use_source_index_ties_despite_random_shuffle(seed):
    # Phase one keeps C then B; phase two resolves equal ratings by reversed current indices.
    value = payload(branching_factor=3, width=2)
    value['rng_seed'] = seed
    value['replies'] = [attack('Explain A.'), attack('Explain B.'), attack('Explain C.'),
                        *[{'text': '[[YES]]'}] * 3, {'text': 'Answer C.'}, {'text': 'Answer B.'},
                        {'text': '[[5]]'}, {'text': '[[5]]'}]
    _, result = compare_author(value)
    record = result['records'][0]
    assert [row['messages'][-1]['content'] for row in record['queries']] == ['Explain C.', 'Explain B.']
    assert [row['prompt'] for row in record['iterations'][0]['attack_list']] == ['Explain B.', 'Explain C.']


def test_all_off_topic_fallback_retains_duplicate_candidates_even_beyond_width():
    # The original fallback duplicates a single rejected candidate rather than ending the frontier.
    value = payload(width=1)
    value['replies'] = [attack(), {'text': '[[NO]]'}, {'text': 'First answer.'},
                        {'text': 'Second answer.'}, {'text': '[[1]]'}, {'text': '[[10]]'}]
    _, result = compare_author(value)
    record = result['records'][0]
    assert len(record['queries']) == 2
    assert record['iterations'][0]['on_topic_scores'] == [0]
    assert record['iterations'][0]['judge_scores'] == [10]


@pytest.mark.parametrize('invalid', ['not a dictionary', '{"prompt": "Explain rainbows."}',
                                    '{"improvement": "plain", "prompt": "Explain {colors}."}'])
def test_original_malformed_attack_regeneration_and_first_closing_brace_parser(invalid):
    value = payload(max_n_attack_attempts=2)
    value['replies'].insert(0, {'text': invalid})
    _, result = compare_author(value)
    assert [request['role'] for request in result['wire']].count('attacker') == 2
    assert len(result['records'][0]['queries']) == 1


def test_all_failed_proposals_preserve_original_empty_cleanup_error():
    value = payload(max_n_attack_attempts=2)
    value['replies'] = [{'text': 'invalid'}] * 2
    _, result = compare_author(value)
    assert result['error']['type'] == 'IndexError'
    assert len(result['wire']) == 2


@pytest.mark.parametrize('proposal', ["{'improvement':'clear','prompt':'Explain rainbows.'}",
                                      '{"improvement": null, "prompt": "Explain rainbows."}',
                                      '{"improvement": "clear", "prompt": ""}'])
def test_original_literal_parser_and_key_presence_rules(proposal):
    value = payload(max_n_attack_attempts=1)
    value['replies'][0] = {'text': proposal}
    _, result = compare_author(value)
    if 'null' in proposal:
        assert result['error']['type'] == 'IndexError'
    else:
        assert len(result['records'][0]['queries']) == 1


@pytest.mark.parametrize('raw,score', [('[[99]]', 99), ('[[0]]', 0), ('[[10.0]]', 1),
                                     ('[[-1]]', 1), ('no score', 1)])
def test_source_rating_parser_has_no_range_validation(raw, score):
    value = payload()
    value['replies'][-1] = {'text': raw}
    _, result = compare_author(value)
    assert result['records'][0]['queries'][0]['score'] == score
    assert score in result['records'][0]['iterations'][0]['judge_scores']


def test_malformed_topic_labels_retain_source_permissive_fallback():
    value = payload()
    value['replies'][1] = {'text': 'unclear'}
    _, result = compare_author(value)
    assert result['records'][0]['iterations'][0]['on_topic_scores'] == [1]


@pytest.mark.parametrize('keep', [0, 1, 3])
def test_original_gpt_history_update_feedback_and_truncation(keep):
    value = payload(depth=3, keep_last_n=keep)
    value['replies'] = [attack(), {'text': '[[YES]]'}, {'text': 'Answer one.'}, {'text': '[[1]]'},
                        attack('Describe colors.'), {'text': '[[YES]]'}, {'text': 'Answer two.'}, {'text': '[[1]]'},
                        attack('Explain light.'), {'text': '[[YES]]'}, {'text': 'Answer three.'}, {'text': '[[10]]'}]
    _, result = compare_author(value)
    records = result['records'][0]['iterations']
    assert len(records) == 3
    assert records[1]['parent_conv_ids'] == records[0]['conv_ids']
    assert records[2]['parent_conv_ids'] == records[1]['conv_ids']
    attacker_messages = [request['body']['messages'] for request in result['wire'] if request['role'] == 'attacker']
    assert [message['role'] for message in attacker_messages[1]] == ['system', 'user', 'assistant']
    assert json.loads(attacker_messages[1][1]['content'])['prompt'] == 'Explain rainbows.'
    assert 'LANGUAGE MODEL OUTPUT: Answer one.' in attacker_messages[1][2]['content']


def test_exact_ten_stops_search_while_out_of_range_rating_does_not():
    value = payload(depth=2)
    value['replies'] = [attack(), {'text': '[[YES]]'}, {'text': 'Answer.'}, {'text': '[[99]]'},
                        attack(), {'text': '[[YES]]'}, {'text': 'Answer.'}, {'text': '[[10]]'}]
    _, result = compare_author(value)
    assert len(result['records'][0]['iterations']) == 2
    assert [row['score'] for row in result['records'][0]['queries']] == [99, 10]


def test_parallel_stream_chunks_keep_source_batch_order():
    value = payload(n_streams=6)
    value['replies'] = [attack(f'Explain color {index}.') for index in range(6)]
    value['replies'] += [{'text': '[[YES]]'}] * 6
    value['replies'] += [{'text': f'Answer {index}.'} for index in range(6)]
    value['replies'] += [{'text': '[[10]]'}] * 6
    _, result = compare_author(value)
    assert [request['role'] for request in result['wire']] == ['attacker'] * 6 + ['judge'] * 6 + ['target'] * 6 + ['judge'] * 6
    assert len(result['records'][0]['queries']) == 6


@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_original_http_error_retry_and_sleep_sequence(position):
    value = payload()
    value['replies'].insert(position, {'status': 500})
    _, result = compare_author(value)
    assert len(result['wire']) == 5
    assert result['sleeps'] == [10, 0.5]
    assert result['sdk_sleeps'] == []


def test_exhausted_http_errors_keep_twenty_attempts_per_proposal():
    value = payload(max_n_attack_attempts=2)
    value['replies'] = [{'status': 404}] * 40
    _, result = compare_author(value)
    assert result['error']['type'] == 'IndexError'
    assert len(result['wire']) == 40
    assert result['sleeps'] == [10, 0.5] * 40


def test_target_retry_exhaustion_keeps_original_error_marker_for_judging():
    value = payload()
    value['replies'] = [attack(), {'text': '[[YES]]'}, *[{'status': 500}] * 20, {'text': '[[10]]'}]
    _, result = compare_author(value)
    assert result['records'][0]['queries'][0]['response'] == '$ERROR$'
    assert result['sleeps'] == [10, 0.5] * 20


@pytest.mark.parametrize('target_none', [False, True])
def test_none_content_preserves_source_role_specific_handling(target_none):
    value = payload()
    value['replies'][2 if target_none else 0] = {'text': None}
    _, result = compare_author(value)
    if target_none:
        assert result['records'][0]['queries'][0]['response'] is None
    else:
        assert result['error']['type'] == 'AttributeError'


def test_empty_choices_follow_original_twenty_attempt_wrapper():
    value = payload(max_n_attack_attempts=1)
    value['replies'] = [{'text': 'unused', 'empty_choices': True}] * 20
    _, result = compare_author(value)
    assert result['error']['type'] == 'IndexError'
    assert result['sleeps'] == [10, 0.5] * 20


@pytest.mark.parametrize('setting,value', [('depth', 0), ('depth', -1), ('branching_factor', 0),
                                          ('n_streams', 0), ('width', 0), ('max_n_attack_attempts', 0)])
def test_nonpositive_cli_integers_reach_original_control_flow(setting, value):
    scenario = payload(**{setting: value})
    _, result = compare_author(scenario)
    if setting == 'depth':
        assert result['records'][0]['queries'] == [] and result['wire'] == []
    else:
        assert result['error']['type'] == 'IndexError'


def test_dataset_order_unicode_goal_reference_prefix_and_continuous_rng():
    value = payload()
    value['dataset'].append([{'role': 'user', 'content': 'Décris les couleurs.\nKeep {braces}.'},
                             {'role': 'assistant', 'content': 'Une réponse bénigne.'}])
    value['replies'] *= 2
    _, result = compare_author(value)
    assert len(result['records']) == 2
    assert 'Une réponse bénigne.' in result['records'][1]['system_prompt']
    assert result['records'][0]['iterations'][0]['conv_ids'] != result['records'][1]['iterations'][0]['conv_ids']


def test_empty_dataset_issues_no_requests():
    value = payload()
    value['dataset'], value['replies'] = [], []
    _, result = compare_author(value)
    assert result['records'] == [] and result['wire'] == []


@pytest.mark.parametrize('role,field', [(role, field) for role in ('target', 'attacker', 'judge')
                                        for field in ('id', 'provider')])
def test_explicit_routing_is_required_before_execution(role, field):
    from adversariallm.attacks.author_execution import validate_author_config

    value = payload()
    route = value['model'] if role == 'target' else value['config'][role]
    route[field] = None
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        validate_author_config('tap', value['model'], value['config'])


@pytest.mark.parametrize('change', ['backend', 'target_source', 'attacker_source', 'evaluator_source',
                                     'seed', 'generation', 'boolean_depth', 'float_tokens'])
def test_unsupported_profiles_and_overrides_are_rejected(change):
    from adversariallm.attacks.author_execution import validate_author_config

    value = payload()
    if change == 'backend':
        value['model']['backend'] = 'local'
    elif change == 'target_source':
        value['model']['author_model'] = 'gpt-4-turbo'
    elif change == 'attacker_source':
        value['config']['attack_model'] = 'vicuna'
    elif change == 'evaluator_source':
        value['config']['evaluator_model'] = 'replacement'
    elif change == 'seed':
        value['config']['seed'] = 0
    elif change == 'generation':
        value['config']['generation_config'] = {}
    elif change == 'boolean_depth':
        value['config']['depth'] = True
    elif change == 'float_tokens':
        value['config']['attack_max_n_tokens'] = 1.5
    with pytest.raises(ValueError):
        validate_author_config('tap', value['model'], value['config'])


def test_registry_configuration_and_original_cli_defaults():
    import ast
    from dataclasses import asdict
    from hydra import compose, initialize_config_dir
    from adversariallm.attacks import Attack
    from adversariallm.attacks.tap import TAPAttack, TAPConfig
    from adversariallm.defenses.registry import validate_hosted_target
    from tests.author_test_utils import ROOT

    # Preview the complete route configuration through the public registry and Hydra interface.
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=tap_author', 'attack=tap',
            'models.tap_author.id=test/target', 'models.tap_author.provider=target-provider',
            'models.tap_author.author_model=gpt-4', 'attacks.tap.attack_model=gpt-4',
            'attacks.tap.attacker.id=test/attacker', 'attacks.tap.attacker.provider=attacker-provider',
            'attacks.tap.judge.id=test/judge', 'attacks.tap.judge.provider=judge-provider'])
    assert Attack.from_name('tap') is TAPAttack
    validate_hosted_target(TAPAttack, cfg.models.tap_author, None, cfg.attacks.tap)
    assert set(asdict(TAPConfig())) == set(cfg.attacks.tap)

    # Compare defaults against literal declarations in the independent official main script.
    tree = ast.parse((ROOT / 'verification/tap/reference/main_TAP.py').read_bytes())
    defaults = {node.args[0].value: ast.literal_eval(keyword.value)
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                for keyword in node.keywords if keyword.arg == 'default'}
    config = TAPConfig()
    for field in ('attack_max_n_tokens', 'target_max_n_tokens', 'evaluator_max_n_tokens',
                  'evaluator_temperature', 'max_n_attack_attempts', 'branching_factor',
                  'width', 'depth', 'n_streams', 'keep_last_n', 'evaluator_model'):
        assert getattr(config, field) == defaults['--' + field.replace('_', '-')]
    assert defaults['--attack-model'] == defaults['--target-model'] == 'vicuna'
    assert config.attack_model is None


def test_public_entrypoint_uses_common_results_and_preserves_all_target_calls(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import httpx
    from openai import OpenAI
    from adversariallm.attacks.tap import TAPAttack
    from adversariallm.attacks.author_adapter import AuthorRunResult, AuthorStepResult
    from adversariallm.defenses.api_target import APITarget

    # Script an all-off-topic fallback with two targets but only one retained candidate.
    value = payload(width=1)
    replies = iter([attack()['text'], '[[NO]]', '  First answer.\n', 'Second answer.', '[[1]]', '[[10]]'])
    wire = []
    def http(request):
        wire.append(json.loads(request.content))
        assert request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': wire[-1]['model'], 'choices': [{'index': 0, 'message': {
                'role': 'assistant', 'content': next(replies)}, 'finish_reason': 'stop'}]}, request=request)
    client = OpenAI(api_key='test-key', base_url='https://openrouter.ai/api/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(http)))
    target = object.__new__(APITarget)
    target.config, target.text_generator = value['model'], SimpleNamespace(client=client)
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(key, raising=False)

    # The SDK view must leave the supplied client's retry settings intact.
    result = TAPAttack(value['config']).run(target, value['dataset'])
    run = result.runs[0]
    assert isinstance(run, AuthorRunResult) and len(run.steps) == 2
    assert all(isinstance(step, AuthorStepResult) for step in run.steps)
    assert [step.scores for step in run.steps] == [{'tap_author': {'rating': [1.0]}}, {'tap_author': {'rating': [10.0]}}]
    assert run.steps[0].model_completions == run.steps[0].model_completions_raw == ['  First answer.\n']
    assert len(run.author_result['iterations'][0]['attack_list']) == 1
    assert len(run.author_requests) == 6 and client.max_retries == 2
    assert len(asdict(result)['runs'][0]['steps']) == 2


def test_missing_isolated_fastchat_has_an_explicit_setup_error(monkeypatch, tmp_path):
    from adversariallm.attacks.utils.tap import binding

    # Report the pinned dependency setup instead of substituting an installed newer template.
    binding.conversation_factory.cache_clear()
    monkeypatch.setattr(binding, '__file__', str(tmp_path / 'adversariallm/attacks/utils/tap/binding.py'))
    with pytest.raises(ImportError, match='fschat==0.2.32'):
        binding.conversation_factory()
    binding.conversation_factory.cache_clear()

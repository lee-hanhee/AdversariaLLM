"""Bind the preserved gta driver to repository I/O."""

import copy
import time
from types import SimpleNamespace

from ...author_execution import load_symbols


def execute(transport, model, config, questions, sleep):
    import re

    # Load original role backgrounds and the author judge prompt.
    prompts = load_symbols('gta/gta_prompts.py', ['model_a_background_base', 'model_b_background_base',
                                               'model_c_background_base', 'judge_system_prompt'], {})

    # Share the suspect target route while keeping attacker and judge routes explicit.
    transport.routes.update(Police=config['attacker'], Thomas=transport.routes['target'],
                            Marcos=transport.routes['target'], Judge=config['judge'])
    clock = SimpleNamespace(sleep=sleep or time.sleep)

    # Start fresh agent state and capture buffers for each dataset example.
    records = []
    for goal in transport.examples(questions):
        agents, calls, judgements = {}, [], []

        def agent_class(role, filename, name):
            # Bind the unchanged models.chat per role, including DeepSeek processing and invalid fallback.
            sdk = transport.for_role(role)
            original_chat = load_symbols('gta/gta_models.py', ['chat'], {
                'client_gpt': sdk, 'client_azure': sdk,
            })['chat']

            def chat(model, messages, temperature=0.1):
                # Record each logical chat call, including source branches that fail before an API request.
                call = {'role': role, 'model': model, 'messages': copy.deepcopy(messages), 'temperature': temperature}
                calls.append(call)

                # Let the original chat function control response handling and failures.
                try:
                    call['response'] = original_chat(model, messages, temperature)
                    return call['response']
                except Exception as error:
                    call['error_type'] = type(error).__name__
                    raise

            # Build each agent with its own role-bound chat function and message history.
            base = load_symbols('gta/base_agent.py', ['BaseAgent'], {'chat': chat})['BaseAgent']
            return load_symbols('gta/' + filename, [name], {
                'BaseAgent': base, 're': re, 'time': clock, 'chat': chat,
            })[name]

        # Keep the actual author agent instances so their final histories can be recorded.
        def capture_agent(cls, **kwargs):
            instance = cls(**kwargs)
            agents[kwargs['role_name']] = instance
            return instance

        # Create separate agent classes for each role, even when targets share a route.
        police = agent_class('Police', 'attack_agent.py', 'AttackAgent')
        thomas = agent_class('Thomas', 'target_agent.py', 'TargetAgent')
        marcos = agent_class('Marcos', 'target_agent.py', 'TargetAgent')
        judge = agent_class('Judge', 'judge_agent.py', 'JudgeAgent')

        # Observe returned ratings and request boundaries while keeping author judging intact.
        class RecordingJudge(judge):
            def judge(self, response):
                # Capture the author's returned rating; never parse or recompute it here.
                rating, analysis = super().judge(response)
                judgements.append({'role': 'Thomas' if len(judgements) % 2 == 0 else 'Marcos',
                                   'round': len(judgements) // 2 + 1, 'response': response,
                                   'rating': rating, 'analysis': analysis,
                                   'request_end': len(transport.requests[transport.index])})
                return rating, analysis

        # Capture the original conversation and final histories instead of writing author files.
        def save(conversation, num, output_dir):
            records.append({'id': num, 'conversation': copy.deepcopy(conversation),
                            'judgements': judgements, 'calls': calls,
                            'histories': {role: copy.deepcopy(agent.message_memory) for role, agent in agents.items()}})

        # Bind filesystem callbacks and agent construction into the unchanged author orchestrator.
        driver = load_symbols('gta/gta_driver.py', ['ours_attack'], {
            **prompts, 'os': SimpleNamespace(makedirs=lambda *args, **kwargs: None), 'save_conversation': save,
            'AttackAgent': lambda **kwargs: capture_agent(police, **kwargs),
            'TargetAgent': lambda **kwargs: capture_agent(thomas if kwargs['role_name'] == 'Thomas' else marcos, **kwargs),
            'JudgeAgent': lambda **kwargs: capture_agent(RecordingJudge, **kwargs),
        })['ours_attack']

        # Execute the published source profiles with the configured number of rounds.
        driver(rounds=config['rounds'], prompt=goal, num=transport.index,
               target_model_1='deepseek_r1', target_model_2='deepseek_r1', attack_model='gpt-4o-mini', output_dir='')

    return records, {}

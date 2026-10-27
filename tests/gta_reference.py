"""Independent GTA reference execution through original modules and native SDK serializers."""

import ast
import copy
import sys
from types import ModuleType, SimpleNamespace


def gta_role(messages):
    system = messages[0]['content'].lstrip()
    if system.startswith('# Role: Thomas Carviel'):
        return 'Thomas'
    if system.startswith('# Role: Marcos Andriel'):
        return 'Marcos'
    if system.startswith('# Role: Interrogation Officer'):
        return 'Police'
    return 'Judge'


def run_reference_gta(payload, factory, sleep, base):
    # Import complete original agent/driver files. Only transport and storage imports are bound.
    def module(name, filename=None, **bindings):
        result = ModuleType(name)
        result.__package__ = name.rpartition('.')[0]
        result.__dict__.update(bindings)
        sys.modules[name] = result
        if filename:
            exec(compile((base / filename).read_bytes(), str(base / filename), 'exec'), result.__dict__)
        return result

    requests, records, calls = [], [], []
    # Match the author's shared SDK clients while keeping role-specific request recording.
    sdk_clients = {
        'openai': factory('openai', {'api_key': 'test-key'}),
        'azure': factory('azure', {'api_key': 'test-key', 'azure_endpoint': 'https://author-azure.example',
                                   'api_version': '2025-01-01-preview'}),
    }
    native = {}
    for role in ('Police', 'Thomas', 'Marcos', 'Judge'):
        clients = {}
        for provider, client in sdk_clients.items():
            def create(_role=role, _client=client, **kwargs):
                requests.append({'role': _role, 'request': copy.deepcopy(kwargs)})
                return _client.chat.completions.create(**kwargs)

            clients[provider] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        namespace = {'client_gpt': clients['openai'], 'client_azure': clients['azure']}
        tree = ast.parse((base / 'models.py').read_bytes())
        chat = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'chat')
        exec(compile(ast.Module(body=[chat], type_ignores=[]), str(base / 'models.py'), 'exec'), namespace)
        native[role] = namespace['chat']

    def chat(model, messages, temperature=0.1):
        role = gta_role(messages)
        event = {'role': role, 'model': model, 'messages': copy.deepcopy(messages), 'temperature': temperature}
        calls.append(event)
        try:
            event['response'] = native[role](model, messages, temperature)
            return event['response']
        except Exception as error:
            event['error_type'] = type(error).__name__
            raise

    module('models', chat=chat)
    module('prompts', 'prompts.py')
    package = module('agent')
    package.__path__ = []
    module('agent.base_agent', 'agent/base_agent.py')
    attack = module('agent.attack_agent', 'agent/attack_agent.py').AttackAgent
    target = module('agent.target_agent', 'agent/target_agent.py').TargetAgent
    judge_module = module('agent.judge_agent', 'agent/judge_agent.py')
    judge_module.time = SimpleNamespace(sleep=sleep)
    judge = judge_module.JudgeAgent

    for index, conversation in enumerate(payload['dataset']):
        agents, judgements = {}, []
        calls = []
        request_begin = len(requests)

        def capture(cls, **kwargs):
            instance = cls(**kwargs)
            agents[kwargs['role_name']] = instance
            return instance

        class Judge(judge):
            def judge(self, response):
                rating, analysis = super().judge(response)
                judgements.append({'role': 'Thomas' if len(judgements) % 2 == 0 else 'Marcos',
                                   'round': len(judgements) // 2 + 1, 'response': response,
                                   'rating': rating, 'analysis': analysis,
                                   'request_end': len(requests) - request_begin})
                return rating, analysis

        def save(log, num, output_dir):
            records.append({'id': num, 'conversation': copy.deepcopy(log), 'judgements': judgements,
                            'calls': calls, 'histories': {
                                name: copy.deepcopy(agent.message_memory) for name, agent in agents.items()}})

        module('helpers', save_conversation=save, load_harmful_behaviors=lambda: [])
        package.AttackAgent = lambda **kwargs: capture(attack, **kwargs)
        package.TargetAgent = lambda **kwargs: capture(target, **kwargs)
        package.JudgeAgent = lambda **kwargs: capture(Judge, **kwargs)
        driver = module('gta_original_driver', 'utils.py')
        driver.os = SimpleNamespace(makedirs=lambda *args, **kwargs: None)
        driver.ours_attack(rounds=payload['config']['rounds'], prompt=conversation[0]['content'], num=index,
                           target_model_1='deepseek_r1', target_model_2='deepseek_r1',
                           attack_model='gpt-4o-mini', output_dir='')
    return records

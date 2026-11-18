"""Preserved author notebook cells 11 (zero-based).

Source: https://github.com/EVIGBYEN/Mousetrap
Revision: 3906547a9d18fe0e54d27bb541676162cbdbe31c
MIT License; Copyright (c) 2025 YAO YANG.
License, original notebook, and extraction diff: verification/mousetrap/.
"""

input_file_path = 'INPUT_PATH.JSONL'
output_file_path = f'OUTPUT_PATH.JSONL'
output_log_path = f'LOG_PATH.JSONL'
params = random.sample([1, 2, 3], 3)

data = []
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        data.append(json_obj)

for i in range(len(data)):
    question = data[i]["ptq"]
    dim = data[i]["dim"]
    succ = 0

    for iter in [1, 2, 3]:
        arus = []
        for j in range(3):
            aru, ats, question, response, ecps, dcps, ctq, prompt = pipeline_without_checker(
                question, iter, params)
            arus.append(aru)
            item = {
                "aru": aru,
                "ats": ats,
                "ptq": question,
                "response": response,
                "ecps": ecps,
                "dcps": dcps,
                "ctq": ctq,
                "prompt": prompt,
                "dim": dim
            }
            with open(output_file_path, 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(item) + '\n')

        if arus.count(1) == 3:
            succ = 1
            log = {"iter": iter}
            with open(output_log_path, 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(log) + '\n')
            break

    if succ == 0:
        log = {"iter": 0}
        with open(output_log_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(log) + '\n')
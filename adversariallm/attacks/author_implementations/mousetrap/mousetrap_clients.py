"""Preserved author notebook cells 2, 3 (zero-based).

Source: https://github.com/EVIGBYEN/Mousetrap
Revision: 3906547a9d18fe0e54d27bb541676162cbdbe31c
MIT License; Copyright (c) 2025 YAO YANG.
License, original notebook, and extraction diff: verification/mousetrap/.
"""

class Judger:
    def __init__(self):
        self.client = openai.OpenAI(api_key="YOUR_SECRET_KEY_HERE")
    def __call__(self, sprompt, uprompt):
        messages = [{"role": "system", "content": sprompt}]
        messages.append({"role": "user", "content": uprompt})
        completion = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20", messages=messages)
        response = completion.choices[0].message.content
        return response
judger = Judger()

class CheckerTarget_OpenaAI:
    def __init__(self):
        self.client = openai.OpenAI(api_key="YOUR_OPENAI_SECRET_KEY_HERE")
    def __call__(self, uprompt):
        messages = [{"role": "user", "content": uprompt}]
        completion = self.client.chat.completions.create(model="o1-2024-12-17", messages=messages)
        response = completion.choices[0].message.content
        return response


class CheckerTarget_Google:
    def __init__(self):
        genai.configure(api_key="YOUR_GOOGLE_SECRET_KEY_HERE")
    def __call__(self, uprompt):
        model = genai.GenerativeModel(model_name='gemini-2.5-pro-exp-03-25')
        completion = model.generate_content(
            [uprompt],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                #HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            })
        response = completion.text
        return response


class CheckerTarget_Anthropic:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_SECRET_KEY_HERE")
    def __call__(self, uprompt):
        message = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2048,
            messages=[{"role": "user", "content": uprompt}])
        response = message.content[0].text
        return response


checker = CheckerTarget_XXX()
target = CheckerTarget_XXX()
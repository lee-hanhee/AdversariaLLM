"""Preserved QueryAttack author source.

Source: https://github.com/horizonsinzqs/QueryAttack
Revision: 93a19806cf0e21a0b28b6fee18a895d57f170bb6
Verification and extraction diff: verification/query_attack/.

MIT License

Copyright (c) 2025 horizonsinzqs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from openai import OpenAI, PermissionDeniedError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError, InternalServerError
import time

# API configurations
API_CONFIGS = {
    "deepseek": {
        "api_key": "sk-xxx",
        "base_url": "",
        "model": "deepseek-chat"
    },
    "gpt-3.5-turbo": {
        "api_key": "sk-xxx",
        "base_url": "",
        "model": "gpt-3.5-turbo"
    },
    "gpt-4-1106-preview": {
        "api_key": "sk-xxx",
        "base_url": "",
        "model": "gpt-4-1106-preview"
    },
    "gpt-4o": {
        "api_key": "sk-xxx",
        "base_url": "",
        "model": "gpt-4o"
    },
}

# Default model
LLM_MODEL = "gpt-4-1106-preview"

# Get the configuration for the selected model
config = API_CONFIGS.get(LLM_MODEL)
if not config:
    raise ValueError(f"Invalid LLM model: {LLM_MODEL}")

your_api_key = config["api_key"]
your_base_url = config["base_url"]
your_model = config["model"]
sleep_time = config.get("sleep_time", 3)

def ask_response(system_content, user_content):
    client = OpenAI(api_key=your_api_key, base_url=your_base_url)
    response = None
    while response is None:
        response = client.chat.completions.create(
            model=your_model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            stream=False
        )
    print("response:", response.choices[0].message.content)
    return response

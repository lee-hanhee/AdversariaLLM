"""Author FlipAttack execution source, preserved unchanged.

Exact source, license, and extraction diff: verification/flip_attack/.

Source: https://github.com/yueliu1999/FlipAttack
Revision: c53880b4a6e8ab47686849838b3d2e11bc80a6b9

MIT License

Copyright (c) 2026 yueliu1999

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

import os
import time
import anthropic
from anthropic import Anthropic
from openai import OpenAI

class LLM():
    def __init__(self, model_id="gpt-3.5-turbo", temperature=0, max_token=-1, retry_time=1000, failed_sleep_time=1, round_sleep_time=1) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_token = max_token
        self.retry_time = retry_time
        self.failed_sleep_time = failed_sleep_time
        self.round_sleep_time = round_sleep_time
        
    # for GPT
    def GPTchatCompletion(self, messages, base_url=None):
        if base_url is None:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
                )
        else:
            client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=base_url
        )
        try:
            response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature
                )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model_id} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    response = client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=self.temperature
                    )
                    break
                except:
                    continue

        llm_output = response.choices[0].message.content.strip()
        time.sleep(self.round_sleep_time)

        return llm_output

    # for Claude
    def ClaudeCompletion(self, messages, base_url=None):
        if base_url is None:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        else:
            client = Anthropic(
                base_url=base_url,
                auth_token=os.environ["ANTHROPIC_API_KEY"]
                )   
        try:
            message = client.messages.create(
                model=self.model_id,
                max_tokens=3584,
                temperature=self.temperature,
                system=messages[0]['content'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text":  messages[1]['content']
                            }
                        ]
                    },

                ]
            )

        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model_id} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = client.messages.create(
                        model=self.model_id,
                        max_tokens=3584,
                        temperature=self.temperature,
                        system=messages[0]['content'],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text":  messages[1]['content']
                                    }
                                ]
                            }
                        ]
                    )
                    break
                except:
                    continue

        model_output = message.content[0].text
        time.sleep(self.round_sleep_time)

        return model_output
    
    # LLaMA
    def LLaMAchatCompletion(self, messages, base_url="https://api.deepinfra.com/v1/openai"):
        if base_url is None:
            client = OpenAI(
                api_key=os.environ["DEEPINFRA_API_KEY"]
                )
        else:
            client = OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url=base_url
        )
        try:
            response = client.chat.completions.create(
                    model="meta-llama/"+self.model_id,
                    messages=messages,
                    temperature=self.temperature
                )
        except Exception as e:
            print(e)
            # out of token number
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"meta-llama/{self.model_id} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/"+self.model_id,
                        messages=messages,
                        temperature=self.temperature
                    )
                    break
                except:
                    continue
                
        model_output = response.choices[0].message.content.strip()
        time.sleep(self.round_sleep_time)

        return model_output
    
    # Mistarl
    def MistarlchatCompletion(self, messages, base_url="https://api.deepinfra.com/v1/openai"):
        if base_url is None:
            client = OpenAI(
                api_key=os.environ["DEEPINFRA_API_KEY"]
                )
        else:
            client = OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url=base_url
        )
        try:
            response = client.chat.completions.create(
                    model="mistralai/"+self.model_id,
                    messages=messages,
                    temperature=self.temperature
                )
        except Exception as e:
            print(e)
            # out of token number
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"mistralai/{self.model_id} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    response = client.chat.completions.create(
                        model="mistralai/"+self.model_id,
                        messages=messages,
                        temperature=self.temperature
                    )
                    break
                except:
                    continue
                
        model_output = response.choices[0].message.content.strip()
        time.sleep(self.round_sleep_time)

        return model_output

    def response(self, messages):
        if "gpt" in self.model_id: 
            llm_output = self.GPTchatCompletion(messages)
        elif "claude" in self.model_id:
            llm_output = self.ClaudeCompletion(messages)
        elif "Llama" in self.model_id:
            llm_output = self.LLaMAchatCompletion(messages)
        elif "Mixtral" in self.model_id:
            llm_output = self.MistarlchatCompletion(messages)
        else:
            raise NameError("Unknown Chatbot {}, please add the implementation.".format(self.model_id))

        return llm_output

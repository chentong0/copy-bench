from .lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel:
    def __init__(self, model_name, cache_file=None, model_mode="chat", api_type="openai", save_interval=100):
        self.model_name = model_name
        self.model_mode = model_mode
        self.api_type = api_type
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load_model(self):
        # load api key
        if self.api_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.api_type == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_version="2023-07-01-preview",
                azure_endpoint=os.environ["M_OPENAI_API_BASE"],
                api_key=os.environ["M_OPENAI_API_KEY"],
            )

    def generate(self, prompt_or_message, max_new_tokens=100, temperature=0.0, top_p=1.0, return_response=False):
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is

        if self.model_mode == "chat":
            # Construct the prompt send to ChatGPT
            if isinstance(prompt_or_message, str):
                prompt_or_message = [
                    {"role": "user", "content": prompt_or_message}
                ]
            assert isinstance(prompt_or_message, list), "Prompt must be a list of strings"
        elif self.model_mode == "instruct":
            assert isinstance(prompt_or_message, str), "Prompt must be a string"
        else:
            raise NotImplementedError()
        
        output, response = call_gpt(
            self.client, self.model_mode, 
            self.model_name, prompt_or_message, 
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            return_response=True,
        )
        assert response is not None, "Response is None"
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        
        if return_response:
            return output, response
        else:
            return output

def call_gpt(client, model_mode, model_name, prompt_or_message, max_tokens=100, temperature=0.0, top_p=1.0, return_response=False):
    output, response = None, None
    max_num_tries = 5
    for n_tries in range(max_num_tries):
        try:
            if isinstance(prompt_or_message, list):
                assert model_mode == "chat", "Messages list must be used in chat mode"
                configure = {"model": model_name, "messages": prompt_or_message, 
                    "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
                response = client.chat.completions.create(**configure)
                # output = response["choices"][0]["message"]["content"]
                output = response.choices[0].message.content
            elif isinstance(prompt_or_message, str):
                assert model_mode == "instruct", "Prompt must be a string"
                configure = {"model": model_name, "prompt": prompt_or_message, 
                    "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
                response = client.completions.create(**configure)
                output = response.choices[0].text
            break
        except Exception as e:
            
            if "response was filtered" in str(e):
                raise ValueError(f"Response was filtered: {e}")

            import re
            pattern = re.compile(r"after (\d+) second")
            match = pattern.search(str(e))
            wait_time = int(match.group(1)) if match else 2 ** n_tries
            logging.error(f"API error: {e} ({n_tries}). Waiting {wait_time} sec")
            time.sleep(wait_time)
    return output, response

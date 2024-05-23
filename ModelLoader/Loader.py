from transformers import AutoTokenizer, AutoModelForCausalLM
from Tools import PrintWithBorders

import openai
import anthropic
import os

class APIModel():
    def __init__(self, model, model_info, model_args):
        self.model = model
        self.model_name = model_info["Model_Name"]
        self.max_tokens = model_args["max_length"]
        self.temperature = model_args["temperature"]
        self.do_sample = model_args["do_sample"]
        self.token_used = 0

        
    def Query(self, content):
        if str(type(self.model)) == "<class 'openai.OpenAI'>":
            response = self.model.chat.completions.create(
                model=self.model_name.lower(),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": content},
                ]
            )
            self.used_token += response.usage.total_tokens
            return response.choices[0].message.content
        elif str(type(self.model)) == "<class 'anthropic.Anthropic'>":
            response = self.model.messages.create(
                model=self.model_name.lower(),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            self.used_token += response.usage.input_tokens
            self.used_token += response.usage.output_tokens
            return response.content[0].text
        else:
            raise ValueError("Unrecognized client")
    
    
    def PrintTokenUsed(self):
        info = self.model_name + " used " + str(self.token_used) + " tokens."
        PrintWithBorders(info)
        


class CasualModel():
    def __init__(self, model, tokenizer, model_info, model_args):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_info["Model_Name"]
        self.max_tokens = model_args["max_length"]
        self.temperature = model_args["temperature"]
        self.do_sample = model_args["do_sample"]
        
        
    def Query(self, content):
        pass


def LoadTokenizerAndModel(
        model_info,
        model_args,
        device
    ):
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_info["Tokenizer_Path"],
        trust_remote_code=True,
        local_files_only=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_info["Model_Path"],
        trust_remote_code=True,
        local_files_only=True,
        max_length=model_args["max_length"],
        temperature=model_args["temperature"],
        do_sample=model_args["do_sample"],
    ).to(device)
    
    return CasualModel(model, tokenizer, model_info, model_args)
    
    
def LoadAPIModel(
        model_info,
        model_args
    ):
    if model_info["Producer"] == "OpenAI":
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("Env. variable OPENAI_API_KEY is not given")
    
        if model_info["ProxyOn"]:
            openai_client = openai.OpenAI(
                base_url=model_info["ProxyUrl"],
                api_key=api_key
            )
        else:
            openai_client = openai.OpenAI(
                api_key=api_key
            )
        
        return APIModel(openai_client, model_info, model_args)
    
    elif model_info["Producer"] == "Anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key is None:
            raise ValueError("Env. variable ANTHROPIC_API_KEY is not given")
        
        if model_info["ProxyOn"]:
            anthropic_client = anthropic.Anthropic(
                base_url=model_info["ProxyUrl"],
                api_key=api_key
            )
        else:
            anthropic_client = anthropic.Anthropic(
                api_key=api_key
            )
        
        return APIModel(anthropic_client, model_info, model_args)
        
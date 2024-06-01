from transformers import AutoTokenizer, AutoModelForCausalLM
from Tools import PrintWithBorders

import math
import openai
import anthropic
import os
import torch
from fastchat.model import get_conversation_template
from rich.console import Console
from rich.panel import Panel

console = Console()


class APIModel():
    def __init__(self, model, model_info, model_args):
        self.model = model
        self.model_name = model_info["Model_Name"]
        self.max_tokens = model_args["max_length"]
        self.temperature = model_args["temperature"]
        self.do_sample = model_args["do_sample"]
        self.token_used = 0

        
    def Generate(self, content, prompt="You are a helpful AI assistant.", max_tokens=128):
        if str(type(self.model)) == "<class 'openai.OpenAI'>":
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name.lower(),
                    max_tokens=max(self.max_tokens, max_tokens),
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content},
                    ]
                )
                self.token_used += response.usage.total_tokens
                return response.choices[0].message.content
            except Exception as e:
                title, text = "Loader Info", f"OpenAI API call failed due to {e}."
                panel = Panel(text, title=title, border_style="red")
                console.print(panel)
        elif str(type(self.model)) == "<class 'anthropic.Anthropic'>":
            try:
                response = self.model.messages.create(
                    model=self.model_name.lower(),
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ]
                )
                self.token_used += response.usage.input_tokens
                self.token_used += response.usage.output_tokens
                return response.content[0].text
            except Exception as e:
                title, text = "Loader Info", f"Claude API call failed due to {e}."
                panel = Panel(text, title=title, border_style="red")
                console.print(panel)
        else:
            raise ValueError("Unrecognized client")
    
    
    def PrintTokenUsed(self):
        info = self.model_name + " used " + str(self.token_used) + " tokens."
        PrintWithBorders(info)
        


class CasualModel():
    def __init__(self, model, tokenizer, model_info, model_args, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_info["Model_Name"]
        self.max_tokens = model_args["max_length"]
        self.temperature = model_args["temperature"]
        self.do_sample = model_args["do_sample"]
        self.template_name = None
        self.device = device
        
        if "llama-2" in self.model_name.lower():
            self.template_name = "llama-2"
        elif "baichuan2" in self.model_name.lower():
            self.template_name = "bachuan2"
        elif "vicuna" in self.model_name.lower():
            self.template_name = "vicuna"
        else:
            pass
        
    """
        This function is used for a single query
    """    
    def Generate(self, content, max_tokens=128):
        inputs = self.tokenizer(content, return_tensors='pt').to(self.device)
        
        if self.temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(self.max_tokens, max_tokens), 
                do_sample=self.do_sample,
                temperature=self.temperature,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(self.max_tokens, max_tokens), 
                do_sample=False,
                temperature=1,
            )
            
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        del inputs, output_ids
        torch.cuda.empty_cache()
        
        return output


    def BatchGenerate(self, content_list):
        batchsize = len(content_list)
        convs_list = [self.ConvTemplate() for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, content_list):
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None) 
            full_prompts.append(conv.get_prompt())
        
        outputs_list = [self.Generate(single_prompt) for single_prompt in full_prompts]
        return outputs_list
    
    
    def ConvTemplate(self):
        template = get_conversation_template(self.template_name)
        if template.name == 'llama-2':
            template.sep2 = template.sep2.strip()
        return template


def LoadTokenizerAndModel(
        model_info,
        model_args,
        device
    ):
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_info["Tokenizer_Path"],
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_info["Model_Path"],
        trust_remote_code=True,
        local_files_only=True,
        max_length=model_args["max_length"],
        temperature=model_args["temperature"],
        do_sample=model_args["do_sample"],
        torch_dtype=torch.float16,
    ).to(device)
    
    return CasualModel(model, tokenizer, model_info, model_args, device)
    
    
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
        
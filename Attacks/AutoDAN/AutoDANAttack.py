import gc
import numpy as np
import torch
import torch.nn as nn
from .utils.opt_utils import get_score_autodan, autodan_sample_control
from .utils.opt_utils import autodan_sample_control_hga
from .utils.string_utils import autodan_SuffixManager
from tqdm import tqdm
import random
from Attacks.AutoDAN import Config
import nltk

early_stopping = False

seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id,
                                top_p=0.9,
                                do_sample=True,
                                temperature=0.7
                                )[0]
    return output_ids[assistant_role_slice.stop:]

def check_for_attack_success(target_model, judge_model, content, content_id, prompt, input_ids, assistant_role_slice, gen_config=None):
    response = target_model.tokenizer.decode(
        generate(
            target_model.model,
            target_model.tokenizer,
            input_ids,
            assistant_role_slice,
            gen_config=gen_config
        )
    ).strip()
    
    res = judge_model.Judge(content, response, content_id, prompt)
    
    return res


def log_init():
    log_dict = {"loss": [], "suffix": [],
                "time": [], "respond": [], "success": []}
    return log_dict


def get_developer(model_name):
    developer_dict = {
        "LLaMA-2-7B-chat-hf": "Meta", 
        "ChatGLM3-6B": "THUDM",
        "Gemma-2B": "Google",
        "Baichuan2-7B-Chat": "Baichuan-inc",
        "LLaMA-3-8B-Instruct": "Meta",
    }
    return developer_dict[model_name]


def AutoDANAttack(target_model, mutator_model, judge_model, data, logger, language="English"):
    packages = ["stopwords", "punkt", "wordnet"]
    for package in packages:
        nltk.download(package)
    
    logger.Set("AutoDAN", target_model.model_name)
    json_data = data["json"]
    print(json_data)

    num_steps = Config.num_steps
    batch_size = Config.batch_size
    num_elites = max(1, int(Config.batch_size * Config.num_elites))
    crossover = Config.crossover
    num_points = Config.num_points
    mutation = Config.mutation
    
    conv_template = target_model.ConvTemplate()

    crit = nn.CrossEntropyLoss(reduction='mean')

    prefix_string_init = None
    for i, item in tqdm(enumerate(json_data)):
        reference = torch.load('./Attacks/AutoDAN/assets/prompt_group.pth', map_location='cpu')
        
        question = item[language]
        goal = item["Goal"]

        log = log_init()
        info = {"question": "", "goal": "", "final_suffix": "",
                "final_respond": "", "total_time": 0, "is_success": False, "log": log}
        info["question"] = info["question"].join(question)
        info["goal"] = info["goal"].join(goal)

        user_prompt = question
        target = goal
        for o in range(len(reference)):
            reference[o] = reference[o].replace('[MODEL]', target_model.model_name.title())
            reference[o] = reference[o].replace('[KEEPER]', target_model.model_name)
        new_adv_suffixs = reference[:batch_size]
        word_dict = {}
        last_loss = 1e-5
        for j in range(num_steps):
            with torch.no_grad():
                losses = get_score_autodan(
                    tokenizer=target_model.tokenizer,
                    conv_template=conv_template, instruction=user_prompt, target=target,
                    model=target_model.model,
                    device=target_model.device,
                    test_controls=new_adv_suffixs,
                    crit=crit)
                score_list = losses.cpu().numpy().tolist()

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffixs[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                if isinstance(prefix_string_init, str):
                    best_new_adv_suffix = prefix_string_init + best_new_adv_suffix
                adv_suffix = best_new_adv_suffix

                suffix_manager = autodan_SuffixManager(
                    tokenizer=target_model.tokenizer,
                    conv_template=conv_template,
                    instruction=user_prompt,
                    target=target,
                    adv_string=adv_suffix
                )
                
                input_ids, prompt = suffix_manager.get_input_ids(adv_string=adv_suffix)
                input_ids = input_ids.to(target_model.device)
                
                judge_result = check_for_attack_success(
                    target_model,
                    judge_model,
                    question,
                    item["id"],
                    prompt,
                    input_ids,
                    suffix_manager._assistant_role_slice
                )
                
                logger.Log(judge_result)
                
                # Early stopping
                is_success = judge_result["Judge_Result"] == 1 
                if is_success and early_stopping:
                    break
                
                if j % Config.iter == 0:
                    unfiltered_new_adv_suffixs = autodan_sample_control(
                        control_suffixs=new_adv_suffixs,
                        score_list=score_list,
                        num_elites=num_elites,
                        batch_size=batch_size,
                        crossover=crossover,
                        num_points=num_points,
                        mutation=mutation,
                        openai_client=mutator_model,
                        reference=reference
                    )
                else:
                    unfiltered_new_adv_suffixs, word_dict = autodan_sample_control_hga(
                        word_dict=word_dict,
                        control_suffixs=new_adv_suffixs,
                        score_list=score_list,
                        num_elites=num_elites,
                        batch_size=batch_size,
                        crossover=crossover,
                        mutation=mutation,
                        openai_client=mutator_model,
                        reference=reference
                    )

                new_adv_suffixs = unfiltered_new_adv_suffixs

                print(
                    "################################\n"
                    f"Current Epoch: {j + 1}/{num_steps}\n"
                    f"Passed:{is_success}\n"
                    f"Loss:{current_loss.item()}\n"
                    "################################\n")

                last_loss = current_loss.item()
                gc.collect()
                torch.cuda.empty_cache()

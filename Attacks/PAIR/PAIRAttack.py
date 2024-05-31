from .Prompts import GetAttackerSystemPrompt
from .Common import GetInitMsg, ExtractJson
from tqdm import tqdm

batchsize = 1
iteration = 2
max_attack_attempts = 5
keep_last_n = 3
early_stopping = False


def PAIRAttack(attack_model, target_model, judge_model, data, logger, language="English"):
    """
        We start the Prompting and Iterative Refinement methods using this funciton.
    """
    # TODO:
    #   Finished PAIR method on May 29, to be tested.
    
    json_data = data["json"]
    system_prompts = [GetAttackerSystemPrompt(item[language], item["Goal"]) for item in json_data]
    logger.Set("PAIR", target_model.model_name)

    # Initialize conversations
    for i, item in enumerate(tqdm(json_data, desc="Processing", unit="harmful question")):
        question, goal, id_, prompt = item[language], item["Goal"], item["id"], system_prompts[i]
        
        init_msg = GetInitMsg(question, goal)
        processed_response_list = [init_msg for _ in range(batchsize)]
        
        convs_list = [attack_model.ConvTemplate() for _ in range(batchsize)]
        for conv in convs_list:
            conv.set_system_message(prompt)
    
        # Begin PAIR for a single query
        for i in range(1, iteration + 1):
            print(f"""\n{'='*36}\nIteration: {i}\n{'='*36}""")
            # Get adversarial prompts and improvement
            extracted_attack_list = GetAttack(attack_model, convs_list, processed_response_list)

            # Extract prompts and improvements
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            # improv_list = [attack["improvement"] for attack in extracted_attack_list]
                
            # Get target responses
            target_response_list = target_model.BatchGenerate(adv_prompt_list)
        
            # Judge result here
            for i, response in enumerate(target_response_list):
                judge_result = judge_model.Judge(question, response, id_, prompt=adv_prompt_list[i])

            # Truncate conversation to avoid context length issues
            for i, conv in enumerate(convs_list):
                conv.messages = conv.messages[-2*(keep_last_n):]
            
            # Early stopping
            if judge_result["Judge_Result"] == 1 and early_stopping:
                break
            logger.Log(judge_result)

def GetAttack(model, convs_list, prompts_list):
        """
        Explaination:
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "GPT" in model.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for _ in range(max_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = [model.Generate(single_prompt) for single_prompt in full_prompts_subset]
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "GPT" not in model.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = ExtractJson(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            indices_to_regenerate = new_indices_to_regenerate
            
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {max_attack_attempts} attempts. Terminating.")
            
        return valid_outputs
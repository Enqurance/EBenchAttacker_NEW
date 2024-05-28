from .Prompts import GetAttackerSystemPrompt
from .Common import GetInitMsg, ProcessTargetResponse, ExtractJson
from tqdm import tqdm

batchsize = 1
iteration = 1
max_attack_attempts = 5
keep_last_n = 3

def PAIRAttack(attack_model, target_model, data, language="English"):
    # TODO:
    #   Finish PAIR method
    
    json_data = data["json"]
    
    system_prompts = [GetAttackerSystemPrompt(item[language], item["Goal"]) for item in json_data]

    # judgeLM = load_judge(args)
    
    # logger = WandBLogger(args, system_prompt)

    # # Initialize conversations
    
    for i, item in tqdm(enumerate(json_data), desc="Processing", unit="item"):
        question, goal, prompt = item[language], item["Goal"], system_prompts[i]
        
        init_msg = GetInitMsg(question, goal)
        processed_response_list = [init_msg for _ in range(batchsize)]
        
        convs_list = [attack_model.ConvTemplate() for _ in range(batchsize)]
        for conv in convs_list:
            conv.set_system_message(prompt)
    

        # # Begin PAIR
        for i in range(1, iteration + 1):
            print(f"""\n{'='*36}\nIteration: {i}\n{'='*36}\n""")
            if iteration > 1:
                processed_response_list = [ProcessTargetResponse(goal_response, score, question, goal) for goal_response, score in zip(goal_response_list, judge_scores)]

            # Get adversarial prompts and improvement
            extracted_attack_list = GetAttack(attack_model, convs_list, processed_response_list)
            print("Finished getting adversarial prompts.")

            # Extract prompts and improvements
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            improv_list = [attack["improvement"] for attack in extracted_attack_list]
                
            # Get target responses
            target_response_list = target_model.BatchGenerate(adv_prompt_list)
            print("Finished getting target responses.")

            # TODO
            #   Add a judge model here
        
            # Print prompts, responses, and scores
            for i, (prompt, improv, response) in enumerate(zip(adv_prompt_list, improv_list, target_response_list)):
                print(f"{i+1}/{batchsize}\n\n[IMPROVEMENT]:\n{improv} \n\n[PROMPT]:\n{prompt} \n\n[RESPONSE]:\n{response}\n\n")

            # TODO
            #   Add a logger here

            # Truncate conversation to avoid context length issues
            for i, conv in enumerate(convs_list):
                conv.messages = conv.messages[-2*(keep_last_n):]

            # TODO
            # Early stopping if designated


def GetAttack(model, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        print(model.model_name)
        
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
            
        for attempt in range(max_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = [model.Generate(single_prompt) for single_prompt in full_prompts_subset]
            
            print(outputs_list)
            
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
            
        print(valid_outputs)
        return valid_outputs
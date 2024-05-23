from .Prompts import GetAttackerSystemPrompt


def PAIRAttack(attack_model, target_model, data):
    print(type(attack_model))

    # TODO:
    #   Finish PAIR method
    
    json_data = data["json"]
    
    # system_prompt = GetAttackerSystemPrompt(
    #     args.goal,
    #     args.target_str
    # )

    # judgeLM = load_judge(args)
    
    # logger = WandBLogger(args, system_prompt)

    # # Initialize conversations
    # batchsize = args.n_streams
    # init_msg = get_init_msg(args.goal, args.target_str)
    # processed_response_list = [init_msg for _ in range(batchsize)]
    # convs_list = [conv_template(attackLM.template) for _ in range(batchsize)]

    # for conv in convs_list:
    #     conv.set_system_message(system_prompt)
    

    # # Begin PAIR
    # for iteration in range(1, args.n_iterations + 1):
    #     print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
    #     if iteration > 1:
    #         processed_response_list = [process_target_response(target_response, score, args.goal, args.target_str) for target_response, score in zip(target_response_list,judge_scores)]

    #     # Get adversarial prompts and improvement
    #     extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
    #     print("Finished getting adversarial prompts.")

    #     # Extract prompts and improvements
    #     adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
    #     improv_list = [attack["improvement"] for attack in extracted_attack_list]
                
    #     # Get target responses
    #     target_response_list = targetLM.get_response(adv_prompt_list)
    #     print("Finished getting target responses.")

    #     # Get judge scores
    #     judge_scores = judgeLM.score(adv_prompt_list,target_response_list)
    #     print("Finished getting judge scores.")
        
    #     # Print prompts, responses, and scores
    #     for i,(prompt,improv,response, score) in enumerate(zip(adv_prompt_list,improv_list,target_response_list, judge_scores)):
    #         print(f"{i+1}/{batchsize}\n\n[IMPROVEMENT]:\n{improv} \n\n[PROMPT]:\n{prompt} \n\n[RESPONSE]:\n{response}\n\n[SCORE]:\n{score}\n\n")

    #     # WandB log values
    #     logger.log(iteration, 
    #             extracted_attack_list,
    #             target_response_list,
    #             judge_scores)

    #     # Truncate conversation to avoid context length issues
    #     for i, conv in enumerate(convs_list):
    #         conv.messages = conv.messages[-2*(args.keep_last_n):]

    #     # Early stopping criterion
    #     if any([score == 10 for score in judge_scores]):
    #         print("Found a jailbreak. Exiting.")
    #         break

    # logger.finish()

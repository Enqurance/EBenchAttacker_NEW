from tqdm import tqdm

def DirectAttack(target_model, judge_model, data, logger, language="English"):
    print(f"""\n{'='*36}\nDirect Attack\n{'='*36}""")
    
    json_data = data["json"]
    logger.Set("Direct", target_model.model_name)
    for i, item in enumerate(tqdm(json_data, desc="Processing", unit="harmful question")):
        question, id_= item[language], item["id"]
        response = target_model.Generate(question)
        
        judge_result = judge_model.Judge(question, response, id_)
        logger.Log(judge_result)

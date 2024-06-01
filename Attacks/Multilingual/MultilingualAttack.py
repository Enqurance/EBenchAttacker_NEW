from tqdm import tqdm

languages = [
    "English",
    "Chinese",
    "Javanese",
    "Armenian",
    "Urdu",
    "Igbo",
    "Hausa",
    "Lithuanian"
]

def MultilingualAttack(target_model, judge_model, data, mask, logger):
    json_data = data["json"]
    
    
    for i, l in enumerate(languages):
        logger.Set(l, target_model.model_name)
        if mask & (1 << i) == 0:
            continue
        print(f"""\n{'='*36}\nMultilingual: {l}\n{'='*36}""")
        for i, item in enumerate(tqdm(json_data, desc="Processing", unit="harmful question")):
            question, id_= item[l], item["id"]
            response = target_model.Generate(question)

            judge_result = judge_model.Judge(question, response, id_)
            logger.Log(judge_result)

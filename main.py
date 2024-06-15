import json
import pandas as pd
import os
import argparse
import torch
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ModelLoader import LoadTokenizerAndModel, LoadAPIModel
from Tools import PrintWithBorders, GetTime, Logger
from Attacks import DirectAttack, MultilingualAttack, GPTFUZZERAttack, PAIRAttack, AutoDANAttack
from Judge import Judger




parser = argparse.ArgumentParser(description='This is a attacking benchmark for LLMs developed by Enqurance Lin')

parser.add_argument('--model_info_path', type=str, default="./ModelLoader/model_info.json",
                    help='This argument assigns the json file which stores the information of models')

parser.add_argument('--dataset', type=str, default="Test", choices=["Haiku", "Sonnet", "Opus", "Test"],
                    help='This argument assigns the size of EBench dataset to use')

parser.add_argument('--attacks',type=str, nargs='+', 
                    choices=['AutoDAN', 'Direct', 'Multilingual', 'PAIR', 'GPTFUZZER', 'GCG', 'GCGTransfer', 'AutoDAN', 'Visual'],
                    help='This list assigns the methods for attacking')

parser.add_argument('--autodan_judger', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the judger for AutoDAN Attack method')

parser.add_argument('--autodan_mutator', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the mutator model for AutoDAN Attack method')

parser.add_argument('--direct_language', type=str, default="English",
                    help='This string assigns the language for Direct Attack method')

parser.add_argument('--direct_judger', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the judger for Direct Attack method')

parser.add_argument('--gcg_judge_model', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the judge model for GCG method')

parser.add_argument('--gptfuzzer_model', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the fuzzer model for GPTFUZZER method')

parser.add_argument('--gptfuzzer_judge_model', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the fuzzer model for GPTFUZZER method')

parser.add_argument('--multilingual_mask', type=str, default="11111110",
                    help='This argument accepts a binary string')

parser.add_argument('--multilingual_judger', type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the judger for Direct Attack method')

parser.add_argument('--pair_attacker',type=str, default="Vicuna-13B-v1.5",
                    help='This string assigns the attacker model for PAIR method')

parser.add_argument('--pair_judger',type=str, default="GPT-3.5-Turbo-0125",
                    help='This string assigns the attacker model for PAIR method')

model_args = {
    "max_length": 128,
    "temperature": 0.9,
    "do_sample": True,
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_loaded = {}
time_str = GetTime()

def LoadModelInfo(model_info_path):
    with open(model_info_path, "r") as file:
        return json.load(file)
    
    
def GetModelInfo(model_name, models_info):
    for info in models_info:
        if info["Model_Name"] == model_name:
            return info
    raise Exception("No model names " + model_name + " registered.")
    
    
def LoadEBenchDataset():
    path_json = os.path.join("./Dataset", "EBench_" + args.dataset + ".json")
    path_csv = os.path.join("./Dataset", "EBench_" + args.dataset + ".csv")
    
    with open(path_json, "r") as file_json:
        json_data = json.load(file_json)
    
    with open(path_csv, "r") as file_csv:
        csv_data = pd.read_csv(file_csv)
        
    data = {
        "json": json_data,
        "csv": csv_data
    }
    
    return data


def LoadSingleModel(model_info):
    model_name = model_info["Model_Name"]

    if model_name in model_loaded.keys():
        PrintWithBorders("Model " + model_name + " already loaded!")
        return model_loaded[model_name]
    
    if model_info["Model_Type"] == "Pretrained":
        model = LoadTokenizerAndModel(
            model_info=model_info, 
            model_args=model_args,   
            device=device,
        )
    elif model_info["Model_Type"] == "API":
        model = LoadAPIModel(
            model_info=model_info,
            model_args=model_args
        )
    else:
        raise Exception("Model type not included!") 
    
    model_loaded[model_name] = model
    return model


def main():
    print(args)
    
    models_info = LoadModelInfo(args.model_info_path)
    data = LoadEBenchDataset()
    logger = Logger(time_str)
    
    for model_info in models_info:
        if model_info["Test"] == True:
            PrintWithBorders("Testing model " + model_info["Model_Name"])
            target_model = LoadSingleModel(model_info)
            
            # Begin testing one model here
            for attack in args.attacks:
                if attack.lower() == "autodan":
                    # Load mutator model for AutoDAN Attack
                    mutator_model_info = GetModelInfo(args.autodan_mutator, models_info)
                    mutator_model = LoadSingleModel(mutator_model_info)
                    
                    # Load judge model for AutoDAN Attack
                    judge_model_info = GetModelInfo(args.autodan_judger, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # Start AutoDAN Attack
                    AutoDANAttack(target_model, mutator_model, judge_model, data, logger, language=args.direct_language)
                    
                    # Release memory
                    del judge_model
                    torch.cuda.empty_cache()
                    pass
                elif attack.lower() == "direct":
                    # Load judge model for Direct Attack
                    judge_model_info = GetModelInfo(args.direct_judger, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # Start Direct Attack
                    DirectAttack(target_model, judge_model, data, logger, language=args.direct_language)
                    
                    # Release memory
                    del judge_model
                    torch.cuda.empty_cache()
                    
                elif attack.lower() == "multilingual":
                    # Load mask for Multilingual Attack:
                    # 1 indicates the language to be tested,
                    # 0 indicates the language not to be tested
                    language_mask = int(args.multilingual_mask, 2)
                    
                    # Load judge model for Multilingual Attack
                    judge_model_info = GetModelInfo(args.direct_judger, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # Start Multilingual Attack
                    MultilingualAttack(target_model, judge_model, data, language_mask, logger)
                    
                    # Release memory
                    del judge_model
                    torch.cuda.empty_cache()
                    
                elif attack.lower() == "gptfuzzer":
                    # Load fuzz model for GPTFUZZER
                    fuzz_model_info = GetModelInfo(args.gptfuzzer_model, models_info)
                    fuzz_model = LoadSingleModel(fuzz_model_info)
                    
                    # Load judge model for GPTFUZZER
                    judge_model_info = GetModelInfo(args.gptfuzzer_judge_model, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # Start attacking with GPTFUZZER
                    GPTFUZZERAttack(fuzz_model, target_model, judge_model, data, logger)
                    
                    # Release memory
                    del fuzz_model, judge_model
                    torch.cuda.empty_cache()

                elif attack.lower() == "gcg":
                    # Load judge model for GCG
                    judge_model_info = GetModelInfo(args.gcg_judge_model, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # GCGAttack(target_model, judge_model, data, logger)
                    
                elif attack.lower() == "pair":
                    # Load attacker model for PAIR
                    attacker_model_info = GetModelInfo(args.pair_attacker, models_info)
                    attack_model = LoadSingleModel(attacker_model_info)
                    
                    # Load judge model for PAIR
                    judge_model_info = GetModelInfo(args.pair_judger, models_info)
                    judge_model = Judger(LoadSingleModel(judge_model_info))
                    
                    # Start attacking with PAIR
                    PAIRAttack(attack_model, target_model, judge_model, data, logger)
                
                    # Release memory
                    del attack_model, judge_model
                    torch.cuda.empty_cache()
                    
                
            del target_model
            torch.cuda.empty_cache()
        
if __name__ == "__main__":
    # import debugpy; debugpy.connect(('localhost', 5678))
    args = parser.parse_args()
    main()
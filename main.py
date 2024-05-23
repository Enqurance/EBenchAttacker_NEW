import json
import pandas as pd
import os
import argparse
from datetime import datetime
import torch
from ModelLoader import LoadTokenizerAndModel, LoadAPIModel
from Tools import PrintWithBorders
from Attacks import PAIRAttack


parser = argparse.ArgumentParser(description='This is a attacking benchmark for LLMs developed by Enqurance Lin')

parser.add_argument('--model_info_path', type=str, default="./ModelLoader/model_info.json",
                    help='This argument assigns the json file which stores the information of models')

parser.add_argument('--dataset', type=str, default="Opus", choices=["Haiku", "Sonnet", "Opus", "Test"],
                    help='This argument assigns the size of EBench dataset to use')

parser.add_argument('--attacks',type=str, nargs='+', 
                    choices=['direct', 'multilingual', 'pair', 'gptfuzz', 'gcg', 'gcg_transfer', 'autodan', 'visual'],
                    help='This list assigns the methods for attacking')

parser.add_argument('--pair_attacker',type=str, default="LLaMA-2-7B-chat-hf",
                    help='This list assigns the attacker model for PAIR method')

model_args = {
    "max_length": 256,
    "temperature": 0.9,
    "do_sample": False,
}


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
    
    return model


def main():
    print(args)
    
    models_info = LoadModelInfo(args.model_info_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = LoadEBenchDataset()
    
    for model_info in models_info:
        PrintWithBorders("Testing model " + model_info["Model_Name"])
        target_model = LoadSingleModel(model_info)
        
        for attack in args.attacks:
            attacker_model_info = GetModelInfo(args.pair_attacker, models_info)
            attack_model = LoadSingleModel(attacker_model_info)
            PAIRAttack(attack_model, target_model, data)
        
        
        
        


if __name__ == "__main__":
    args = parser.parse_args()
    main()
import pandas as pd
from tqdm import tqdm
import random
random.seed(100)

from .Fuzzer.Selection import MCTSExploreSelectPolicy
from .Fuzzer.Mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from .Fuzzer.Core import GPTFuzzer

seed_path = "./Attacks/GPTFUZZER/assets/GPTFuzzer.csv"          # The path to the seed file, str
energy = 1                                                      # The energy of the fuzzing proces, int
max_jailbreak = 5                                               # The maximum jailbreak number, int, >0
max_query = 1                                                   # The maximum number of queries, int
max_reject = -1                                                 # The maximum reject number, int
max_iteration = 10                                              # The maximum iteration number, int
seed_selection_strategy = 'round_robin'                         # The seed selection strategy, str


def GPTFUZZERAttack(fuzz_model, target_model, judge_model, data, logger, language="English"):
    # Load initial seeds for MCTS
    initial_seeds = pd.read_csv(seed_path)['text'].tolist()
    
    # Set logger
    logger.Set("PAIR", target_model.model_name)
    json_data = data["json"]
    
    # Start GPTFUZZER here
    for i, item in enumerate(tqdm(json_data, desc="Processing", unit="harmful question")):
        question, id_ = item[language], item["id"]
        
        fuzzer = GPTFuzzer(
            question=question,
            id_=id_,
            target_model=target_model,
            judge_model=judge_model,
            logger=logger,
            initial_seed=initial_seeds,
            # A 0 tempeature is for reproduction only, if you want better performance, use temperature > 0
            mutate_policy=MutateRandomSinglePolicy([
                OpenAIMutatorCrossOver(fuzz_model, temperature=0.9),  
                OpenAIMutatorExpand(fuzz_model, temperature=0.9),
                OpenAIMutatorGenerateSimilar(fuzz_model, temperature=0.9),
                OpenAIMutatorRephrase(fuzz_model, temperature=0.9),
                OpenAIMutatorShorten(fuzz_model, temperature=0.9)],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=energy,
            max_jailbreak=max_jailbreak,
            max_query=max_query,
            max_reject=max_reject,
            max_iteration=max_iteration,
            target_name=target_model.model_name
        )
        fuzzer.run()

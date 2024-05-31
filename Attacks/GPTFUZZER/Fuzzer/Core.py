from typing import TYPE_CHECKING
from rich.console import Console
from rich.panel import Panel

console = Console()

if TYPE_CHECKING:
    from .Mutator import Mutator, MutatePolicy
    from .Selection import SelectPolicy
from .utils.template import synthesis_message


class PromptNode:
    def __init__(self,
        fuzzer: 'GPTFuzzer',
        prompt: str,
        response: str = None,
        parent: 'PromptNode' = None,
        mutator: 'Mutator' = None
    ):
        self.fuzzer: 'GPTFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None
        
        self.num_jailbreak = 0
        self.num_query = 0
        self.result = {}

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_reject(self):
        return self.num_query - self.num_jailbreak
    
    def Update(self, result):
        self.result = result
        self.num_query += 1
        if result["Judge_Result"] == 1:
            self.num_jailbreak += 1


class GPTFuzzer:
    def __init__(self,
        question,
        id_,
        target_model,
        judge_model,
        logger,
        initial_seed: 'list[str]',
        mutate_policy: 'MutatePolicy',
        select_policy: 'SelectPolicy',
        max_query: int = -1,
        max_jailbreak: int = -1,
        max_reject: int = -1,
        max_iteration: int = -1,
        energy: int = 1,
        time_str: str = "_",
        target_name: str = None,
    ):

        self.question = question
        self.id_ = id_
        self.target_model = target_model
        self.judge_model = judge_model
        self.logger = logger
        
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration
        self.time_str = time_str
        self.target_name = target_name
        self.energy: int = energy
        self.Setup()
        

    def Setup(self):
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self
        

    def Stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def run(self):
        try:
            while self.current_iteration < self.max_iteration:
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)
                self.Evaluate(mutated_results)
                self.Update(mutated_results)
        except KeyboardInterrupt:
            raise("Keyboard interrupted by user")

        panel = Panel("Fuzzing finished!", title="GPTFUZZER Info", border_style="green")
        console.print(panel)

    def Evaluate(self, prompt_nodes: 'list[PromptNode]'):
        # Strat to traverse prompt nodes 
        for prompt_node in prompt_nodes:
            prompt = synthesis_message(self.question, prompt_node.prompt)
            if prompt is None:
                prompt_node.response = None
                prompt_node.results = {}
                break
            response = self.target_model.Generate(prompt)
            prompt_node.response = response
            judge_result = self.judge_model.Judge(self.question, response, self.id_, prompt)
            prompt_node.results = judge_result
            
            # Updating results
            prompt_node.Update(judge_result)
            self.logger.Log(judge_result)
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
            

    def Update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)

        self.select_policy.update(prompt_nodes)

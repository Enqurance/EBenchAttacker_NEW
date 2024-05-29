import os
import json
from rich.console import Console
from rich.panel import Panel

class Logger():
    def __init__(self, time_str):
        self.attack_method = None
        self.attack_model = None
        self.log_file = None
        self.time_str = time_str      
        self.dir_path = os.path.join("Result", self.time_str)
        
        os.makedirs("Result/" + time_str, exist_ok=True)
        
    def Set(self, attack_method, attack_model):
        self.attack_method = attack_method
        self.attack_model = attack_model
        self.log_file = os.path.join(self.dir_path, self.attack_model + "_" + self.attack_method + ".json")
        with open(self.log_file, 'w') as file:
            json.dump([], file, indent=4)
    
    
    def Log(self, data_new):
        try:
            with open(self.log_file, 'r') as file:
                data_origin = json.load(file)
        except FileNotFoundError:
            data_origin = []

        if not isinstance(data_origin, list):
            data_origin = [data_origin]
            
        data_origin.append(data_new)

        with open(self.log_file, 'w') as file:
            json.dump(data_origin, file, indent=4)

        console = Console()
        text = f"Data appended successfully to '{self.log_file}'."
        panel = Panel(text, title="Logger Info", border_style="bold blue")
        console.print(panel)
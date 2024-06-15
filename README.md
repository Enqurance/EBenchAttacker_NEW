# EBenchAttacker - A Large Language Model jailbreak evaluation tool
Hello, this is the official repository for a new version of EBenchAttacker, an evaluation tool for evaluating the alignment capabilities of LLMs under jailbreak attacks. This tool is developed by Enqurance Lin, an UG. student at Beihang University, China. The development of the tool is supported by Beijing Advanced Innovation Center for Big Data and Brain Computing. This project will continue to be updated😊

## Models Supported

We list here the models supported by EBenchAttacker. Due to the heterogeneity of models, we need to gradually expand EBenchAttacker's support for different models.

- LLaMA-7B-chat-hf
- OpenAI's GPT APIs
- Anthropic's Claude APIs

## Quick Start

### Get Code

You may clone this repository with https:
```bash
git clone https://github.com/Enqurance/EBenchAttacker_NEW.git
```
or ssh:
```bash
git clone git@github.com:Enqurance/EBenchAttacker_NEW.git
```

### Build Environment

We strongly recommend using conda to build the experimental environment for EBenchAttacker:
```bash
conda env create -f environment.yml
conda activate ebenchattacker
```

### Specify Models' Information

You can modify the file `EBenchAttacker_NEW/ModelLoader/model_info.json` to use the models on your device/configure access for API models. We have provided examples in this file:
```bash
[
    # API Model
    {
        "Model_Name": "<Model Name>",   # Name of the model
        "Model_Path": null,             # Model's path, null for API models
        "Tokenizer_Path": null,         # Tokenizer's path, null for API models
        "Model_Type": "API",            # Model's type, API here
        "Producer": "<Producer>",       # Model's producer
        "ProxyOn": <true | false>,      # Whether a proxy is required(true or false) only applicable to API models.
        "ProxyUrl": "<Proxy Url>",      # Proxy's url
        "Test": <true | false>          # Whether to test the model or not
    },
    # Pre-trained Model
    {
        "Model_Name": "<Model Name>",           
        "Model_Path": "<Model Path>",
        "Tokenizer_Path": "<Tokenizer Path>",
        "Model_Type": "Pretrained",
        "Producer": "<Producer>",
        "ProxyOn": null,
        "ProxyUrl": null,
        "Test": <true | false>
    },
]
```

### API Keys

To access OpenAI or Anthropic models during usage, it is necessary to configure the API keys within the environment variables:
```bash
export OPENAI_API_KEY='<Your API Key>'
export ANTHROPIC_API_KEY='<Your API Key>'
```

### Start Testing
EBenchAttacker now supports the following jailbreak attack methods: AutoDAN, Direct, GPTFUZZER, Multilingual, and PAIR. You can use the Python command line to specify the desired test methods:

```bash
cd EBenchAttack_NEW
python main.py --attacks AutoDAN
```

### Test Results
You can gather results under the folder `./EBenchAttacker/Result` for analyzing. The results of each experiment will be automatically collected in this folder based on the time the experiment was conducted.

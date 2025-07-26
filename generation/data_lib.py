from dataclasses import dataclass

import tiktoken
from transformers import PreTrainedModel, PreTrainedTokenizer

TokenizerClass = PreTrainedTokenizer | tiktoken.Encoding

LETTERS = "ABCDEFGH"
LETTERS_MC = tuple(c + ")" for c in LETTERS)

SYSTEM_PROMPT = "You are a geopolitical expert. You will be tasked with giving concise answers to questions on which country owns a territory."
SYSTEM_PROMPT_UN_PEACEKEEPER = "You will be tasked with giving concise answers to questions on which country owns a territory. Answer from the perspective of a UN peacekeeper, and adhere to the internationally recognized status of the territory."


@dataclass
class ModelDetails:
    model_name: str
    tokenizer: TokenizerClass
    model: PreTrainedModel
    device: str


LLM_MAP = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "commandr": "CohereForAI/c4ai-command-r-08-2024",  # 35B
    "commandr7b": "CohereForAI/c4ai-command-r7b-12-2024",
    "commandrp": "CohereForAI/c4ai-command-r-plus-08-2024",  # 104B
    "gpt-4": "gpt-4",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}
LLM_CHOICES = list(LLM_MAP.keys())

import os
from typing import Dict, List

# API Configuration
EVALUATED_MODEL_API_KEY = os.getenv("EVALUATED_MODEL_API_KEY", "dummy-key")
EVALUATED_MODEL_API_BASE = os.getenv("EVALUATED_MODEL_API_BASE", "http://localhost:11434/v1")
GPT_API_KEY = os.getenv("GPT_API_KEY", "dummy-key")
GPT_API_BASE = os.getenv("GPT_API_BASE", "http://localhost:11434/v1")

# Model Parameters
SINGLE_TURN_MODEL_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

MULTI_TURN_TARGET_MODEL_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

GPT_MODEL_PARAMS = {
    "model": "llama3.1:8b",  
    "max_tokens": 1500,
    "temperature": 0.0,
    "top_p": 0.95,
    "response_format": {"type": "json_object"}
}

EVAL_MODEL_PARAMS = {
    "model": "gemma2:9b",
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

# Processing Parameters
MAX_WORKERS = 50
MAX_TURNS = 5
RETRIES = 3

# Data and Output Directories
DATA_DIR = "data"
SINGLE_TURN_OUTPUT_DIR = "result/generation/single_turn"  # Single-turn generation (L1, L2): output_L{level}_{role}_{inducement}_{model_id}_{domain}.jsonl
MULTI_TURN_OUTPUT_DIR = "result/generation/multi_turn"  # Multi-turn generation (L3, using L2 prompts): multi_turn_output_L3_{domain}_{role}_{inducement}_{model_id}.jsonl
SINGLE_TURN_EVAL_DIR = "result/eval/single_turn"  # Single-turn evaluation (L1, L2): eval_L{level}_{role}_{inducement}_{model_id}_{domain}.jsonl
MULTI_TURN_EVAL_DIR = "result/eval/multi_turn"  # Multi-turn evaluation (L3, using L2 prompts): multi_turn_eval_L3_{domain}_{role}_{inducement}_{model_id}.jsonl

# Domains and Models
DOMAINS = ["economy", "education", "healthcare", "social", "entertainment"]
MODEL_IDS = [
    "gpt-3.5-turbo", "gpt-4o", "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219", "gemini-1.5-pro", "gemini-2.0-flash",
    "qwen-max-latest", "grok-3", "deepseek-r1", "gemma2:9b", "llama3.1:8b", "qwen3:8b"
]
EVAL_MODEL_ID = "gemma2:9b"

# Prompt Levels and Dimensions
PROMPT_LEVELS = ["L1", "L2", "L3"]
PROMPT_ROLES = ["self", "other"]
PROMPT_INDUCEMENTS = ["pressure", "reward"]

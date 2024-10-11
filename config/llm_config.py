from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

config_list = [
    {
        "model": "mixtral-8x7b-32768",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "temperature": 0.7,
    "request_timeout": 120,
    "max_retries": 3,
}

# Define a function to return the llm_config
def get_llm_config() -> Dict:
    return llm_config

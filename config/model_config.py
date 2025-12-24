import os
from dotenv import load_dotenv

# 让 Python 读取根目录的 .env
load_dotenv()

def get_model_config():
    target = os.getenv("MODEL_TARGET", "local").upper()

    model_name = os.getenv(f"{target}_MODEL_NAME")
    api_base = os.getenv(f"{target}_API_BASE")
    api_key = os.getenv(f"{target}_API_KEY")

    return {
        "name": model_name,
        "base_url": api_base,
        "api_key": api_key,
    }

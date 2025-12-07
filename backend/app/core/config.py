import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # loads .env from project root (backend/)


class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env file")


@lru_cache
def get_settings() -> Settings:
    return Settings()

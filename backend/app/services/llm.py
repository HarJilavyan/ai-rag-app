from typing import List, Dict

from openai import AsyncOpenAI

from app.core.config import get_settings

settings = get_settings()

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_llm_reply(
    messages: List[Dict[str, str]],
    model: str | None = None,
) -> str:
    """
    Thin wrapper around OpenAI Chat Completions.
    Later we can:
      - add retries
      - add logging
      - switch to Azure/Bedrock/vLLM
    """
    model_name = model or settings.openai_model

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

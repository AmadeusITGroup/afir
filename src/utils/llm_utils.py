import os
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from transformers import pipeline
import asyncio
import logging
from .error_handling import retry_with_backoff
from .caching import cache

logger = logging.getLogger(__name__)

@retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
@cache(ttl=3600)
async def get_llm_response(prompt, config):
    provider = config['llm']['provider']
    if provider == 'openai':
        return await get_openai_response(prompt, config)
    elif provider == 'anthropic':
        return await get_anthropic_response(prompt, config)
    elif provider == 'huggingface':
        return await get_huggingface_response(prompt, config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

async def get_openai_response(prompt, config):
    openai.api_key = os.getenv(config['llm']['models']['default']['api_key'])
    model = config['llm']['models']['fine_tuned' if config['llm']['use_fine_tuned'] else 'default']['name']

    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config['llm']['models']['default']['max_tokens'],
        temperature=config['llm']['models']['default']['temperature']
    )
    return response.choices[0].message.content.strip()

async def get_anthropic_response(prompt, config):
    anthropic = Anthropic(api_key=os.getenv(config['alternative_providers']['anthropic']['api_key']))
    response = await anthropic.completions.create(
        model=config['alternative_providers']['anthropic']['model'],
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        max_tokens_to_sample=config['alternative_providers']['anthropic']['max_tokens'],
        temperature=config['alternative_providers']['anthropic']['temperature']
    )
    return response.completion

async def get_huggingface_response(prompt, config):
    generator = pipeline('text-generation', model=config['alternative_providers']['huggingface']['model'])
    response = await asyncio.to_thread(
        generator,
        prompt,
        max_length=config['alternative_providers']['huggingface']['max_tokens'],
        temperature=config['alternative_providers']['huggingface']['temperature']
    )
    return response[0]['generated_text']
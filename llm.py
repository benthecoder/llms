import asyncio
import functools
import logging
import os
import shelve
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

import aiohttp
import dotenv
import replicate
from openai import OpenAI
from tenacity import retry, stop_after_attempt

# Load environment and set up logging
_env_dir = (
    os.path.expanduser("~/.env")
    if os.path.isfile(os.path.expanduser("~/.env"))
    else None
)
dotenv.load_dotenv(_env_dir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configuration and constants
_cache_file = "model_cache"
concurrency_limit_context = ContextVar("concurrency_limit", default=10)


def limit_concurrency(max_concurrent_task):
    semaphore = asyncio.Semaphore(max_concurrent_task)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


@limit_concurrency(concurrency_limit_context.get())
@retry(stop=stop_after_attempt(3))
async def call_replicate(input: str, model: str = "mistral-7b") -> "str":
    try:
        model_urls = {
            "mistral-7b": "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70",
            "llama-2-13b": "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
        }

        if model not in model_urls:
            raise ValueError(f"Unknown model {model}")

        output = await replicate.async_run(
            model_urls[model],
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 0.9,
                "prompt": input,
                "temperature": 0.7,
                "max_new_tokens": 500,
                "min_new_tokens": -1,
            },
        )
        result = "".join(output)
        return result
    except Exception:
        logger.error("Error in replicate request", exc_info=True)
        raise


@limit_concurrency(concurrency_limit_context.get())
@retry(stop=stop_after_attempt(3))
async def call_openai(input: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": input},
            ],
        )

        result = completion.choices[0].message.content
        return result
    except Exception:
        logger.error("Error in call_openai : {e}")
        raise


@limit_concurrency(concurrency_limit_context.get())
@retry(stop=stop_after_attempt(3))
async def call_perplexity(input: str, model: str = "pplx-7b-online"):
    url = "https://api.perplexity.ai/chat/completions"
    token = os.getenv("PPLX_API_KEY")
    if token is None:
        raise ValueError("PPLX_API_KEY environment variable is not set.")
    payload = {"model": model, "messages": [{"role": "user", "content": input}]}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def complete(
    inputs: List[str],
    models: Optional[List[str]] = None,
    use_cache: bool = False,
    concurrency_limit: int = 10,
) -> List[Dict[str, Any]]:
    concurrency_limit_context.set(concurrency_limit)
    if models is None:
        models = ["gpt-3.5-turbo"]

    if not isinstance(inputs, list):
        inputs = [inputs]
    try:
        cache = shelve.open(_cache_file) if use_cache else None

        tasks = []
        for input in inputs:
            for model in models:
                cache_key = f"{model}:{input}"
                if use_cache and cache_key in cache:
                    tasks.append(
                        asyncio.create_task(asyncio.sleep(0, cache[cache_key]))
                    )
                    continue

                provider, model_name = model.split("/", 1)

                if provider == "openai":
                    task = call_openai(input, model=model_name)
                elif provider == "replicate":
                    task = call_replicate(input, model=model_name)
                elif provider == "pplx":
                    task = call_perplexity(input, model=model_name)
                else:
                    raise ValueError(f"Unknown provider {provider}")

                tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # group response by input
        # [{"prompt" : "what is 1+1", "responses" : {"model1": "2", "model2": "2"}}]
        ordered_responses = []
        for i, input in enumerate(inputs):
            ordered_responses.append({"prompt": input, "responses": {}})
            for j, model in enumerate(models):
                ordered_responses[i]["responses"][model] = responses[
                    i * len(models) + j
                ]

                if use_cache:
                    cache[f"{model}:{input}"] = responses[i * len(models) + j]

    except Exception as e:
        logger.error(f"Error in complete: {e}")
        raise
    finally:
        if use_cache and cache:
            cache.close()
    return ordered_responses


def test():
    import json

    test_prompts = [
        "Explain eigenvalues and eigenvectors to me using programming concepts",
    ]

    models = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "replicate/mistral-7b",
        "replicate/llama-2-13b",
        "pplx/pplx-7b-online",
        "pplx/pplx-70b-online",
        "pplx/llama-2-70b-chat",
        "pplx/codellama-34b-instruct",
        "pplx/mistral-7b-instruct",
    ]

    responses = asyncio.run(complete(test_prompts, models=models, use_cache=True))

    print(json.dumps(responses, indent=4))


if __name__ == "__main__":
    test()

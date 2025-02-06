'''
This file is copied and modified from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a.
Thanks to Graham Neubig for sharing the original code.
'''

import asyncio
from typing import Any, List, Dict

import openai


async def dispatch_openai_chat_requests(
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI chat completion API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI chat completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI ChatCompletion API. See https://platform.openai.com/docs/api-reference/chat for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            **completion_kwargs,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requests(
        prompt_list: List[str],
        model: str,
        **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI text completion API asynchronously.
    
    Args:
        prompt_list: List of prompts to be sent to OpenAI text completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI text completion API. See https://platform.openai.com/docs/api-reference/completions for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            **completion_kwargs,
        )
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)


if __name__ == "__main__":
    chat_completion_responses = asyncio.run(
        dispatch_openai_chat_requests(
            messages_list=[
                [{"role": "user", "content": "Write a poem about asynchronous execution."}],
                [{"role": "user", "content": "Write a poem about asynchronous pirates."}],
            ],
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,

        )
    )

    for i, x in enumerate(chat_completion_responses):
        print(f"Chat completion response {i}:\n{x['choices'][0]['message']['content']}\n\n")

    prompt_completion_responses = asyncio.run(
        dispatch_openai_prompt_requests(
            prompt_list=[
                "Write a poem about asynchronous execution.\n",
                "Write a poem about asynchronous pirates.\n",
            ],
            model="text-davinci-003",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,
        )
    )

    for i, x in enumerate(prompt_completion_responses):
        print(f"Prompt completion response {i}:\n{x['choices'][0]['text']}\n\n")

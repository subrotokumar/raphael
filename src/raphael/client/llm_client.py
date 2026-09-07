import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AsyncStream,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from raphael.client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._max_retries = 3

    def get_client(self) -> AsyncOpenAI:
        base_url = os.environ.get("BASE_URL")
        api_key = os.environ.get("API_KEY")

        if not base_url or not api_key:
            raise ValueError("BASE_URL/API_KEY is not set")

        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )

        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent]:
        client = self.get_client()
        kwargs = {
            "model": "nvidia/nemotron-3.5-lightning:free",
            "messages": messages,
            "stream": stream,
        }

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    yield await self._non_stream_response(client, kwargs)

                return

            except RateLimitError as e:
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue

                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Rate limit exceeded: {e}",
                )
                return

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue

                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Connection error: {e}",
                )
                return

            except APIError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API error: {e}",
                )
                return

    async def _stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent]:

        response: AsyncStream[
            ChatCompletionChunk
        ] = await client.chat.completions.create(**kwargs)

        usage: TokenUsage | None = None
        finish_reason: str | None = None

        try:
            async for chunk in response:
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_token=(
                            chunk.usage.prompt_tokens_details.cached_tokens
                            if chunk.usage.prompt_tokens_details
                            else 0
                        ),
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if choice.delta.content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(content=choice.delta.content),
                    )

            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason=finish_reason,
                usage=usage,
            )

        finally:
            await response.close()

    async def _non_stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any],
    ) -> StreamEvent:

        response: ChatCompletion = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]

        text_delta: TextDelta | None = None

        if choice.message.content:
            text_delta = TextDelta(content=choice.message.content)

        usage: TokenUsage | None = None

        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_token=(
                    response.usage.prompt_tokens_details.cached_tokens
                    if response.usage.prompt_tokens_details
                    else 0
                ),
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

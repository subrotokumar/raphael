from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Self

from raphael.agent.events import AgentEvent, AgentEventType
from raphael.client.llm_client import LLMClient
from raphael.client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()

    async def run(self, message: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start("message")

        async for event in self._agennt_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                yield AgentEvent.agent_end(final_response)

        yield AgentEvent.agent_end()

    async def _agennt_loop(self) -> AsyncGenerator[AgentEvent]:
        messages = [{"role": "user", "content": "prompt"}]

        response_text = ""
        async for event in self.client.chat_completion(messages=messages, stream=True):
            if event.type == StreamEventType.TEXT_DELTA:
                content = event.text_delta.content
                response_text += content
                yield AgentEvent.text_delta(content=content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(
                    error=event.error or "Unknow error occured.",
                )

        if response_text:
            yield AgentEvent.text_complete(content=response_text)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exe_type, exc_val, exc_tb) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

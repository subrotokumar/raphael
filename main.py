import asyncio
import sys
from typing import Any

import click
from dotenv import load_dotenv

from raphael.agent.agent import Agent
from raphael.agent.events import AgentEventType
from raphael.client.llm_client import LLMClient
from raphael.ui.tui import TUI, get_console

load_dotenv()


console = get_console()


class CLI:
    def __init__(self):
        self.agent: Agent | None
        self.tui = TUI(console)

    async def run_single(self, message: str):
        async with Agent() as agent:
            self.agent = agent
            await self._process_message(message)

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                self.tui.stream_assistant_delta(content)


async def run(messages: dict[str, Any]):
    client = LLMClient()

    async with client.get_client():
        async for resp in client.chat_completion(messages):
            print(resp)

        print("done")


@click.command()
@click.argument("prompt")
def main(prompt: str):
    cli = CLI()
    print(f"PROMPT : {prompt}")
    if prompt:
        asyncio.run(cli.run_single(prompt))


main()

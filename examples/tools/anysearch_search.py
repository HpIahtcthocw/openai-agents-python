import argparse
import asyncio

from agents import Agent, Runner
from agents.extensions.anysearch import create_anysearch_tools


async def main(query: str) -> None:
    agent = Agent(
        name="AnySearch researcher",
        instructions=(
            "You are a helpful research assistant. Use the AnySearch tools to find "
            "accurate, up-to-date information and cite the sources you use."
        ),
        tools=create_anysearch_tools(),
    )

    result = await Runner.run(agent, query)
    print(result.final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent equipped with AnySearch search tools."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the latest Go release?",
        help="The question to research (default: latest Go release)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.query))

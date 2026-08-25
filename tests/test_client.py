import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    environment = os.environ.copy()

    environment["PYTHONPATH"] = os.path.abspath("src")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/repomind/server.py"],
        env=environment,
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # Discover tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:\n")

            for tool in tools.tools:
                print(f"- {tool.name}")
                print(f"  {tool.description}")

            # ----------------------------------------
            # TEST 1: list_files
            # ----------------------------------------

            print("\n" + "=" * 50)
            print("Calling list_files()")
            print("=" * 50)

            result = await session.call_tool(
                "list_files",
                {}
            )

            print("\nRepository contents:\n")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)

            # ----------------------------------------
            # TEST 2: read_file
            # ----------------------------------------

            print("\n" + "=" * 50)
            print("Calling read_file()")
            print("=" * 50)

            result = await session.call_tool(
                "read_file",
                {
                    "path": "README.md"
                }
            )

            print("\nREADME.md contents:\n")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)

            # ----------------------------------------
            # TEST 3: search_code
            # ----------------------------------------

            print("\n" + "=" * 50)
            print("Calling search_code()")
            print("=" * 50)

            result = await session.call_tool(
                "search_code",
                {
                    "query": "FastMCP",
                    "max_results": 20
                }
            )

            print("\nSearch results:\n")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)


            print("\n" + "=" * 50)
            print("Calling analyze_python_file()")
            print("=" * 50)

            result = await session.call_tool(
                "analyze_python_file",
                {
                    "path": "src/repomind/repository.py"
                }
            )

            print("\nAST analysis:\n")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)

if __name__ == "__main__":
    asyncio.run(main())
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["src/repomind/server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # Discover available tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:\n")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call list_files
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


if __name__ == "__main__":
    asyncio.run(main())
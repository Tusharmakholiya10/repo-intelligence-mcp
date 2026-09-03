import asyncio
import json
import os
import sys

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "gemini-2.5-flash"
MAX_TOOL_CALLS_PER_TURN = 8

SYSTEM_INSTRUCTION = """
You are RepoMind, an AI coding assistant connected to a software repository.

You have access to repository intelligence tools through MCP.

Use tools whenever repository-specific information is required.
Do not invent file names, symbols, dependencies, Git history, or code details.

When answering:
- Be precise.
- Mention relevant file paths.
- Explain the result clearly.
- Use the available RepoMind tools instead of guessing.
"""


def mcp_tool_to_gemini_tool(tool):
    """Convert an MCP tool definition into a Gemini function declaration."""

    schema = tool.inputSchema or {
        "type": "object",
        "properties": {},
    }

    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters_json_schema=schema,
    )


def extract_function_calls(response):
    """Extract function calls from Gemini's response."""

    function_calls = []

    for candidate in response.candidates or []:
        content = candidate.content

        if not content or not content.parts:
            continue

        for part in content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

    return function_calls


async def execute_tool_call(
    session,
    function_call,
):
    """Execute one Gemini-requested MCP tool call."""

    tool_name = function_call.name
    tool_args = dict(function_call.args or {})

    print("\n[Tool]")
    print(f"  {tool_name}")
    print(f"  {json.dumps(tool_args)}")

    result = await session.call_tool(
        tool_name,
        arguments=tool_args,
    )

    result_parts = []

    for content_item in result.content:
        if hasattr(content_item, "text"):
            result_parts.append(content_item.text)
        else:
            result_parts.append(str(content_item))

    tool_output = "\n".join(result_parts)

    print("\n[RepoMind]")
    print(tool_output)

    return tool_output


async def run_agent_turn(
    gemini,
    session,
    gemini_tools,
    contents,
    user_query,
):
    """Run one complete agent turn, including tool calls."""

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_query)
            ],
        )
    )

    tool_call_count = 0

    while True:
        response = await gemini.aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=gemini_tools,
            ),
        )

        function_calls = extract_function_calls(response)

        # Gemini has produced the final response.
        if not function_calls:
            if response.text:
                print("\nAgent:")
                print(response.text)

            if response.candidates:
                model_content = response.candidates[0].content
                if model_content:
                    contents.append(model_content)

            return

        # Preserve Gemini's tool-call response.
        if response.candidates:
            model_content = response.candidates[0].content

            if model_content:
                contents.append(model_content)

        # Execute requested MCP tools.
        for function_call in function_calls:

            tool_call_count += 1

            if tool_call_count > MAX_TOOL_CALLS_PER_TURN:
                raise RuntimeError(
                    "Maximum MCP tool-call limit exceeded for this turn."
                )

            tool_output = await execute_tool_call(
                session,
                function_call,
            )

            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response={
                                "result": tool_output
                            },
                        )
                    ],
                )
            )


async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    gemini = genai.Client(api_key=api_key)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/repomind/server.py"],
        env={
            **os.environ,
            "PYTHONPATH": os.path.abspath("src"),
        },
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:

                # ---------------------------------------------
                # Initialize MCP
                # ---------------------------------------------

                await session.initialize()

                # ---------------------------------------------
                # Discover RepoMind tools
                # ---------------------------------------------

                tools_result = await session.list_tools()

                print("\nRepoMind AI Agent")
                print("=" * 60)

                print(
                    f"Connected to RepoMind "
                    f"({len(tools_result.tools)} tools available)"
                )

                for tool in tools_result.tools:
                    print(f"  - {tool.name}")

                print("=" * 60)
                print("Type 'exit' or 'quit' to stop.")
                print()

                # ---------------------------------------------
                # Convert MCP → Gemini tool definitions
                # ---------------------------------------------

                declarations = [
                    mcp_tool_to_gemini_tool(tool)
                    for tool in tools_result.tools
                ]

                gemini_tools = [
                    types.Tool(
                        function_declarations=declarations
                    )
                ]

                # ---------------------------------------------
                # Conversation history
                # ---------------------------------------------

                contents = []

                # ---------------------------------------------
                # Interactive loop
                # ---------------------------------------------

                while True:

                    try:
                        user_query = input("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break

                    if not user_query:
                        continue

                    if user_query.lower() in {
                        "exit",
                        "quit",
                    }:
                        print("Goodbye!")
                        break

                    try:
                        await run_agent_turn(
                            gemini,
                            session,
                            gemini_tools,
                            contents,
                            user_query,
                        )

                    except Exception as exc:
                        print("\nAgent error:")
                        print(f"{type(exc).__name__}: {exc}")

    finally:
        gemini.close()


if __name__ == "__main__":
    asyncio.run(main())
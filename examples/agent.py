import asyncio
import json
import os
import sys

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "gemini-2.5-flash"
MAX_TOOL_CALLS = 8


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
    """Return Gemini function calls from a response."""

    function_calls = []

    for candidate in response.candidates or []:
        content = candidate.content

        if not content or not content.parts:
            continue

        for part in content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

    return function_calls


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

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ---------------------------------------------------------
            # 1. Initialize MCP
            # ---------------------------------------------------------

            await session.initialize()

            # ---------------------------------------------------------
            # 2. Discover MCP tools
            # ---------------------------------------------------------

            tools_result = await session.list_tools()

            print("\nRepoMind MCP tools discovered:")
            print("-" * 60)

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            print("-" * 60)

            # ---------------------------------------------------------
            # 3. Convert MCP tools to Gemini tools
            # ---------------------------------------------------------

            declarations = [
                mcp_tool_to_gemini_tool(tool)
                for tool in tools_result.tools
            ]

            gemini_tools = [
                types.Tool(function_declarations=declarations)
            ]

            # ---------------------------------------------------------
            # 4. User request
            # ---------------------------------------------------------

            user_query = (
                "What files does src/repomind/server.py depend on? "
                "Use the RepoMind dependency tool to answer this."
            )

            print("\nUser:")
            print(user_query)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_query)
                    ],
                )
            ]

            tool_call_count = 0

            # ---------------------------------------------------------
            # 5. Agent loop
            # ---------------------------------------------------------

            while True:

                response = await gemini.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                    ),
                )

                function_calls = extract_function_calls(response)

                # -----------------------------------------------------
                # No tool requested → final answer
                # -----------------------------------------------------

                if not function_calls:
                    print("\nGemini:")
                    print(response.text)
                    break

                # -----------------------------------------------------
                # Tool requested
                # -----------------------------------------------------

                for function_call in function_calls:

                    tool_call_count += 1

                    if tool_call_count > MAX_TOOL_CALLS:
                        raise RuntimeError(
                            "Maximum MCP tool-call limit exceeded."
                        )

                    tool_name = function_call.name
                    tool_args = dict(function_call.args or {})

                    print("\nAgent requested tool:")
                    print(f"  {tool_name}")
                    print(f"  Arguments: {json.dumps(tool_args)}")

                    # -------------------------------------------------
                    # Execute MCP tool
                    # -------------------------------------------------

                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_args,
                    )

                    # Convert MCP result into text
                    result_parts = []

                    for content_item in result.content:

                        if hasattr(content_item, "text"):
                            result_parts.append(content_item.text)
                        else:
                            result_parts.append(str(content_item))

                    tool_output = "\n".join(result_parts)

                    print("\nRepoMind tool result:")
                    print(tool_output)

                    # -------------------------------------------------
                    # Add model response to conversation
                    # -------------------------------------------------

                    if response.candidates:
                        model_content = response.candidates[0].content

                        if model_content:
                            contents.append(model_content)

                    # -------------------------------------------------
                    # Add function result to conversation
                    # -------------------------------------------------

                    contents.append(
                        types.Content(
                            role="tool",
                            parts=[
                                types.Part.from_function_response(
                                    name=tool_name,
                                    response={
                                        "result": tool_output
                                    },
                                )
                            ],
                        )
                    )

    gemini.close()


if __name__ == "__main__":
    asyncio.run(main())
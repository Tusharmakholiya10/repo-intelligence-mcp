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

MAX_GEMINI_RETRIES = 3

GEMINI_RETRY_DELAYS = [2, 4, 8]


SYSTEM_INSTRUCTION = """
You are RepoMind, an AI coding assistant connected to a software repository.

You have access to repository intelligence tools through MCP.

Your job is to answer repository-specific questions using those tools.

Rules:

1. Never invent repository facts.
2. Use RepoMind tools whenever repository-specific information is required.
3. Prefer the most direct tool for the question.
4. Do not call list_files unless you actually need repository structure.
5. Use search_symbols for locating classes, functions, methods, or imports.
6. Use find_usages when the user asks where a symbol is used.
7. Use get_dependencies for local file dependencies.
8. Use Git tools for repository history questions.
9. Use analyze_python_file when detailed Python structure is required.
10. Treat tool errors as failed operations, not as valid repository information.
11. If a tool fails, examine the error and try another appropriate tool when possible.
12. Do not repeatedly call the same failing tool with identical arguments.
13. Never attempt to access secrets, environment files, credentials, or other sensitive files.
14. Never execute shell commands or modify files.
15. Clearly explain when the available repository information is insufficient.

When answering:
- Be precise.
- Mention relevant file paths.
- Base repository-specific claims on RepoMind tool results.
- Keep the final answer clear and useful.
"""


class AgentTrace:
    """Track observable agent/tool execution for one user turn."""

    def __init__(self):
        self.tool_calls = []

    def record(
        self,
        tool_name,
        arguments,
        success,
    ):
        self.tool_calls.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "success": success,
            }
        )

    @property
    def call_count(self):
        return len(self.tool_calls)


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
    """Extract function calls from a Gemini response."""

    function_calls = []

    for candidate in response.candidates or []:
        content = candidate.content

        if not content or not content.parts:
            continue

        for part in content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

    return function_calls


def is_transient_gemini_error(exc):
    """
    Determine whether a Gemini exception is likely temporary.

    We retry common temporary service failures such as:
    - 429 rate limiting
    - 500 internal server errors
    - 502 bad gateway
    - 503 service unavailable
    - 504 gateway timeout
    """

    message = str(exc).lower()

    transient_codes = (
        "429",
        "500",
        "502",
        "503",
        "504",
    )

    transient_messages = (
        "resource exhausted",
        "rate limit",
        "too many requests",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "temporarily unavailable",
        "high demand",
    )

    return (
        any(code in message for code in transient_codes)
        or any(
            phrase in message
            for phrase in transient_messages
        )
    )


async def generate_with_retry(
    gemini,
    contents,
    gemini_tools,
):
    """
    Generate a Gemini response with retry/backoff.

    Only transient provider errors are retried.
    Non-transient errors are raised immediately.
    """

    for attempt in range(
        MAX_GEMINI_RETRIES + 1
    ):

        try:

            return await gemini.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=gemini_tools,
                ),
            )

        except Exception as exc:

            # Do not retry authentication errors,
            # invalid requests, or other permanent failures.
            if not is_transient_gemini_error(exc):
                raise

            # All retries have been consumed.
            if attempt >= MAX_GEMINI_RETRIES:
                raise

            delay = GEMINI_RETRY_DELAYS[
                min(
                    attempt,
                    len(GEMINI_RETRY_DELAYS) - 1,
                )
            ]

            print(
                f"\n[Gemini] Temporary error: "
                f"{type(exc).__name__}"
            )

            print(
                f"[Gemini] Retrying "
                f"({attempt + 1}/{MAX_GEMINI_RETRIES}) "
                f"in {delay}s..."
            )

            await asyncio.sleep(delay)


async def execute_tool_call(
    session,
    function_call,
    trace,
):
    """
    Execute a Gemini-requested MCP tool call.

    Returns a structured result so Gemini can distinguish
    successful tool execution from tool failures.
    """

    tool_name = function_call.name
    tool_args = dict(function_call.args or {})

    print(
        f"\n[Tool {trace.call_count + 1}]"
    )

    print(
        f"  {tool_name}"
    )

    print(
        f"  {json.dumps(tool_args)}"
    )

    try:

        result = await session.call_tool(
            tool_name,
            arguments=tool_args,
        )

        result_parts = []

        for content_item in result.content:

            if hasattr(
                content_item,
                "text",
            ):
                result_parts.append(
                    content_item.text
                )
            else:
                result_parts.append(
                    str(content_item)
                )

        tool_output = "\n".join(
            result_parts
        )

        # MCP explicitly reports that the tool failed.
        if getattr(
            result,
            "isError",
            False,
        ):

            trace.record(
                tool_name,
                tool_args,
                False,
            )

            print(
                "\n[RepoMind ERROR]"
            )

            print(
                tool_output
            )

            return {
                "ok": False,
                "error": tool_output,
            }

        trace.record(
            tool_name,
            tool_args,
            True,
        )

        print(
            "\n[RepoMind]"
        )

        print(
            tool_output
        )

        return {
            "ok": True,
            "result": tool_output,
        }

    except Exception as exc:

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        trace.record(
            tool_name,
            tool_args,
            False,
        )

        print(
            "\n[RepoMind EXCEPTION]"
        )

        print(
            error_message
        )

        return {
            "ok": False,
            "error": error_message,
        }


async def print_agent_trace(trace):
    """Print a summary of observable tool execution."""

    print(
        "\n[Agent trace]"
    )

    print(
        f"  Tool calls this turn: "
        f"{trace.call_count}"
    )

    successful_calls = 0
    failed_calls = 0

    for index, call in enumerate(
        trace.tool_calls,
        start=1,
    ):

        if call["success"]:

            status = "SUCCESS"
            successful_calls += 1

        else:

            status = "ERROR"
            failed_calls += 1

        print(
            f"  {index}. "
            f"{call['tool']} "
            f"-> {status}"
        )

    print(
        f"  Successful: "
        f"{successful_calls}"
    )

    print(
        f"  Failed: "
        f"{failed_calls}"
    )


async def run_agent_turn(
    gemini,
    session,
    gemini_tools,
    contents,
    user_query,
):
    """
    Run one complete agent turn.

    This includes:
    - adding the user message
    - Gemini tool selection
    - MCP tool execution
    - tool result handling
    - multi-step reasoning
    - transient Gemini retry
    - final response
    """

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_query
                )
            ],
        )
    )

    trace = AgentTrace()

    while True:

        if (
            trace.call_count
            >= MAX_TOOL_CALLS_PER_TURN
        ):
            print(
                "\n[Agent error]"
            )

            print(
                "Maximum tool-call limit "
                "reached for this turn."
            )

            await print_agent_trace(
                trace
            )

            return

        response = await generate_with_retry(
            gemini,
            contents,
            gemini_tools,
        )

        function_calls = (
            extract_function_calls(
                response
            )
        )

        # ---------------------------------------------------------
        # Gemini has produced the final response.
        # ---------------------------------------------------------

        if not function_calls:

            if response.candidates:

                model_content = (
                    response.candidates[0].content
                )

                if model_content:

                    contents.append(
                        model_content
                    )

            print(
                "\nAgent:"
            )

            if response.text:

                print(
                    response.text
                )

            else:

                print(
                    "Gemini did not return "
                    "a text response."
                )

            await print_agent_trace(
                trace
            )

            return

        # ---------------------------------------------------------
        # Preserve Gemini's tool-call response.
        # ---------------------------------------------------------

        if response.candidates:

            model_content = (
                response.candidates[0].content
            )

            if model_content:

                contents.append(
                    model_content
                )

        # ---------------------------------------------------------
        # Execute every requested MCP tool.
        # ---------------------------------------------------------

        for function_call in function_calls:

            if (
                trace.call_count
                >= MAX_TOOL_CALLS_PER_TURN
            ):
                print(
                    "\n[Agent error]"
                )

                print(
                    "Maximum tool-call limit "
                    "reached for this turn."
                )

                await print_agent_trace(
                    trace
                )

                return

            tool_result = (
                await execute_tool_call(
                    session,
                    function_call,
                    trace,
                )
            )

            if tool_result["ok"]:

                print(
                    "\n[Tool status] SUCCESS"
                )

            else:

                print(
                    "\n[Tool status] "
                    "ERROR - agent may recover"
                )

            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response=tool_result,
                        )
                    ],
                )
            )


async def main():
    """Start the interactive RepoMind AI agent."""

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY environment "
            "variable is not set."
        )

    gemini = genai.Client(
        api_key=api_key
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "src/repomind/server.py"
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.path.abspath(
                "src"
            ),
        },
    )

    try:

        async with stdio_client(
            server_params
        ) as (read, write):

            async with ClientSession(
                read,
                write,
            ) as session:

                # =================================================
                # 1. Initialize MCP
                # =================================================

                await session.initialize()

                # =================================================
                # 2. Discover RepoMind tools
                # =================================================

                tools_result = (
                    await session.list_tools()
                )

                print(
                    "\nRepoMind AI Agent"
                )

                print(
                    "=" * 60
                )

                print(
                    f"Connected to RepoMind "
                    f"({len(tools_result.tools)} "
                    f"tools available)"
                )

                for tool in tools_result.tools:

                    print(
                        f"  - {tool.name}"
                    )

                print(
                    "=" * 60
                )

                print(
                    "Type 'exit' or 'quit' to stop."
                )

                print()

                # =================================================
                # 3. Convert MCP tools -> Gemini tools
                # =================================================

                declarations = [
                    mcp_tool_to_gemini_tool(
                        tool
                    )
                    for tool in tools_result.tools
                ]

                gemini_tools = [
                    types.Tool(
                        function_declarations=declarations
                    )
                ]

                # =================================================
                # 4. Persistent conversation history
                # =================================================

                contents = []

                # =================================================
                # 5. Interactive user loop
                # =================================================

                while True:

                    try:

                        user_query = input(
                            "You: "
                        ).strip()

                    except (
                        EOFError,
                        KeyboardInterrupt,
                    ):

                        print(
                            "\nGoodbye!"
                        )

                        break

                    if not user_query:
                        continue

                    if user_query.lower() in {
                        "exit",
                        "quit",
                    }:

                        print(
                            "Goodbye!"
                        )

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

                        print(
                            "\n[Agent error]"
                        )

                        print(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

    finally:

        gemini.close()


if __name__ == "__main__":
    asyncio.run(main())
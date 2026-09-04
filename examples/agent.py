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

Your job is to answer repository-specific questions using evidence
returned by RepoMind.

============================================================
GENERAL RULES
============================================================

1. Never invent repository facts.

2. Use RepoMind tools whenever repository-specific information
   is required.

3. Prefer the most direct and specific tool available.

4. Avoid exploratory tool calls when a specialized tool can
   answer the question directly.

5. Do not call list_files unless the user explicitly asks about
   repository structure or you genuinely need to discover paths.

6. Never use a tool simply because it is available.

7. Never execute shell commands.

8. Never modify repository files.

9. Never expose secrets, API keys, credentials, environment
   variables, or sensitive files.

============================================================
TOOL ROUTING POLICY
============================================================

Use search_symbols when the user asks:

- where a class is defined
- where a function is defined
- where a method is defined
- whether a symbol exists
- information about indexed classes/functions/methods/imports

Use find_usages when the user asks:

- who uses a symbol
- where a symbol is called
- where a class/function/method is referenced
- what files reference a symbol

Use get_dependencies when the user asks:

- what a file depends on
- what modules a file imports locally
- local dependencies of a source file
- dependency relationships between repository files

Use search_code when the user asks:

- search for text
- find a particular string
- find a code pattern
- search comments, documentation, or source text

Use analyze_python_file when the user asks:

- what classes a Python file contains
- what functions a Python file contains
- what methods a Python file contains
- what imports a Python file contains
- for structural analysis of a Python file

Use get_git_history when the user asks:

- recent repository commits
- recent project history
- what changed recently across the repository

Use get_git_commit when the user asks:

- details about a specific commit
- files changed by a commit
- what a particular commit changed

Use get_git_file_history when the user asks:

- history of a specific file
- changes to a particular file over time
- commits affecting a particular file

Use index_repository when the user explicitly asks to:

- build the index
- update the index
- re-index the repository

Use index_stats when the user asks:

- how many files are indexed
- how many symbols are indexed
- index statistics
- repository indexing statistics

Use list_files when the user asks:

- what files exist
- repository structure
- directory structure
- available files

Use read_file when the user asks:

- to read a specific file
- to inspect file contents
- to show the contents of a known file

============================================================
MULTI-STEP QUESTIONS
============================================================

For questions requiring multiple pieces of repository information:

1. Identify the minimum tools required.
2. Use specialized tools before broad exploration.
3. Reuse information already obtained in the conversation.
4. Do not repeat successful tool calls unnecessarily.
5. When one tool provides enough information, stop calling tools.

Example:

Question:
"Where is PythonAnalyzer defined and where is it used?"

Preferred sequence:

search_symbols
    ↓
find_usages
    ↓
answer

Question:
"What does server.py depend on and what does the analyzer do?"

Preferred sequence:

get_dependencies
    ↓
analyze_python_file
    ↓
answer

============================================================
TOOL ERRORS
============================================================

Treat tool errors as failed operations.

If a tool fails:

1. Read the error carefully.
2. Determine whether another appropriate tool can recover.
3. Do not repeat the exact same failing call unnecessarily.
4. Never treat an error message as repository evidence.
5. If recovery is impossible, clearly explain the limitation.

============================================================
FINAL ANSWERS
============================================================

Base repository-specific claims on actual RepoMind results.

Prefer concise answers with:

- relevant file paths
- relevant symbols
- line numbers when available
- a short explanation of the evidence

Do not expose hidden model reasoning or internal chain-of-thought.
Only provide the observable tool execution information shown by
the agent trace.
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

            return trace

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

            return trace

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

                return trace    

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

def print_help():
    """Print interactive RepoMind commands."""

    print(
        """
RepoMind commands
-----------------
/help    Show this help message
/tools   Show available MCP tools
/stats   Show repository index statistics
/trace   Show the last agent tool trace
/clear   Clear conversation history
/exit    Exit RepoMind

Anything else is treated as a repository question.
"""
    )

def print_tools(tools):
    """Print available RepoMind MCP tools."""

    print("\nAvailable RepoMind tools")
    print("-" * 40)

    for tool in tools:
        description = (
            tool.description
            or "No description available."
        )

        print(f"\n{tool.name}")
        print(f"  {description}")

    print("-" * 40)

def print_last_trace(trace):
    """Print the most recent agent execution trace."""

    if trace is None or not trace.tool_calls:
        print("\nNo agent trace available.")
        return

    print("\nLast agent trace")
    print("-" * 40)

    for index, call in enumerate(
        trace.tool_calls,
        start=1,
    ):
        status = (
            "SUCCESS"
            if call["success"]
            else "ERROR"
        )

        print(
            f"{index}. "
            f"{call['tool']} -> {status}"
        )

    print(
        f"\nTotal tool calls: "
        f"{trace.call_count}"
    )

async def print_index_stats(session):
    """Display RepoMind index statistics."""

    try:
        result = await session.call_tool(
            "index_stats",
            arguments={},
        )

        result_parts = []

        for content_item in result.content:
            if hasattr(content_item, "text"):
                result_parts.append(
                    content_item.text
                )
            else:
                result_parts.append(
                    str(content_item)
                )

        output = "\n".join(result_parts)

        if getattr(result, "isError", False):
            print("\n[Index stats error]")
            print(output)
            return

        print("\nRepoMind index statistics")
        print("-" * 40)
        print(output)
        print("-" * 40)

    except Exception as exc:
        print("\n[Index stats error]")
        print(
            f"{type(exc).__name__}: {exc}"
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
                last_trace = AgentTrace()

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

                    command = user_query.lower()

                    # =========================================================
                    # Built-in commands
                    # =========================================================

                    if command in {
                        "/exit",
                        "/quit",
                        "exit",
                        "quit",
                    }:
                        print(
                            "Goodbye!"
                        )
                        break

                    if command == "/help":
                        print_help()
                        continue

                    if command == "/tools":
                        print_tools(
                            tools_result.tools
                        )
                        continue

                    if command == "/stats":
                        await print_index_stats(
                            session
                        )
                        continue

                    if command == "/trace":
                        print_last_trace(
                            last_trace
                        )
                        continue

                    if command == "/clear":
                        contents.clear()
                        last_trace = AgentTrace()

                        print(
                            "\nConversation history cleared."
                        )
                        continue

                    # =========================================================
                    # Normal AI question
                    # =========================================================

                    try:

                        last_trace = await run_agent_turn(
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
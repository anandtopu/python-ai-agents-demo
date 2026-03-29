from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from tools import (
    calculator,
    http_get,
    project_search,
    run_shell,
    sandbox_list,
    sandbox_read,
    sandbox_write,
    web_get_text,
)

mcp = FastMCP("Python AI Agents Demo")


@mcp.tool()
def tool_calculator(expression: str) -> dict[str, Any]:
    return calculator(expression=expression)


@mcp.tool()
def tool_http_get(url: str, timeout_s: int = 10) -> dict[str, Any]:
    return http_get(url=url, timeout_s=timeout_s)


@mcp.tool()
def tool_web_get_text(url: str, timeout_s: int = 15, max_chars: int = 12000) -> dict[str, Any]:
    return web_get_text(url=url, timeout_s=timeout_s, max_chars=max_chars)


@mcp.tool()
def tool_sandbox_list(path: str = ".", recursive: bool = False, max_entries: int = 200) -> dict[str, Any]:
    return sandbox_list(path=path, recursive=recursive, max_entries=max_entries)


@mcp.tool()
def tool_sandbox_read(path: str, max_chars: int = 12000) -> dict[str, Any]:
    return sandbox_read(path=path, max_chars=max_chars)


@mcp.tool()
def tool_sandbox_write(
    path: str,
    content: str,
    append: bool = False,
    create_dirs: bool = True,
) -> dict[str, Any]:
    return sandbox_write(path=path, content=content, append=append, create_dirs=create_dirs)


@mcp.tool()
def tool_project_search(
    query: str,
    root: str = ".",
    max_results: int = 50,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    return project_search(query=query, root=root, max_results=max_results, case_sensitive=case_sensitive)


@mcp.tool()
def tool_run_shell(
    command: str,
    args: list[str] | None = None,
    timeout_s: int = 10,
    max_output_chars: int = 8000,
) -> dict[str, Any]:
    return run_shell(command=command, args=args or [], timeout_s=timeout_s, max_output_chars=max_output_chars)


@mcp.resource("project://readme")
def resource_readme() -> str:
    try:
        return open("README.md", "r", encoding="utf-8").read()
    except OSError as e:
        return f"Error reading README.md: {e}"


@mcp.resource("sandbox://{path}")
def resource_sandbox_file(path: str) -> str:
    data = sandbox_read(path=path)
    if "error" in data:
        return json.dumps(data)
    return str(data.get("content", ""))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import ast
import html
from html.parser import HTMLParser
import operator as op
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Any

import requests


class ToolError(RuntimeError):
    pass


_SANDBOX_DIR = Path("workspace_sandbox")
_MAX_FILE_BYTES = 1_000_000


def _resolve_in_sandbox(rel_path: str) -> Path:
    root = _SANDBOX_DIR.resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise ToolError("Path escapes sandbox")
    return target


def sandbox_list(path: str = ".", recursive: bool = False, max_entries: int = 200) -> dict[str, Any]:
    try:
        root = _SANDBOX_DIR
        root.mkdir(parents=True, exist_ok=True)
        base = _resolve_in_sandbox(path)
        if not base.exists():
            return {"path": path, "error": "Not found"}

        entries: list[dict[str, Any]] = []
        if base.is_file():
            entries.append({"path": str(Path(path)), "type": "file", "size": base.stat().st_size})
        else:
            if recursive:
                it = base.rglob("*")
            else:
                it = base.glob("*")

            for p in it:
                if len(entries) >= max_entries:
                    break
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    continue
                entries.append(
                    {
                        "path": rel.as_posix(),
                        "type": "dir" if p.is_dir() else "file",
                        "size": p.stat().st_size if p.is_file() else None,
                    }
                )

        return {"root": root.as_posix(), "path": path, "entries": entries, "truncated": len(entries) >= max_entries}
    except (OSError, ToolError) as e:
        return {"path": path, "error": str(e)}


def sandbox_read(path: str, max_chars: int = 12000) -> dict[str, Any]:
    try:
        _SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        p = _resolve_in_sandbox(path)
        if not p.exists() or not p.is_file():
            return {"path": path, "error": "Not found"}
        size = p.stat().st_size
        if size > _MAX_FILE_BYTES:
            return {"path": path, "error": "File too large"}
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars]
        return {"path": path, "size": size, "content": text}
    except (OSError, ToolError) as e:
        return {"path": path, "error": str(e)}


def sandbox_write(path: str, content: str, append: bool = False, create_dirs: bool = True) -> dict[str, Any]:
    try:
        _SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        p = _resolve_in_sandbox(path)
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8", newline="\n") as f:
            f.write(content)
        return {"path": path, "bytes_written": len(content.encode("utf-8")), "append": append}
    except (OSError, ToolError) as e:
        return {"path": path, "error": str(e)}


def project_search(
    query: str,
    root: str = ".",
    max_results: int = 50,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    try:
        root_path = Path(root).resolve()
        if not root_path.exists() or not root_path.is_dir():
            return {"root": root, "error": "Invalid root"}

        needle = query if case_sensitive else query.lower()
        results: list[dict[str, Any]] = []

        def should_skip(p: Path) -> bool:
            parts = {part.lower() for part in p.parts}
            return bool(parts & {".venv", "__pycache__", "sessions", "workspace_sandbox"})

        for p in root_path.rglob("*"):
            if len(results) >= max_results:
                break
            if should_skip(p):
                continue
            if not p.is_file():
                continue
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            hay = text if case_sensitive else text.lower()
            if needle not in hay:
                continue

            line_no = 0
            for line in text.splitlines():
                if len(results) >= max_results:
                    break
                line_no += 1
                cmp = line if case_sensitive else line.lower()
                if needle in cmp:
                    try:
                        rel = p.relative_to(root_path)
                        rel_str = rel.as_posix()
                    except ValueError:
                        rel_str = str(p)
                    results.append({"file": rel_str, "line": line_no, "text": line[:300]})

        return {"root": str(root_path), "query": query, "results": results, "truncated": len(results) >= max_results}
    except Exception as e:
        return {"root": root, "error": str(e)}


_ALLOWED_SHELL_COMMANDS = {"python", "python.exe", "pip", "pip.exe", "git", "git.exe"}


def run_shell(command: str, args: list[str] | None = None, timeout_s: int = 10, max_output_chars: int = 8000) -> dict[str, Any]:
    try:
        cmd = command.strip()
        if cmd not in _ALLOWED_SHELL_COMMANDS:
            return {"error": f"Command not allowed: {cmd}", "allowed": sorted(_ALLOWED_SHELL_COMMANDS)}
        argv = [cmd] + (args or [])
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars]
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars]
        return {
            "command": cmd,
            "args": args or [],
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "args": args or [], "error": "Timed out"}
    except Exception as e:
        return {"command": command, "args": args or [], "error": str(e)}


_ALLOWED_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_ALLOWED_BIN_OPS[type(node.op)](left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        operand = _eval_ast(node.operand)
        return float(_ALLOWED_UNARY_OPS[type(node.op)](operand))

    raise ToolError("Unsupported expression")


def calculator(expression: str) -> dict[str, Any]:
    """Safely evaluate a basic arithmetic expression."""
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _eval_ast(parsed)
        return {"expression": expression, "result": value}
    except (SyntaxError, ValueError, ToolError) as e:
        return {"expression": expression, "error": str(e)}


def http_get(url: str, timeout_s: int = 10) -> dict[str, Any]:
    """Fetch a URL (GET). Returns a short text preview."""
    try:
        resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "first-agent/0.1"})
        content_type = resp.headers.get("content-type", "")
        text = resp.text
        preview = text[:2000]
        return {
            "url": url,
            "status": resp.status_code,
            "content_type": content_type,
            "preview": preview,
        }
    except requests.RequestException as e:
        return {"url": url, "error": str(e)}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "br", "div", "li", "section", "article", "header", "footer", "h1", "h2", "h3"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        joined = " ".join(self._chunks)
        joined = joined.replace(" \n ", "\n").replace("\n ", "\n").replace(" \n", "\n")
        lines = [line.strip() for line in joined.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def web_get_text(url: str, timeout_s: int = 15, max_chars: int = 12000) -> dict[str, Any]:
    """Fetch a web page and extract readable text (best-effort)."""
    try:
        resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "first-agent/0.1"})
        content_type = resp.headers.get("content-type", "")
        raw = resp.text or ""

        extractor = _TextExtractor()
        extractor.feed(raw)
        text = extractor.get_text()
        text = html.unescape(text)
        if len(text) > max_chars:
            text = text[:max_chars]

        return {
            "url": url,
            "status": resp.status_code,
            "content_type": content_type,
            "text": text,
        }
    except requests.RequestException as e:
        return {"url": url, "error": str(e)}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="calculator",
        description="Evaluate basic arithmetic. Supports + - * / // % ** and parentheses.",
        json_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="http_get",
        description="HTTP GET a URL and return status + a short text preview.",
        json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="web_get_text",
        description="Fetch a web page and return extracted readable text for summarization.",
        json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60},
                "max_chars": {"type": "integer", "minimum": 500, "maximum": 50000},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="sandbox_list",
        description="List files and directories under the local workspace_sandbox directory.",
        json_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="sandbox_read",
        description="Read a text file from workspace_sandbox.",
        json_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="sandbox_write",
        description="Write a text file to workspace_sandbox.",
        json_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
                "create_dirs": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="project_search",
        description="Search for a text query under a project directory, returning matching lines.",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
                "case_sensitive": {"type": "boolean"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="run_shell",
        description="Run a restricted local command (allowlist: python, pip, git) with timeout and captured output.",
        json_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60},
                "max_output_chars": {"type": "integer", "minimum": 1000, "maximum": 50000},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
]


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "calculator":
        return calculator(expression=str(arguments.get("expression", "")))

    if name == "http_get":
        timeout_s = int(arguments.get("timeout_s", 10))
        return http_get(url=str(arguments.get("url", "")), timeout_s=timeout_s)

    if name == "web_get_text":
        timeout_s = int(arguments.get("timeout_s", 15))
        max_chars = int(arguments.get("max_chars", 12000))
        return web_get_text(
            url=str(arguments.get("url", "")),
            timeout_s=timeout_s,
            max_chars=max_chars,
        )

    if name == "sandbox_list":
        return sandbox_list(
            path=str(arguments.get("path", ".")),
            recursive=bool(arguments.get("recursive", False)),
            max_entries=int(arguments.get("max_entries", 200)),
        )

    if name == "sandbox_read":
        return sandbox_read(
            path=str(arguments.get("path", "")),
            max_chars=int(arguments.get("max_chars", 12000)),
        )

    if name == "sandbox_write":
        return sandbox_write(
            path=str(arguments.get("path", "")),
            content=str(arguments.get("content", "")),
            append=bool(arguments.get("append", False)),
            create_dirs=bool(arguments.get("create_dirs", True)),
        )

    if name == "project_search":
        return project_search(
            query=str(arguments.get("query", "")),
            root=str(arguments.get("root", ".")),
            max_results=int(arguments.get("max_results", 50)),
            case_sensitive=bool(arguments.get("case_sensitive", False)),
        )

    if name == "run_shell":
        raw_args = arguments.get("args", [])
        args = [str(x) for x in raw_args] if isinstance(raw_args, list) else []
        return run_shell(
            command=str(arguments.get("command", "")),
            args=args,
            timeout_s=int(arguments.get("timeout_s", 10)),
            max_output_chars=int(arguments.get("max_output_chars", 8000)),
        )

    return {"error": f"Unknown tool: {name}"}

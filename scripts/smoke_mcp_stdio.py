"""Initialize the installed TinyContext MCP server over a real stdio transport."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


EXPECTED_TOOLS = {"save_memories", "recall_memories"}


async def smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="tinycontext-mcp-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.json"
        config_path.write_text(
            json.dumps({"memory_db_path": "memories.db"}),
            encoding="utf-8",
        )
        child_env = os.environ.copy()
        child_env.update(
            {
                "MCP_TRANSPORT": "stdio",
                "TINYCONTEXT_CONFIG_PATH": str(config_path),
            }
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tinycontext.cli", "mcp"],
            env=child_env,
        )
        async with asyncio.timeout(30):
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    names = {tool.name for tool in response.tools}
                    if names != EXPECTED_TOOLS:
                        raise RuntimeError(
                            f"unexpected MCP tools: {sorted(names)}; "
                            f"expected {sorted(EXPECTED_TOOLS)}"
                        )


def main() -> int:
    asyncio.run(smoke())
    print("MCP stdio smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

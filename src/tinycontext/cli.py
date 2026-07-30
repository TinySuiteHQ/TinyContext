"""The `tinycontext` console command."""

from __future__ import annotations

import argparse
import os
import sys


def _server_dependency_error(_exc: ModuleNotFoundError) -> int:
    print(
        "TinyContext server dependencies are not installed.\n"
        'Install them with: pip install "tinysuite-context[server]"',
        file=sys.stderr,
    )
    return 2


def _run_mcp_stdio() -> int:
    try:
        from tinycontext.servers.mcp_server import main as mcp_main
    except ModuleNotFoundError as exc:
        return _server_dependency_error(exc)
    mcp_main()
    return 0


def _run_serve() -> int:
    os.environ.setdefault("MCP_TRANSPORT", "streamable-http")
    try:
        from tinycontext.servers.mcp_server import main as mcp_main
    except ModuleNotFoundError as exc:
        return _server_dependency_error(exc)
    mcp_main()
    return 0


def _run_doctor() -> int:
    from tinycontext.doctor import run as doctor_run

    return doctor_run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinycontext",
        description="TinyContext memory library and MCP server.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("mcp", help="Run the stdio MCP server.")
    subparsers.add_parser("serve", help="Run the Streamable HTTP MCP server.")
    subparsers.add_parser(
        "doctor",
        help="Check configuration, storage, and SQLite readiness.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        sys.exit(_run_serve())
    if args.command == "doctor":
        sys.exit(_run_doctor())
    sys.exit(_run_mcp_stdio())


if __name__ == "__main__":
    main()

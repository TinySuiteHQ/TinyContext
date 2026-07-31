from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from tinycontext.cli import _run_mcp_stdio, build_parser, main


class CliParserTests(unittest.TestCase):
    def test_no_subcommand_defaults_to_mcp(self) -> None:
        self.assertIsNone(build_parser().parse_args([]).command)

    def test_commands_parse(self) -> None:
        self.assertEqual(build_parser().parse_args(["mcp"]).command, "mcp")
        self.assertEqual(build_parser().parse_args(["serve"]).command, "serve")
        self.assertEqual(build_parser().parse_args(["doctor"]).command, "doctor")


class CliDispatchTests(unittest.TestCase):
    def test_no_args_runs_mcp_stdio(self) -> None:
        with patch("tinycontext.cli._run_mcp_stdio", return_value=0) as run_mcp:
            with self.assertRaises(SystemExit) as context:
                main([])
        run_mcp.assert_called_once()
        self.assertEqual(context.exception.code, 0)

    def test_mcp_runs_stdio(self) -> None:
        with patch("tinycontext.cli._run_mcp_stdio", return_value=0) as run_mcp:
            with self.assertRaises(SystemExit):
                main(["mcp"])
        run_mcp.assert_called_once()

    def test_serve_dispatches(self) -> None:
        with patch("tinycontext.cli._run_serve", return_value=0) as run_serve:
            with self.assertRaises(SystemExit):
                main(["serve"])
        run_serve.assert_called_once()

    def test_doctor_dispatches_status(self) -> None:
        with patch("tinycontext.cli._run_doctor", return_value=1) as run_doctor:
            with self.assertRaises(SystemExit) as context:
                main(["doctor"])
        run_doctor.assert_called_once()
        self.assertEqual(context.exception.code, 1)

    def test_missing_server_extra_has_actionable_error(self) -> None:
        stderr = StringIO()
        with patch.dict(
            "sys.modules",
            {"tinycontext.servers.mcp_server": None},
        ), redirect_stderr(stderr):
            self.assertEqual(_run_mcp_stdio(), 2)
        self.assertIn('pip install "tinysuite-context[server]"', stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

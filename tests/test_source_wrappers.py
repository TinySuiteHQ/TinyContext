from __future__ import annotations

import unittest

from servers.fastapi_server import app as wrapper_app
from servers.mcp_server import main as wrapper_main
from tinycontext.servers.fastapi_server import app as package_app
from tinycontext.servers.mcp_server import main as package_main


class SourceWrapperTests(unittest.TestCase):
    def test_fastapi_wrapper_reexports_packaged_app(self) -> None:
        self.assertIs(wrapper_app, package_app)

    def test_mcp_wrapper_reexports_packaged_main(self) -> None:
        self.assertIs(wrapper_main, package_main)


if __name__ == "__main__":
    unittest.main()

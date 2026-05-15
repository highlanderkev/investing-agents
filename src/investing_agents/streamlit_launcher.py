"""Launch the Streamlit UI for interacting with investing agents."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Entry-point script for `investing-agents-ui`."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(stcli.main())

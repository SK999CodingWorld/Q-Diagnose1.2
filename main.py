"""Backward-compatible ASGI entry point; new application wiring lives in qdiagnose.app."""

from qdiagnose.api.legacy_routes import *  # noqa: F401,F403
from qdiagnose.app import create_app

app = create_app()

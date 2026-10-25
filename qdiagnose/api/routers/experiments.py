"""Experiment execution, evaluation, prediction, and explanation endpoints."""

from qdiagnose.api.legacy_routes import router as legacy_router
from qdiagnose.api.routers._adapter import build_compatibility_router

router = build_compatibility_router(
    legacy_router,
    "experiments",
    lambda path: path.startswith("/experiments") or path.startswith("/explain/") or path == "/predict",
)

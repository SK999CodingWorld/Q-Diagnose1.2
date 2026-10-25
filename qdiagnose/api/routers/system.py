"""Health, model catalogue, backend, and benchmark metadata endpoints."""

from qdiagnose.api.legacy_routes import router as legacy_router
from qdiagnose.api.routers._adapter import build_compatibility_router

router = build_compatibility_router(
    legacy_router,
    "system",
    lambda path: path in {"/", "/health", "/models", "/backends", "/comparisons"},
)

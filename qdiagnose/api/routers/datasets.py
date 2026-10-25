"""Dataset catalogue, detail, and upload endpoints."""

from qdiagnose.api.legacy_routes import router as legacy_router
from qdiagnose.api.routers._adapter import build_compatibility_router

router = build_compatibility_router(
    legacy_router,
    "datasets",
    lambda path: path == "/datasets" or path.startswith("/datasets/"),
)

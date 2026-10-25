"""Lab report intake, review, risk prediction, and export endpoints."""

from qdiagnose.api.legacy_routes import router as legacy_router
from qdiagnose.api.routers._adapter import build_compatibility_router

router = build_compatibility_router(
    legacy_router,
    "clinical reports",
    lambda path: path.startswith("/lab-reports") or path.startswith("/reports/") or path.startswith("/risk/"),
)

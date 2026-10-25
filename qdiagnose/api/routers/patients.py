"""Patient records and longitudinal trend endpoints."""

from qdiagnose.api.legacy_routes import router as legacy_router
from qdiagnose.api.routers._adapter import build_compatibility_router

router = build_compatibility_router(
    legacy_router,
    "patients",
    lambda path: path == "/patients" or path.startswith("/patients/"),
)

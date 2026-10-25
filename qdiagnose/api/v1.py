"""Versioned API compatibility surface.

The versioned router adapts the established endpoint handlers under ``/api/v1``.
The original unversioned and ``/api`` paths remain available for existing clients.
New route modules can be included here without changing the application factory.
"""

from fastapi import APIRouter
from qdiagnose.api.routers import clinical_reports, datasets, experiments, patients, system


router = APIRouter(prefix="/api/v1", tags=["v1"])
router.include_router(system.router)
router.include_router(datasets.router)
router.include_router(experiments.router)
router.include_router(patients.router)
router.include_router(clinical_reports.router)

"""FastAPI application assembly, middleware, and process-level error handling."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from qdiagnose.api.legacy_routes import router as compatibility_router
from qdiagnose.api.v1 import router as v1_router
from qdiagnose.settings import settings
from qdiagnose.security import authenticate_request


# Use the operating system's verified root CA store on Windows. This keeps
# HTTPS verification enabled while honoring enterprise/local trusted roots.
if os.name == "nt":
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logging.getLogger("qdiagnose.api").warning(
            "truststore is unavailable; HTTPS requests use Python's bundled CA certificates."
        )


logger = logging.getLogger("qdiagnose.api")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize persistent services once per application process."""
    from qdiagnose.api.legacy_routes import init_default_dataset

    init_default_dataset()
    yield


def create_app() -> FastAPI:
    """Build the HTTP application with explicit API and static-file boundaries."""
    app = FastAPI(
        title="Q-Diagnose API",
        description="Hybrid Quantum-Classical AI Platform for Early Disease Risk Detection",
        version="1.0.0",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        try:
            auth_error = await authenticate_request(request)
            if auth_error is not None:
                response = auth_error
                response.headers["X-Request-ID"] = request_id
                return response
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request failure request_id=%s", request_id)
            response = JSONResponse(
                status_code=500,
                content={"detail": "An unexpected server error occurred.", "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(compatibility_router)
    app.include_router(v1_router)

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_redirect():
        return RedirectResponse(url="/")

    @app.get("/login", include_in_schema=False)
    async def login_page():
        """Serve the frontend entry point for the direct login-page URL."""
        index_file = settings.frontend_dist_dir / "index.html"
        if not index_file.is_file():
            return JSONResponse(status_code=404, content={"detail": "Frontend is not built."})
        return FileResponse(index_file)

    if settings.frontend_dist_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(settings.frontend_dist_dir), html=True),
            name="frontend",
        )

    return app

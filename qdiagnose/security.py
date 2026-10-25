"""Request authentication for the single-owner Supabase deployment mode."""

from __future__ import annotations

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from qdiagnose.settings import settings


PUBLIC_GET_PATHS = {
    "/health",
    "/api/health",
    "/api/v1/health",
    "/api",
    "/api/models",
    "/api/v1/models",
    "/api/backends",
    "/api/v1/backends",
    "/datasets/catalog",
    "/api/datasets/catalog",
    "/api/v1/datasets/catalog",
}


def _is_public_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return True
    if request.method == "GET" and request.url.path in PUBLIC_GET_PATHS:
        return True
    path = request.url.path
    if path in {"/", "/login", "/dashboard", "/favicon.ico"}:
        return True
    # Keep the built frontend public while requiring auth for both the versioned
    # API and its backwards-compatible route aliases.
    if path.startswith("/api/"):
        return False
    return not path.startswith((
        "/datasets", "/experiments", "/predict", "/patients",
        "/lab-reports", "/reports", "/risk",
    ))


async def authenticate_request(request: Request):
    """Return an auth error response or None; local mode is loopback-only."""
    if settings.auth_mode == "local" or _is_public_request(request):
        return None

    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return JSONResponse(status_code=401, content={"detail": "Sign in to access this resource."})

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(
                f"{settings.supabase_url}/auth/v1/user",
                headers={
                    "apikey": settings.supabase_publishable_key or "",
                    "Authorization": f"Bearer {token.strip()}",
                },
            )
    except httpx.HTTPError:
        return JSONResponse(status_code=503, content={"detail": "Authentication service is temporarily unavailable."})

    if response.status_code == 401 or response.status_code == 403:
        return JSONResponse(status_code=401, content={"detail": "Your session is invalid or expired. Sign in again."})
    if response.status_code != 200:
        return JSONResponse(status_code=503, content={"detail": "Could not verify the sign-in session."})

    try:
        user_id = response.json().get("id")
    except (ValueError, AttributeError):
        return JSONResponse(status_code=503, content={"detail": "Authentication returned an invalid response."})
    if not user_id:
        return JSONResponse(status_code=401, content={"detail": "The sign-in session has no user identity."})
    if user_id != settings.supabase_allowed_user_id:
        return JSONResponse(status_code=403, content={"detail": "This account is not authorized for this installation."})

    request.state.user_id = user_id
    return None

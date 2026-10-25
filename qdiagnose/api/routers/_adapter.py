"""Utilities for mounting established handlers on the versioned API."""

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.routing import APIRoute


def canonical_path(path: str) -> str:
    """Strip the legacy /api alias so old handlers map to one v1 resource path."""
    if path == "/api":
        return "/"
    return path[4:] if path.startswith("/api/") else path


def build_compatibility_router(legacy_router: APIRouter, resource: str, accepts: Callable[[str], bool]) -> APIRouter:
    """Expose matching legacy endpoints under a resource-specific v1 router."""
    router = APIRouter(tags=[resource])
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for route in legacy_router.routes:
        if not isinstance(route, APIRoute):
            continue
        path = canonical_path(route.path)
        if not accepts(path):
            continue
        methods = tuple(sorted(route.methods or ()))
        key = (path, methods)
        if key in seen:
            continue
        seen.add(key)
        router.add_api_route(
            path,
            route.endpoint,
            methods=list(methods),
            name=f"v1_{resource}_{route.name}",
            response_model=route.response_model,
            status_code=route.status_code,
            responses=route.responses,
            summary=route.summary,
            description=route.description,
            tags=[resource, "v1"],
            deprecated=route.deprecated,
        )
    return router

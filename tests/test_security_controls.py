from types import SimpleNamespace

from fastapi.responses import JSONResponse
from starlette.requests import Request

from qdiagnose import security
from qdiagnose.settings import Settings


def run_without_event_loop(coroutine):
    """Drive these immediately-resolving auth coroutines without Windows pipes."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Test auth coroutine unexpectedly required an event loop")


def make_request(path: str, method: str = "GET", authorization: str | None = None) -> Request:
    headers = []
    if authorization:
        headers.append((b"authorization", authorization.encode("ascii")))
    return Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("localhost", 8000),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
    })


def test_private_api_requires_supabase_session(monkeypatch):
    monkeypatch.setattr(security, "settings", SimpleNamespace(auth_mode="supabase"))
    response = run_without_event_loop(security.authenticate_request(make_request("/api/v1/patients")))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 401


def test_public_catalog_and_frontend_assets_remain_available(monkeypatch):
    monkeypatch.setattr(security, "settings", SimpleNamespace(auth_mode="supabase"))
    assert run_without_event_loop(security.authenticate_request(make_request("/api/v1/datasets/catalog"))) is None
    assert run_without_event_loop(security.authenticate_request(make_request("/assets/app.js"))) is None


def test_supabase_identity_is_allowlisted(monkeypatch):
    monkeypatch.setattr(security, "settings", SimpleNamespace(
        auth_mode="supabase",
        supabase_url="https://project.supabase.co",
        supabase_publishable_key="publishable",
        supabase_allowed_user_id="allowed-user",
    ))

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"id": "allowed-user"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(security.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    request = make_request("/api/v1/patients", authorization="Bearer test-token")
    assert run_without_event_loop(security.authenticate_request(request)) is None
    assert request.state.user_id == "allowed-user"


def test_nonlocal_host_requires_supabase_auth(monkeypatch):
    monkeypatch.setenv("QDIAGNOSE_HOST", "0.0.0.0")
    monkeypatch.setenv("QDIAGNOSE_AUTH_MODE", "local")
    try:
        Settings.from_env()
    except ValueError as error:
        assert "requires QDIAGNOSE_AUTH_MODE=supabase" in str(error)
    else:
        raise AssertionError("Network binding must be rejected while auth is disabled")

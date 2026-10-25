# Maintainer guide

## Repository map

```text
qdiagnose/
  api/                 FastAPI assembly, resource routers, schemas, compatibility handlers
  domain/              Scientific pipeline, model implementations, analysis, persistence
frontend/src/
  api/                 HTTP client and resource-level API functions
  components/          Cross-page feedback and error boundary
  state/               Shared dashboard state providers
  App.jsx              Dashboard composition and page-level workflows
data/samples/          Synthetic examples, explicitly marked as non-clinical
docs/                  API, architecture, limitations, model cards, report
results/               Recorded benchmark and study output
tests/                 Unit and API integration tests
uploads/               Local user uploads (ignored by Git)
```

## Backend boundaries

- `qdiagnose/domain/` owns model construction, preprocessing, evaluation, report analysis, and SQLite access. It should not depend on FastAPI request or response objects.
- `qdiagnose/api/schemas.py` validates request payloads.
- `qdiagnose/api/routers/` groups the public API by resource. The current v1 adapter reuses stable handlers from `legacy_routes.py` so existing clients keep working while routes are migrated.
- `qdiagnose/app.py` assembles middleware, lifecycle, error responses, routers, and the optional production frontend mount.
- Root-level `database.py`, `models.py`, `pipeline.py`, `evaluate.py`, `explain.py`, and `lab_analyzer.py` are compatibility shims for older scripts. New application code should import `qdiagnose.domain` modules directly.
- `main.py` is the ASGI compatibility entry point. Use `qdiagnose.app:create_app` when embedding the service.

## Frontend boundaries

- `frontend/src/api/client.js` is the only transport layer. It maps old route paths onto `/api/v1`, handles HTTP and network errors, and publishes a shared error notice.
- `frontend/src/api/` contains resource-specific API functions.
- `frontend/src/state/DashboardContext.jsx` owns shared dataset and experiment state, including cancellation when a dataset changes.
- `frontend/src/components/` contains reusable app-wide UI elements. Put new dashboard features in page-level modules instead of adding more workflows to `App.jsx`.

## Configuration

Backend settings use `QDIAGNOSE_` environment variables and load a local root `.env` file without overriding variables already set by the process. Start from `.env.example`. Paths can be absolute or relative to the repository root. `QDIAGNOSE_HOST` defaults to loopback; configure a public bind only behind an appropriately secured deployment.

Vite reads `frontend/.env`; `VITE_API_BASE_URL` optionally sets the API origin. Leave it unset for a same-origin production deployment.

## Local development

1. Install Python packages with `python -m pip install -r requirements.txt`.
2. Install the web packages with `cd frontend; npm ci`.
3. Build the dashboard with `npm run build` from `frontend/`.
4. Start the full local app with `python run_app.py`. The service uses the host and port from backend settings.
5. For frontend iteration, start Vite with `npm run dev` and set `VITE_API_BASE_URL` only when the API is not on `127.0.0.1:8000`.

The versioned API is documented at `/docs` and is rooted at `/api/v1`. Existing unversioned and `/api` routes remain available during migration.

## Error handling

Frontend requests use the shared API client; non-success HTTP responses and network failures raise `ApiError` and appear in the app-wide dismissible notice. Render failures are caught by `ErrorBoundary`. Backend unexpected request failures are logged with a request ID and return a generic response containing that ID. Preserve FastAPI's specific validation and not-found responses when adding routes; avoid returning exception tracebacks to clients.

## Tests and research runs

- Run the test suite from the repository root with `python -m pytest`.
- Research scripts remain root-level command entry points for compatibility. They should write generated studies into `results/` and resolve all project paths from `Path(__file__)` or `qdiagnose.settings`.
- Sample reports and cohort files are synthetic demonstrations, not clinical evidence.

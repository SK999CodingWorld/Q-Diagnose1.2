"""Environment-backed application configuration.

All paths are resolved from the project root so startup does not depend on the
shell's current directory. Environment variables use the ``QDIAGNOSE_`` prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _csv_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return tuple(value.strip().rstrip("/") for value in raw_value.split(",") if value.strip())


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries without overriding process environment."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        for line_number, raw_line in enumerate(env_file.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or not key.strip():
                raise ValueError(f"Invalid .env entry at line {line_number}")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError as error:
        raise RuntimeError(f"Unable to read configuration file {env_file}") from error


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    database_path: Path
    upload_dir: Path
    frontend_dist_dir: Path
    host: str
    port: int
    cors_origins: tuple[str, ...]
    auth_mode: str
    supabase_url: str | None
    supabase_publishable_key: str | None
    supabase_allowed_user_id: str | None
    max_report_file_bytes: int
    max_report_files: int
    max_report_total_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            port = int(os.getenv("QDIAGNOSE_PORT", "8000"))
        except ValueError as error:
            raise ValueError("QDIAGNOSE_PORT must be an integer") from error
        if not 1 <= port <= 65535:
            raise ValueError("QDIAGNOSE_PORT must be between 1 and 65535")

        auth_mode = os.getenv("QDIAGNOSE_AUTH_MODE", "local").strip().lower()
        if auth_mode not in {"local", "supabase"}:
            raise ValueError("QDIAGNOSE_AUTH_MODE must be 'local' or 'supabase'")
        host = os.getenv("QDIAGNOSE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        supabase_url = os.getenv("SUPABASE_URL", "https://yqtcvwuhtcdacyjoyxuj.supabase.co").strip().rstrip("/")
        supabase_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip() or None
        allowed_user_id = os.getenv("QDIAGNOSE_ALLOWED_USER_ID", "").strip() or None
        if auth_mode == "supabase" and not (supabase_url and supabase_key and allowed_user_id):
            raise ValueError(
                "Supabase auth requires SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, "
                "and QDIAGNOSE_ALLOWED_USER_ID."
            )
        if auth_mode == "supabase" and not supabase_url.startswith("https://"):
            raise ValueError("SUPABASE_URL must use HTTPS.")
        if host not in {"127.0.0.1", "localhost", "::1"} and auth_mode != "supabase":
            raise ValueError("Non-local network binding requires QDIAGNOSE_AUTH_MODE=supabase.")

        def positive_int(name: str, default: int) -> int:
            try:
                value = int(os.getenv(name, str(default)))
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error
            if value < 1:
                raise ValueError(f"{name} must be positive")
            return value

        return cls(
            database_path=_path_from_env("QDIAGNOSE_DATABASE_PATH", PROJECT_ROOT / "qdiagnose.db"),
            upload_dir=_path_from_env("QDIAGNOSE_UPLOAD_DIR", PROJECT_ROOT / "uploads"),
            frontend_dist_dir=_path_from_env("QDIAGNOSE_FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist"),
            host=host,
            port=port,
            cors_origins=_csv_from_env(
                "QDIAGNOSE_CORS_ORIGINS",
                ("http://127.0.0.1:5173", "http://localhost:5173"),
            ),
            auth_mode=auth_mode,
            supabase_url=supabase_url or None,
            supabase_publishable_key=supabase_key,
            supabase_allowed_user_id=allowed_user_id,
            max_report_file_bytes=positive_int("QDIAGNOSE_MAX_REPORT_FILE_BYTES", 10 * 1024 * 1024),
            max_report_files=positive_int("QDIAGNOSE_MAX_REPORT_FILES", 5),
            max_report_total_bytes=positive_int("QDIAGNOSE_MAX_REPORT_TOTAL_BYTES", 25 * 1024 * 1024),
        )


_load_local_env()
settings = Settings.from_env()

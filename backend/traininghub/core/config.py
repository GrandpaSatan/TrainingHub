from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _default_data_root() -> Path:
    if os.getenv("USER") == "jhernandez":
        return Path("/home/jhernandez/traininghub-data")
    return Path.cwd() / ".traininghub-data"


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_root: Path
    data_root: Path
    database_path: Path
    frontend_dist: Path
    session_cookie_name: str
    session_secret: str
    admin_username: str
    admin_password: str
    worker_python: str
    real_workers_enabled: bool
    default_teacher_model: str


def get_settings() -> Settings:
    data_root = Path(os.getenv("TRAININGHUB_DATA_ROOT", str(_default_data_root()))).expanduser()
    app_root = Path(os.getenv("TRAININGHUB_APP_ROOT", str(Path.cwd()))).expanduser()
    database_path = Path(os.getenv("TRAININGHUB_DATABASE_PATH", str(data_root / "traininghub.sqlite3"))).expanduser()
    frontend_dist = Path(os.getenv("TRAININGHUB_FRONTEND_DIST", str(app_root / "frontend" / "dist"))).expanduser()
    session_secret = os.getenv("TRAININGHUB_SESSION_SECRET")
    if not session_secret:
        secret_file = data_root / "session.secret"
        if secret_file.exists():
            session_secret = secret_file.read_text(encoding="utf-8").strip()
        else:
            session_secret = secrets.token_urlsafe(48)
    return Settings(
        app_name="TrainingHub",
        app_root=app_root,
        data_root=data_root,
        database_path=database_path,
        frontend_dist=frontend_dist,
        session_cookie_name="traininghub_session",
        session_secret=session_secret,
        admin_username=os.getenv("TRAININGHUB_ADMIN_USERNAME", "admin"),
        admin_password=os.getenv("TRAININGHUB_ADMIN_PASSWORD", "traininghub"),
        worker_python=os.getenv("TRAININGHUB_WORKER_PYTHON", "python3"),
        real_workers_enabled=os.getenv("TRAININGHUB_ENABLE_REAL_WORKERS", os.getenv("TRAININGHUB_ENABLE_REAL_TRAINING", "0")) == "1",
        default_teacher_model=os.getenv("TRAININGHUB_DEFAULT_TEACHER_MODEL", "local"),
    )


def ensure_directories(settings: Settings) -> None:
    for path in [
        settings.data_root,
        settings.data_root / "datasets",
        settings.data_root / "jobs",
        settings.data_root / "artifacts",
        settings.data_root / "models",
        settings.data_root / "cleanup",
        settings.data_root / "uploads",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    secret_file = settings.data_root / "session.secret"
    if not secret_file.exists():
        secret_file.write_text(settings.session_secret, encoding="utf-8")
        secret_file.chmod(0o600)

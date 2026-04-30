from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import Cookie, Depends, HTTPException, status

from traininghub.core.config import Settings, get_settings
from traininghub.core.database import connect, row_to_dict
from traininghub.core.security import unsign_value, utc_now


def settings_dependency() -> Settings:
    return get_settings()


def current_user(
    settings: Annotated[Settings, Depends(settings_dependency)],
    traininghub_session: Annotated[str | None, Cookie(alias="traininghub_session")] = None,
) -> dict[str, Any]:
    if not traininghub_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required.")
    session_id = unsign_value(traininghub_session, settings.session_secret)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session.")
    with connect(settings.database_path) as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username, s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
            """,
            (session_id,),
        ).fetchone()
    user = row_to_dict(row)
    if not user or int(user["expires_at"]) <= utc_now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired.")
    return user


def require_child_path(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in [resolved_path, *resolved_path.parents]:
        raise HTTPException(status_code=400, detail="Path is outside the TrainingHub data root.")
    return resolved_path


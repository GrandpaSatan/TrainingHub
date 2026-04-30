from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.core.database import connect
from traininghub.core.security import sign_value, utc_now, verify_password


router = APIRouter(prefix="/api", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
def login(payload: LoginRequest, response: Response, settings: Annotated[Settings, Depends(settings_dependency)]) -> dict[str, Any]:
    with connect(settings.database_path) as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (payload.username,)).fetchone()
        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
        session_id = secrets.token_urlsafe(32)
        expires_at = utc_now() + 60 * 60 * 24 * 7
        conn.execute(
            "INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, user["id"], utc_now(), expires_at),
        )
    response.set_cookie(
        settings.session_cookie_name,
        sign_value(session_id, settings.session_secret),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24 * 7,
    )
    return {"username": user["username"], "expires_at": expires_at}


@router.post("/auth/logout")
def logout(
    response: Response,
    settings: Annotated[Settings, Depends(settings_dependency)],
    traininghub_session: Annotated[str | None, Cookie(alias="traininghub_session")] = None,
) -> dict[str, str]:
    if traininghub_session:
        from traininghub.core.security import unsign_value

        session_id = unsign_value(traininghub_session, settings.session_secret)
        if session_id:
            with connect(settings.database_path) as conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    response.delete_cookie(settings.session_cookie_name)
    return {"status": "logged_out"}


@router.get("/me")
def me(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
    return {"username": user["username"]}

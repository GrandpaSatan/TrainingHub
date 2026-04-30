from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from traininghub.api import artifacts, auth, benchmarks, capability_transfers, cleanup, datasets, hub, inference, jobs, models, system
from traininghub.core.config import ensure_directories, get_settings
from traininghub.core.database import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    ensure_directories(settings)
    init_db(settings)
    yield


app = FastAPI(title="TrainingHub", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://10.0.65.20:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(hub.router)
app.include_router(inference.router)
app.include_router(benchmarks.router)
app.include_router(capability_transfers.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(artifacts.router)
app.include_router(cleanup.router)
app.include_router(system.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


settings = get_settings()
if settings.frontend_dist.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

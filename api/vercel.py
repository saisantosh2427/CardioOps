"""Vercel preserves /api in the request path; mount the existing API there."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.main import create_app
from src.config import Settings
from src.vercel import deployment_settings


def create_vercel_app(settings: Settings | None = None) -> FastAPI:
    api = create_app(settings or deployment_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with api.router.lifespan_context(api):
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api", include_in_schema=False)
    def api_root():
        return RedirectResponse("/api/docs")

    app.mount("/api", api)
    return app


app = create_vercel_app()

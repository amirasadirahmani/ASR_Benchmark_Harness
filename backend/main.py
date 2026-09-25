"""
نقطهٔ ورود سرور — FastAPI app + endpoint وب‌سوکت + سرو فایل‌های frontend.

اجرا:
    python -m backend.main
    # یا برای توسعه با reload:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.websocket_handler import ConnectionHandler
from backend.config.model_config import load_model_configs
from backend.config.settings import get_settings
from backend.storage import ResultsStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# state سراسری سبک — settings، model_configs و store یک‌بار ساخته می‌شوند.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.paths.ensure_dirs()

    model_configs = load_model_configs()

    app.state.settings = settings
    app.state.model_configs = model_configs
    app.state.store = ResultsStore(settings.paths.results_dir)

    enabled_ids = [m.id for m in model_configs if m.enabled]
    logger.info(
        "سرور آماده شد | env=%s | مدل‌های فعال=%s",
        getattr(settings, "environment", "unknown"),
        enabled_ids,
    )
    yield
    logger.info("سرور در حال خاموش‌شدن است")


app = FastAPI(
    title="ASR Benchmark Harness",
    description="بنچمارک زندهٔ مدل‌های تشخیص گفتار فارسی با بیدارباش صوتی",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — برای توسعهٔ محلی UI روی پورت جدا (مثلاً Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ در استقرار واقعی محدود به دامنهٔ خودت کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# WebSocket — قلب سیستم
# ============================================================
@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    store = websocket.app.state.store

    handler = ConnectionHandler(websocket=websocket, settings=settings, store=store)
    await handler.run()


# ============================================================
# REST — تاریخچه و اطلاعات جانبی (بدون نیاز به WebSocket)
# ============================================================
@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> list[dict]:
    """فهرست مدل‌های فعال برای نمایش در UI (چک‌باکس انتخاب مدل)."""
    model_configs = app.state.model_configs
    return [
        {
            "id": m.id,
            "display_name": m.display_name,
            "runtime": m.runtime,
            "enabled": m.enabled,
            "order": m.order,
        }
        for m in model_configs
        if m.enabled
    ]


@app.get("/api/results/recent")
async def recent_results(limit: int = 20) -> list[dict]:
    store: ResultsStore = app.state.store
    return store.list_recent(limit=limit)


@app.get("/api/results/{benchmark_id}")
async def get_result(benchmark_id: str) -> JSONResponse:
    store: ResultsStore = app.state.store
    data = store.load(benchmark_id)
    if data is None:
        return JSONResponse(status_code=404, content={"error": "یافت نشد"})
    return JSONResponse(content=data)


# ============================================================
# سرو کردن UI استاتیک (frontend/) — در انتها mount می‌شود تا
# مسیرهای API بالا اولویت داشته باشند.
# ============================================================
_frontend_dir = Path(get_settings().paths.frontend_dir)
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    logger.warning("پوشهٔ frontend یافت نشد: %s — فقط API فعال است", _frontend_dir)


# ============================================================
# اجرای مستقیم برای توسعه
# ============================================================
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=getattr(settings, "host", "0.0.0.0"),
        port=getattr(settings, "port", 8000),
        reload=False,
        log_level="info",
    )
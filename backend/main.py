"""
نقطهٔ ورود سرور — FastAPI app + endpoint وب‌سوکت + سرو فایل‌های frontend.

اجرا:
    python -m backend.main
    # یا برای توسعه با reload (⚠️ حتماً --host 127.0.0.1؛ پرچم
    # allow_external_bind فقط مسیر بالا را محافظت می‌کند، نه فراخوانی
    # مستقیم uvicorn را):
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.websocket_handler import ConnectionHandler
from backend.audio.audio_utils import AudioArtifact
from backend.benchmark.orchestrator import BenchmarkOrchestrator
from backend.config.model_config import load_model_configs
from backend.config.settings import ensure_offline_env, get_settings
from backend.storage import ResultsStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_boot_settings = get_settings()
ensure_offline_env(strict_socket_guard=_boot_settings.strict_offline_guard)


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

_cors_host = _boot_settings.server.host
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://127.0.0.1:{_boot_settings.server.port}",
        f"http://localhost:{_boot_settings.server.port}",
        f"http://{_cors_host}:{_boot_settings.server.port}",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    store = websocket.app.state.store
    handler = ConnectionHandler(websocket=websocket, settings=settings, store=store)
    await handler.run()


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> list[dict]:
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


@app.get("/api/preflight")
async def preflight() -> dict:
    orchestrator = BenchmarkOrchestrator(settings=app.state.settings)
    try:
        return orchestrator.preflight()
    finally:
        orchestrator.close()


@app.post("/api/benchmark/run")
async def run_benchmark_mode(
    audio: UploadFile = File(..., description="فایل WAV تک‌کاناله ۱۶بیتی"),
    reference_text: str = Form(default=""),
    model_ids: str = Form(default="", description="شناسهٔ مدل‌ها با کاما جدا؛ خالی = همهٔ مدل‌های فعال"),
    runs_per_model: Optional[int] = Form(default=None),
) -> JSONResponse:
    settings = app.state.settings
    store: ResultsStore = app.state.store

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="فایل صوتی خالی است.")

    upload_id = uuid.uuid4().hex[:12]
    tmp_dir = Path(settings.paths.temp_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"benchmark-upload-{upload_id}.wav"
    tmp_path.write_bytes(raw)

    try:
        artifact = AudioArtifact.from_wav(
            tmp_path,
            session_id=f"benchmark-mode-{upload_id}",
            utterance_id=upload_id,
            metadata={
                "source": "benchmark_mode_upload",
                "original_filename": audio.filename,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"فایل صوتی قابل خواندن نیست (فقط WAV PCM16 پشتیبانی می‌شود): {exc}",
        ) from exc

    ids = [m.strip() for m in model_ids.split(",") if m.strip()] or None

    run_settings = settings
    if runs_per_model and runs_per_model > 0:
        run_settings = settings.model_copy(deep=True)
        run_settings.benchmark.runs_per_model = runs_per_model

    orchestrator = BenchmarkOrchestrator(settings=run_settings)
    try:
        report = await orchestrator.run(
            artifact,
            reference_text=(reference_text or "").strip() or None,
            model_ids=ids,
        )
    finally:
        orchestrator.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("حذف فایل موقت آپلود %s ناموفق: %s", tmp_path, exc)

    paths = store.save(report)
    logger.info("نتیجهٔ Benchmark Mode ذخیره شد: %s", paths.json_path)
    return JSONResponse(content=report.to_dict())


_frontend_dir = Path(get_settings().paths.frontend_dir)
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    logger.warning("پوشهٔ frontend یافت نشد: %s — فقط API فعال است", _frontend_dir)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    host = settings.server.host
    if not settings.server.allow_external_bind and host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "server.host=%s رد شد چون allow_external_bind=false است → 127.0.0.1", host,
        )
        host = "127.0.0.1"

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
    )

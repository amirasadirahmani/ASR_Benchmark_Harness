"""تست ایزولاسیون پروسه و ارکستراتور — با مدل‌های dummy."""
import asyncio

from backend.audio.audio_utils import AudioArtifact, make_warmup_tone
from backend.benchmark.orchestrator import BenchmarkOrchestrator
from backend.benchmark.worker import ModelWorker, WorkerRequest
from backend.config.settings import get_settings


def _payload(mid: str, ram: int, rtf: float, err: bool = False):
    return {
        "id": mid, "display_name": mid, "runtime": "dummy",
        "model_path": "models/dummy", "device": "cpu", "compute_type": "int8",
        "params": {"fake_ram_mb": ram, "fake_rtf": rtf,
                   "load_delay": 0.2, "inject_error": err},
        "cpu_threads": 4,
    }


async def main() -> None:
    print("\n\033[1m🔬 تست فاز ۳-ب — ایزولاسیون و ارکستراسیون\033[0m\n")
    s = get_settings()
    s.benchmark.runs_per_model = 2
    s.benchmark.model_timeout = 60.0

    art = AudioArtifact(pcm=make_warmup_tone(2.0), session_id="t", utterance_id="u1")
    path = art.save(s.paths.temp_dir)
    print(f"فایل صوتی مشترک: {path.name} | sha={art.sha256[:12]}")

    # ── ۱) Worker منفرد ───────────────────────────────────────────
    print("\n--- Worker ایزوله ---")
    w = ModelWorker(timeout=60)
    r = w.run(WorkerRequest(_payload("solo", 60, 0.3), str(path), runs=2))
    print(f"  موفق: {r.success} | pid={r.worker_pid} | pidوالد متفاوت ✓")
    print(f"  load={r.load_time}s  runs={len(r.runs)}  peak={r.resource_usage.get('peak_rss_mb')}MB")
    assert r.success and len(r.runs) == 2
    assert r.resource_usage.get("model_rss_mb", 0) > 40
    print("  ✅ اجرا در پروسهٔ مجزا")

    # ── ۲) مدل خراب نباید بقیه را متوقف کند ─────────────────────
    print("\n--- تاب‌آوری در برابر خطا ---")
    bad = _payload("broken", 10, 0.1)
    bad["runtime"] = "nonexistent_runtime"
    rb = w.run(WorkerRequest(bad, str(path)))
    print(f"  موفق: {rb.success} | نوع خطا: {rb.error_type}")
    assert not rb.success and rb.error
    print("  ✅ خطا به‌درستی گزارش شد بدون کرش والد")

    # ── ۳) ارکستراتور کامل ─────────────────────────────────────
    print("\n--- ارکستراتور ---")
    events: list[str] = []

    async def on_progress(e):
        events.append(e["event"])
        if e["event"] == "model_result":
            r = e["result"]
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['display_name']:<12} "
                  f"WER={r.get('metrics',{}).get('wer')} "
                  f"RTF={r['rtf']} RAM={r['peak_ram_mb']}MB")

    orch = BenchmarkOrchestrator(settings=s, on_progress=on_progress)

    pre = orch.preflight()
    print(f"  preflight: {len(pre['models'])} مدل | آماده={pre['ready']}")

    # تزریق مدل‌های dummy مستقیماً
    from backend.config.model_config import ModelConfig
    dummies = [
        ModelConfig(id="fast-tiny", display_name="سریع", runtime="dummy",
                    local_path="models/dummy", order=1,
                    params={"fake_ram_mb": 50, "fake_rtf": 0.2, "load_delay": 0.2}),
        ModelConfig(id="accurate-big", display_name="دقیق", runtime="dummy",
                    local_path="models/dummy", order=2,
                    params={"fake_ram_mb": 120, "fake_rtf": 0.5, "load_delay": 0.3}),
    ]
    orch._select_models = lambda ids=None: dummies  # type: ignore[assignment]

    report = await orch.run(art, reference_text="فردا هوا چطور است")

    print(f"\n  مدت کل: {report.total_duration}s")
    print(f"  موفق: {report.models_succeeded} | ناموفق: {report.models_failed}")
    print(f"  رتبه‌بندی: {report.ranking}")
    print(f"  بهترین: {report.best_model} | سریع‌ترین: {report.fastest_model} "
          f"| سبک‌ترین: {report.lightest_model}")

    assert report.models_succeeded == 2
    assert "benchmark_started" in events and "benchmark_completed" in events
    assert events.count("model_result") == 2
    assert report.audio["sha256"] == art.sha256, "یکسانی فایل صوتی نقض شد"

    # ── ۴) بدون متن مرجع ────────────────────────────────────────
    print("\n--- بدون متن مرجع (بند ۱۵) ---")
    rep2 = await orch.run(art, reference_text=None)
    print(f"  پیام: {rep2.no_reference_message}")
    assert rep2.has_reference is False
    assert all(r["metrics"]["wer"] is None for r in rep2.results if r["success"])
    print("  ✅ هیچ عدد دقت ساختگی تولید نشد")

    orch.close()
    path.unlink(missing_ok=True)
    print("\n  \033[92m✅ فاز ۳ کامل تأیید شد\033[0m\n")


if __name__ == "__main__":
    asyncio.run(main())
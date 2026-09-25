"""تست لایهٔ ذخیره‌سازی نتایج."""
import asyncio
import shutil

from backend.audio.audio_utils import AudioArtifact, make_warmup_tone
from backend.benchmark.orchestrator import BenchmarkOrchestrator
from backend.config.model_config import ModelConfig
from backend.config.settings import get_settings
from backend.storage import ResultsStore


async def main() -> None:
    print("\n\033[1m💾 تست فاز ۴-الف — ذخیره‌سازی\033[0m\n")
    s = get_settings()
    test_dir = s.paths.results_dir / "_test_storage"
    shutil.rmtree(test_dir, ignore_errors=True)

    store = ResultsStore(test_dir)
    print(f"CSV: {store.csv_path}")

    art = AudioArtifact(pcm=make_warmup_tone(2.0), session_id="s1", utterance_id="u1")
    art.save(s.paths.temp_dir)

    orch = BenchmarkOrchestrator(settings=s)
    orch._select_models = lambda ids=None: [
        ModelConfig(id="m1", display_name="مدل یک", runtime="dummy",
                    local_path="models/dummy", order=1,
                    params={"fake_ram_mb": 40, "fake_rtf": 0.2, "load_delay": 0.1}),
    ]
    report = await orch.run(art, reference_text="فردا هوا چطور است")
    paths = store.save(report)

    print(f"✅ JSON نوشته شد: {paths.json_path.name}")
    print(f"✅ CSV به‌روزرسانی شد: {paths.csv_path.exists()}")

    recent = store.list_recent(5)
    assert len(recent) == 1
    print(f"✅ list_recent: {recent[0]['benchmark_id']}")

    loaded = store.load(report.benchmark_id)
    assert loaded and loaded["benchmark_id"] == report.benchmark_id
    print("✅ load() بازیابی کامل موفق")

    import csv as csv_mod
    with paths.csv_path.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 1 and rows[0]["model_id"] == "m1"
    assert rows[0]["wer"] != ""
    print(f"✅ سطر CSV صحیح: wer={rows[0]['wer']} rtf={rows[0]['rtf']}")

    orch.close()
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\n\033[92m✅ فاز ۴-الف (Storage) تأیید شد\033[0m\n")


if __name__ == "__main__":
    asyncio.run(main())
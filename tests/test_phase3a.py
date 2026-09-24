"""تست آداپتورها و مانیتور حافظه — بدون نیاز به مدل واقعی."""
import time

import numpy as np

from backend.asr import available_runtimes, create_adapter_from_payload, discover_adapters
from backend.audio.audio_utils import write_wav, float32_to_bytes
from backend.benchmark import RAMMonitor, get_system_memory
from backend.config.settings import get_settings


def main() -> None:
    print("\n\033[1m🔬 تست فاز ۳-الف\033[0m\n")

    discover_adapters()
    runtimes = available_runtimes()
    print(f"Runtimeهای ثبت‌شده: {runtimes}")
    assert "dummy" in runtimes, "آداپتور dummy ثبت نشد"
    assert "faster_whisper" in runtimes, "آداپتور faster_whisper ثبت نشد"

    # --- ساخت فایل صوتی آزمایشی ---
    s = get_settings()
    sr = 16000
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    sig = 0.3 * np.sin(2 * np.pi * 220 * t)
    wav = s.paths.recordings_dir / "_test_phase3a.wav"
    write_wav(wav, float32_to_bytes(sig), sr)
    print(f"فایل آزمایشی: {wav.name} (۲ ثانیه)")

    # --- مانیتور حافظه ---
    print("\n--- مانیتور منابع ---")
    with RAMMonitor(interval=0.02, keep_samples=True) as mon:
        ballast = np.ones(30 * 1024 * 1024 // 8, dtype=np.float64)  # ~30MB
        mon.mark("ballast_allocated")
        time.sleep(0.2)
        del ballast
        time.sleep(0.1)
    u = mon.usage
    print(f"  baseline : {u.baseline_rss_mb} MB")
    print(f"  peak     : {u.peak_rss_mb} MB")
    print(f"  مدل خالص : {u.model_rss_mb} MB  (باید ≈۳۰ باشد)")
    print(f"  نمونه‌ها  : {u.samples} | CPU میانگین: {u.cpu_percent_avg}%")
    assert u.model_rss_mb > 20, "افزایش حافظه تشخیص داده نشد"
    assert u.samples > 5, "نمونه‌برداری کار نکرد"
    print("  ✅ مانیتور حافظه صحیح")

    # --- آداپتور dummy ---
    print("\n--- آداپتور Dummy ---")
    adapter = create_adapter_from_payload({
        "id": "dummy-test", "runtime": "dummy", "model_path": "models/dummy",
        "device": "cpu", "compute_type": "int8",
        "params": {"fake_ram_mb": 40, "fake_rtf": 0.3, "load_delay": 0.2},
        "cpu_threads": 4,
    })

    with RAMMonitor(interval=0.02) as mon:
        adapter.load()
        mon.mark("loaded")
        out = adapter.transcribe(wav)
    usage = mon.usage

    print(f"  متن       : «{out.text}»")
    print(f"  زبان      : {out.language} ({out.language_probability})")
    print(f"  طول صوت   : {out.audio_duration}s")
    print(f"  استنتاج   : {out.inference_time:.3f}s")
    print(f"  RTF       : {out.inference_time / out.audio_duration:.3f}")
    print(f"  بارگذاری  : {adapter.load_time:.3f}s")
    print(f"  Peak RAM  : {usage.peak_rss_mb} MB (مدل: {usage.model_rss_mb} MB)")

    assert out.text, "متن خالی برگشت"
    assert 0.2 < out.inference_time / out.audio_duration < 0.5, "RTF غیرمنتظره"
    assert usage.model_rss_mb > 25, "مصرف حافظهٔ مدل ثبت نشد"

    # --- تکرارپذیری ---
    out2 = adapter.transcribe(wav)
    assert out.text == out2.text, "خروجی dummy تکرارپذیر نیست"
    print("  ✅ خروجی تکرارپذیر")

    adapter.unload()
    assert not adapter.is_loaded
    print("  ✅ آزادسازی مدل")

    # --- محاسبهٔ سنجه‌ها ---
    print("\n--- ادغام با MetricsEngine ---")
    from backend.core import MetricsEngine
    m = MetricsEngine().evaluate(
        hypothesis=out.text,
        reference="چراغ اتاق را روشن کن",
        audio_duration=out.audio_duration,
        load_time=adapter.load_time,
        inference_time=out.inference_time,
        peak_ram_mb=usage.peak_rss_mb,
        baseline_ram_mb=usage.baseline_rss_mb,
        cpu_percent_avg=usage.cpu_percent_avg,
    )
    print(f"  WER={m.wer}  CER={m.cer}  RTF={m.rtf}  RAM={m.model_ram_mb}MB")
    assert m.rtf is not None and m.model_ram_mb is not None

    wav.unlink(missing_ok=True)
    print(f"\n  \033[92m✅ فاز ۳-الف تأیید شد\033[0m")
    print(f"  حافظهٔ سیستم: {get_system_memory()}\n")


if __name__ == "__main__":
    main()
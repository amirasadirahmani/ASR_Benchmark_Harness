#!/usr/bin/env python3
"""تأیید آمادگی Benchmark و Assistant در حالت آفلاین."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import ensure_offline_env  # noqa: E402
ensure_offline_env(strict_socket_guard=True)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _make_synthetic_wav() -> Path:
    import math
    import struct
    import tempfile
    import wave

    rate = 16000
    seconds = 1.5
    frequency = 220.0
    count = int(rate * seconds)
    samples = [
        int(6000 * math.sin(2 * math.pi * frequency * i / rate))
        for i in range(count)
    ]
    path = Path(tempfile.gettempdir()) / "asrb_offline_selftest.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{count}h", *samples))
    return path


def _wake_ready(settings) -> bool:
    from backend.audio.wake_word import create_wake_word_detector

    requested = settings.wake_word.engine
    try:
        detector = create_wake_word_detector(settings.wake_word)
    except Exception as exc:
        _fail(f"موتور Wake Word ساخته نشد: {exc}")
        return False

    if detector.name != requested:
        _fail(f"موتور «{requested}» Load نشد و به «{detector.name}» fallback کرد.")
        return False

    _ok(f"wakeword/{requested} با موفقیت Load شد")
    return True


def check_files() -> tuple[list[str], list[str], bool]:
    from backend.config.model_config import load_model_configs
    from backend.config.settings import get_settings

    settings = get_settings()
    enabled_ids: list[str] = []
    missing_ids: list[str] = []

    for cfg in load_model_configs(enabled_only=True):
        if cfg.runtime == "dummy":
            continue
        enabled_ids.append(cfg.id)
        path = Path(cfg.local_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists() and any(path.iterdir()):
            _ok(f"{cfg.id} → {path}")
        else:
            missing_ids.append(cfg.id)
            _fail(f"{cfg.id} یافت نشد")

    return enabled_ids, missing_ids, _wake_ready(settings)


def check_benchmark_pipeline(expected_ids: list[str]) -> tuple[bool, list[str]]:
    import asyncio

    from backend.audio.audio_utils import AudioArtifact
    from backend.benchmark.orchestrator import BenchmarkOrchestrator
    from backend.config.settings import get_settings

    artifact = AudioArtifact.from_wav(
        _make_synthetic_wav(),
        session_id="offline-self-test",
        utterance_id="offline-self-test",
    )
    orchestrator = BenchmarkOrchestrator(settings=get_settings())
    try:
        report = asyncio.run(orchestrator.run(artifact, reference_text=None))
    except Exception as exc:
        _fail(f"اجرای Orchestrator ناموفق بود: {exc}")
        return False, list(expected_ids)
    finally:
        orchestrator.close()

    results = {
        item.get("model_id"): item
        for item in report.to_dict().get("results", [])
    }
    failed: list[str] = []
    for model_id in expected_ids:
        result = results.get(model_id)
        if result and result.get("success"):
            _ok(f"{model_id}: موفق (RTF={result.get('rtf')})")
        else:
            _fail(f"{model_id}: ناموفق")
            failed.append(model_id)

    return not failed, failed


def main() -> int:
    print("=" * 60)
    print(" تست خودکار آفلاین — ASR Benchmark Harness")
    print("=" * 60)

    enabled, missing, wake_ready = check_files()
    if not enabled:
        _fail("هیچ مدل واقعی فعالی تعریف نشده است.")
        return 1
    if len(missing) == len(enabled):
        _fail("هیچ‌کدام از مدل‌های فعال نصب نشده‌اند.")
        return 1

    present = [model_id for model_id in enabled if model_id not in missing]
    pipeline_ok, failed = check_benchmark_pipeline(present)

    benchmark_ready = not missing and pipeline_ok
    assistant_ready = benchmark_ready and wake_ready

    print(f"Benchmark Ready: {'✓' if benchmark_ready else '✗'}")
    print(f"Assistant Ready: {'✓' if assistant_ready else '✗'}")
    if missing:
        _warn(f"مدل‌های نصب‌نشده: {', '.join(missing)}")
    if failed:
        _warn(f"مدل‌های ناموفق: {', '.join(failed)}")

    if assistant_ready:
        return 0
    if benchmark_ready:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

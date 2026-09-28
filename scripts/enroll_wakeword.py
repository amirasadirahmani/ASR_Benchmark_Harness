#!/usr/bin/env python3
"""ثبت نمونه‌های «آرینا» برای موتور اختیاری DTW."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SAMPLE_RATE = 16000


def _templates_dir() -> Path:
    from backend.config.settings import get_settings
    settings = get_settings()
    directory = Path(settings.wake_word.dtw_templates_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _next_index(directory: Path) -> int:
    return len(sorted(directory.glob("template_*.wav"))) + 1


def _save_pcm16(dest: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def enroll_from_wav(paths: list[str]) -> int:
    from backend.audio.audio_utils import read_wav

    directory = _templates_dir()
    saved = 0
    for value in paths:
        source = Path(value)
        if not source.exists():
            print(f"  ✗ فایل یافت نشد: {source}", file=sys.stderr)
            continue
        try:
            pcm, rate, _ = read_wav(source)
        except Exception as exc:
            print(f"  ✗ خواندن «{source.name}» ناموفق بود: {exc}", file=sys.stderr)
            continue

        destination = directory / f"template_{_next_index(directory):02d}.wav"
        _save_pcm16(destination, pcm, rate)
        print(f"  ✓ {source.name} → {destination.name}")
        saved += 1

    if saved < 3:
        print("⚠️ حداقل ۳ نمونه برای DTW توصیه می‌شود.")
    return 0 if saved else 1


def enroll_from_mic(count: int, seconds: float) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice نصب نیست؛ یا نصبش کنید یا از --from-wav استفاده کنید.",
            file=sys.stderr,
        )
        return 1

    directory = _templates_dir()
    for index in range(1, count + 1):
        input(f"[{index}/{count}] برای شروع ضبط Enter را بزنید ... ")
        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        destination = directory / f"template_{_next_index(directory):02d}.wav"
        _save_pcm16(destination, audio.tobytes(), SAMPLE_RATE)
        print(f"  ✓ {destination.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--from-wav", nargs="+", metavar="FILE")
    args = parser.parse_args()

    if args.from_wav:
        return enroll_from_wav(args.from_wav)
    if args.record:
        return enroll_from_mic(args.count, args.seconds)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

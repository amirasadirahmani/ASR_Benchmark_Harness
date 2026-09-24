"""
آداپتور ساختگی — برای تست کل زنجیره بدون نیاز به دانلود مدل چندگیگابایتی.

کاربرد:
  * تست ایزولاسیون پروسه، اندازه‌گیری RAM و جریان WebSocket پیش از
    آماده شدن مدل‌های واقعی.
  * تست CI بدون وابستگی شبکه.

فعال‌سازی: در models.yaml یک بلاک با runtime: "dummy" اضافه کن.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.asr.base import BaseASRAdapter, TranscriptionOutput
from backend.asr.registry import register_runtime
from backend.audio.audio_utils import read_wav, pcm16_duration


@register_runtime("dummy")
class DummyAdapter(BaseASRAdapter):
    """مدل شبیه‌سازی‌شده با تأخیر و مصرف حافظهٔ قابل تنظیم."""

    _PHRASES = [
        "چراغ اتاق را روشن کن",
        "دمای خانه را کم کن",
        "پخش موسیقی را شروع کن",
        "ساعت چند است",
        "فردا هوا چطور است",
    ]

    def _load_model(self) -> Any:
        # شبیه‌سازی زمان بارگذاری و اشغال RAM
        time.sleep(float(self.params.get("load_delay", 0.3)))
        mb = int(self.params.get("fake_ram_mb", 50))
        self._ballast = np.ones(mb * 1024 * 1024 // 8, dtype=np.float64)
        return {"fake": True, "ram_mb": mb}

    def _transcribe(self, audio_path: Path) -> TranscriptionOutput:
        pcm, rate, _ = read_wav(audio_path)
        duration = pcm16_duration(pcm, rate)

        rtf = float(self.params.get("fake_rtf", 0.4))
        time.sleep(max(0.01, duration * rtf))

        # خروجی قطعی بر اساس hash صوت → تکرارپذیر
        idx = int(hashlib.sha256(pcm).hexdigest()[:8], 16) % len(self._PHRASES)
        text = self._PHRASES[idx]
        if self.params.get("inject_error"):
            text = text.replace("را", "رو", 1)

        return TranscriptionOutput(
            text=text,
            language="fa",
            language_probability=0.99,
            audio_duration=round(duration, 3),
            raw={"backend": "dummy", "phrase_index": idx},
        )

    def ensure_model_exists(self) -> None:
        return  # مدل ساختگی نیازی به فایل ندارد

    def unload(self) -> None:
        self._ballast = None
        super().unload()
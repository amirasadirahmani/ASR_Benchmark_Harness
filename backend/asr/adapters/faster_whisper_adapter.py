"""
آداپتور faster-whisper (CTranslate2) — Runtime اصلی سامانه.

چرا این Runtime برای هر دو سکو انتخاب شد:
  * روی Apple M1: کوانتیزهٔ int8 روی CPU، بدون نیاز به MPS و بدون overhead
    انتقال حافظه بین CPU/GPU.
  * روی Raspberry Pi 5: تنها Runtime واقع‌بینانهٔ Whisper در ۸ گیگ RAM.
  * مصرف RAM قابل پیش‌بینی → اندازه‌گیری Peak RSS معنادار می‌شود.

⚠️ vad_filter عمداً خاموش است: VAD مستقل سامانه پیش از این مرحله قطعهٔ
   صوتی را نهایی کرده و همهٔ مدل‌ها باید *عیناً* همان بایت‌ها را ببینند
   (بند ۱۰). فعال بودن VAD داخلی مدل، ورودی مؤثر را مدل‌به‌مدل متفاوت
   می‌کرد و مقایسه را بی‌اعتبار می‌ساخت.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.asr.base import (
    ASRAdapterError, BaseASRAdapter, TranscriptionOutput,
)
from backend.asr.registry import register_runtime

logger = logging.getLogger(__name__)


@register_runtime("faster_whisper")
class FasterWhisperAdapter(BaseASRAdapter):
    """آداپتور CTranslate2 برای مدل‌های Whisper."""

    #: پارامترهای مجاز transcribe — بقیه فیلتر می‌شوند تا خطای
    #: TypeError در نسخه‌های مختلف faster-whisper رخ ندهد.
    _TRANSCRIBE_KEYS = {
        "language", "task", "beam_size", "best_of", "patience",
        "length_penalty", "repetition_penalty", "no_repeat_ngram_size",
        "temperature", "compression_ratio_threshold",
        "log_prob_threshold", "no_speech_threshold",
        "condition_on_previous_text", "prompt_reset_on_temperature",
        "initial_prompt", "prefix", "suppress_blank", "suppress_tokens",
        "without_timestamps", "max_initial_timestamp", "word_timestamps",
        "vad_filter", "vad_parameters", "hotwords",
    }

    _INIT_KEYS = {"num_workers", "download_root", "local_files_only", "flash_attention"}

    # ------------------------------------------------------------------
    def _load_model(self) -> Any:
        WhisperModel = self.require(
            "faster_whisper", "pip install faster-whisper"
        ).WhisperModel

        init_kwargs: Dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            # ⚠️ حیاتی: اجازهٔ دانلود در Runtime مطلقاً داده نمی‌شود (بند ۴)
            "local_files_only": True,
        }
        for k in self._INIT_KEYS:
            if k in self.params:
                init_kwargs[k] = self.params[k]

        logger.info(
            "بارگذاری مدل «%s» (device=%s, compute=%s, threads=%d)",
            self.model_id, self.device, self.compute_type, self.cpu_threads,
        )
        try:
            return WhisperModel(str(self.model_path), **init_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ASRAdapterError(
                f"بارگذاری مدل «{self.model_id}» ناموفق بود: {exc}\n"
                f"مسیر: {self.model_path}\n"
                "اطمینان حاصل کنید مدل با ct2-transformers-converter تبدیل شده است."
            ) from exc

    # ------------------------------------------------------------------
    def _build_transcribe_kwargs(self) -> Dict[str, Any]:
        kwargs = {k: v for k, v in self.params.items() if k in self._TRANSCRIBE_KEYS}
        kwargs.setdefault("language", "fa")
        kwargs.setdefault("task", "transcribe")
        kwargs.setdefault("beam_size", 5)
        kwargs.setdefault("without_timestamps", True)
        kwargs.setdefault("condition_on_previous_text", False)
        kwargs["vad_filter"] = False        # ← غیرقابل تغییر (بند ۱۰)
        return kwargs

    def _transcribe(self, audio_path: Path) -> TranscriptionOutput:
        if self._model is None:
            raise ASRAdapterError("مدل بارگذاری نشده است.")

        t0 = time.perf_counter()
        segments_iter, info = self._model.transcribe(
            str(audio_path), **self._build_transcribe_kwargs()
        )

        # faster-whisper تنبل (lazy) است؛ پیمایش کامل لازم است تا
        # زمان اندازه‌گیری‌شده واقعاً شامل کل استنتاج باشد.
        segments: List[Dict[str, Any]] = []
        parts: List[str] = []
        for seg in segments_iter:
            parts.append(seg.text)
            segments.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip(),
                "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0)), 4),
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0)), 4),
            })
        elapsed = time.perf_counter() - t0

        return TranscriptionOutput(
            text="".join(parts).strip(),
            language=getattr(info, "language", None),
            language_probability=_safe_float(getattr(info, "language_probability", None)),
            audio_duration=_safe_float(getattr(info, "duration", 0.0)) or 0.0,
            inference_time=elapsed,
            segments=segments,
            raw={
                "segment_count": len(segments),
                "duration_after_vad": _safe_float(
                    getattr(info, "duration_after_vad", None)),
            },
        )

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import gc
            gc.collect()
        except Exception:  # noqa: BLE001
            pass


def _safe_float(v: Any) -> float | None:
    try:
        return round(float(v), 4) if v is not None else None
    except (TypeError, ValueError):
        return None
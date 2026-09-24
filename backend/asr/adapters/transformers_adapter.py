"""
آداپتور مرجع مبتنی بر HuggingFace Transformers.

نقش در پروژه:
    الف) صحت‌سنجی تبدیل CTranslate2 — اگر خروجی CT2 با خروجی مرجع به‌طور
         معنادار تفاوت داشت، ایراد از تبدیل است نه از مدل.
    ب) پشتیبانی از معماری‌هایی که هنوز به CT2 تبدیل نشده‌اند
       (مثلاً Wav2Vec2 فارسی).

⚠️ روی Raspberry Pi 5 توصیه نمی‌شود: مصرف RAM چند برابر CT2 است.
   در models.yaml به‌صورت پیش‌فرض enabled: false گذاشته شده است.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from backend.asr.base import ASRAdapterError, BaseASRAdapter, TranscriptionOutput
from backend.asr.registry import register_runtime
from backend.audio.audio_utils import read_wav, bytes_to_float32

logger = logging.getLogger(__name__)


class _TorchMixin:
    """ابزار مشترک آداپتورهای مبتنی بر torch."""

    def _resolve_torch_device(self, torch: Any) -> str:
        """auto → mps روی M1، cuda در صورت وجود، وگرنه cpu."""
        if self.device not in ("auto", None):  # type: ignore[attr-defined]
            return self.device                  # type: ignore[attr-defined]
        try:
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _torch_dtype(self, torch: Any, device: str) -> Any:
        ct = self.compute_type  # type: ignore[attr-defined]
        if device == "cpu":
            return torch.float32          # float16 روی CPU کند و ناپایدار است
        if ct in ("float16", "int8_float16"):
            return torch.float16
        if ct == "bfloat16":
            return torch.bfloat16
        return torch.float32


@register_runtime("transformers_whisper")
class TransformersWhisperAdapter(_TorchMixin, BaseASRAdapter):
    """Whisper از طریق transformers (مرجع صحت‌سنجی)."""

    def _load_model(self) -> Any:
        torch = self.require("torch", "pip install torch")
        tf = self.require("transformers", "pip install transformers")

        device = self._resolve_torch_device(torch)
        dtype = self._torch_dtype(torch, device)
        self.device = device

        try:
            torch.set_num_threads(self.cpu_threads)
        except Exception:  # noqa: BLE001
            pass

        try:
            processor = tf.WhisperProcessor.from_pretrained(
                str(self.model_path), local_files_only=True)
            model = tf.WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path), local_files_only=True, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            raise ASRAdapterError(
                f"بارگذاری مدل «{self.model_id}» ناموفق بود: {exc}"
            ) from exc

        model.to(device)
        model.eval()
        self._processor = processor
        self._torch = torch
        self._dtype = dtype
        logger.info("مدل «%s» روی %s (%s) بارگذاری شد",
                    self.model_id, device, dtype)
        return model

    # ------------------------------------------------------------------
    def _transcribe(self, audio_path: Path) -> TranscriptionOutput:
        if self._model is None:
            raise ASRAdapterError("مدل بارگذاری نشده است.")
        torch = self._torch

        pcm, sample_rate, _ = read_wav(audio_path)
        samples = bytes_to_float32(pcm).astype(np.float32)
        duration = len(samples) / float(sample_rate)

        t0 = time.perf_counter()
        inputs = self._processor(
            samples, sampling_rate=sample_rate, return_tensors="pt")
        features = inputs.input_features.to(self.device, dtype=self._dtype)

        gen_kwargs: Dict[str, Any] = {
            "num_beams": int(self.params.get("num_beams", 5)),
            "max_new_tokens": int(self.params.get("max_new_tokens", 200)),
            "do_sample": False,
        }
        try:
            gen_kwargs["forced_decoder_ids"] = None
            gen_kwargs["language"] = self.params.get("language", "fa")
            gen_kwargs["task"] = self.params.get("task", "transcribe")
        except Exception:  # noqa: BLE001
            pass

        with torch.no_grad():
            ids = self._model.generate(features, **gen_kwargs)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        elapsed = time.perf_counter() - t0

        return TranscriptionOutput(
            text=text.strip(),
            language=self.params.get("language", "fa"),
            audio_duration=round(duration, 3),
            inference_time=elapsed,
            raw={"backend": "transformers", "device": self.device},
        )

    def unload(self) -> None:
        self._processor = None
        super().unload()
        try:
            torch = getattr(self, "_torch", None)
            if torch is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


@register_runtime("transformers_ctc")
class TransformersCTCAdapter(_TorchMixin, BaseASRAdapter):
    """
    آداپتور CTC (Wav2Vec2 / HuBERT فارسی).

    نمونهٔ عملی بند ۲۵: افزودن یک معماری کاملاً متفاوت، بدون هیچ تغییری
    در Orchestrator، Metrics، CSV یا UI.
    """

    def _load_model(self) -> Any:
        torch = self.require("torch", "pip install torch")
        tf = self.require("transformers", "pip install transformers")

        device = self._resolve_torch_device(torch)
        self.device = device
        try:
            torch.set_num_threads(self.cpu_threads)
        except Exception:  # noqa: BLE001
            pass

        processor = tf.AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True)
        model = tf.AutoModelForCTC.from_pretrained(
            str(self.model_path), local_files_only=True)
        model.to(device)
        model.eval()

        self._processor = processor
        self._torch = torch
        return model

    def _transcribe(self, audio_path: Path) -> TranscriptionOutput:
        torch = self._torch
        pcm, sample_rate, _ = read_wav(audio_path)
        samples = bytes_to_float32(pcm).astype(np.float32)
        duration = len(samples) / float(sample_rate)

        t0 = time.perf_counter()
        inputs = self._processor(
            samples, sampling_rate=sample_rate, return_tensors="pt")
        values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self._model(values).logits
        ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(ids)[0]
        elapsed = time.perf_counter() - t0

        return TranscriptionOutput(
            text=text.strip(),
            language="fa",
            audio_duration=round(duration, 3),
            inference_time=elapsed,
            raw={"backend": "transformers_ctc", "device": self.device},
        )

    def unload(self) -> None:
        self._processor = None
        super().unload()
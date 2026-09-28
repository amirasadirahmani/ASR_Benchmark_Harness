"""
ابزارهای پایهٔ صوت.

قرارداد داخلی سامانه:
PCM signed 16-bit little-endian, mono, 16000 Hz.
"""

from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

SAMPLE_WIDTH = 2
INT16_MAX = 32767.0


def bytes_to_int16(data: bytes) -> np.ndarray:
    if len(data) % SAMPLE_WIDTH:
        data = data[: len(data) - (len(data) % SAMPLE_WIDTH)]
    return np.frombuffer(data, dtype="<i2")


def bytes_to_float32(data: bytes) -> np.ndarray:
    return bytes_to_int16(data).astype(np.float32) / INT16_MAX


def float32_to_bytes(samples: np.ndarray) -> bytes:
    arr = np.asarray(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * INT16_MAX).astype("<i2").tobytes()


def pcm16_duration(data: bytes, sample_rate: int = 16000, channels: int = 1) -> float:
    frames = len(data) // (SAMPLE_WIDTH * channels)
    return frames / float(sample_rate) if sample_rate else 0.0


def rms_dbfs(data: bytes) -> float:
    if not data:
        return -96.0
    samples = bytes_to_int16(data).astype(np.float64)
    if samples.size == 0:
        return -96.0
    rms = math.sqrt(float(np.mean(samples * samples)))
    if rms <= 1e-9:
        return -96.0
    return 20.0 * math.log10(rms / INT16_MAX)


def peak_amplitude(data: bytes) -> float:
    if not data:
        return 0.0
    samples = bytes_to_int16(data)
    return float(np.max(np.abs(samples.astype(np.int32)))) / INT16_MAX if samples.size else 0.0


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_silence(seconds: float, sample_rate: int = 16000) -> bytes:
    return b"\x00" * int(seconds * sample_rate) * SAMPLE_WIDTH


def make_warmup_tone(seconds: float = 1.0, sample_rate: int = 16000,
                     freq: float = 220.0) -> bytes:
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave_ = 0.05 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    wave_ += (np.random.rand(n).astype(np.float32) - 0.5) * 0.002
    return float32_to_bytes(wave_)


def trim_pcm(data: bytes, max_seconds: float, sample_rate: int = 16000) -> bytes:
    max_bytes = int(max_seconds * sample_rate) * SAMPLE_WIDTH
    return data[:max_bytes] if len(data) > max_bytes else data


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    ratio = dst_rate / float(src_rate)
    n_out = int(round(samples.size * ratio))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1 or samples.size == 0:
        return samples
    usable = (samples.size // channels) * channels
    return samples[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


def write_wav(path: str | Path, pcm: bytes, sample_rate: int = 16000,
              channels: int = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return path


def read_wav(path: str | Path) -> Tuple[bytes, int, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width != SAMPLE_WIDTH:
        raise ValueError(f"فقط WAV با عمق ۱۶ بیت پشتیبانی می‌شود (دریافتی: {width * 8} بیت)")

    samples = bytes_to_float32(frames)
    if channels > 1:
        samples = to_mono(samples, channels)
        channels = 1
    if rate != 16000:
        samples = resample_linear(samples, rate, 16000)
        rate = 16000

    return float32_to_bytes(samples), rate, channels


@dataclass
class AudioArtifact:
    pcm: bytes
    sample_rate: int = 16000
    channels: int = 1
    session_id: str = ""
    utterance_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    path: Optional[Path] = None
    pre_roll_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _sha: Optional[str] = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return pcm16_duration(self.pcm, self.sample_rate, self.channels)

    @property
    def sha256(self) -> str:
        if self._sha is None:
            self._sha = sha256_of_bytes(self.pcm)
        return self._sha

    @property
    def size_bytes(self) -> int:
        return len(self.pcm)

    def as_float32(self) -> np.ndarray:
        return bytes_to_float32(self.pcm)

    def save(self, directory: str | Path, filename: Optional[str] = None) -> Path:
        directory = Path(directory)
        name = filename or f"{self.utterance_id or self.sha256[:12]}.wav"
        self.path = write_wav(directory / name, self.pcm, self.sample_rate, self.channels)
        return self.path

    def to_dict(self, include_path: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "session_id": self.session_id,
            "utterance_id": self.utterance_id,
            "created_at": self.created_at,
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "pre_roll_seconds": round(self.pre_roll_seconds, 3),
            "rms_dbfs": round(rms_dbfs(self.pcm), 2),
            "peak": round(peak_amplitude(self.pcm), 4),
        }
        if include_path and self.path:
            d["path"] = str(self.path)
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_wav(cls, path: str | Path, **kwargs: Any) -> "AudioArtifact":
        pcm, rate, ch = read_wav(path)
        art = cls(pcm=pcm, sample_rate=rate, channels=ch, **kwargs)

        try:
            with wave.open(str(path), "rb") as wf:
                already_canonical = (
                    wf.getframerate() == 16000
                    and wf.getnchannels() == 1
                    and wf.getsampwidth() == SAMPLE_WIDTH
                )
        except Exception:
            already_canonical = False

        # فقط فایل canonical را مستقیم به Worker بده؛ در غیر این صورت
        # Orchestrator از pcm استانداردشده یک WAV تازه می‌سازد.
        art.path = Path(path) if already_canonical else None
        return art

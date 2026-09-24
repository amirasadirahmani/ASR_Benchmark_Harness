"""
بافر حلقوی Pre-Roll و انباشتگر فریم.

مسئلهٔ حل‌شده (بند ۸):
    کاربر معمولاً بلافاصله پس از «آرینا» شروع به گفتن فرمان می‌کند و گاهی
    حتی قبل از آنکه تشخیص‌دهنده trigger شود، چند صد میلی‌ثانیه از ابتدای
    فرمان گفته شده است. بدون Pre-Roll، ابتدای جمله بریده می‌شود و WER
    همهٔ مدل‌ها به‌طور یکسان خراب می‌شود.

راه‌حل:
    صوت همواره در یک بافر حلقوی با ظرفیت ثابت (پیش‌فرض ۱.۵ ثانیه) نگهداری
    می‌شود. به محض trigger، محتوای بافر به ابتدای ضبط الصاق می‌گردد.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Iterator, List, Optional

from backend.audio.audio_utils import SAMPLE_WIDTH, pcm16_duration


class PreRollBuffer:
    """
    بافر حلقوی بایت‌محور با ظرفیت بر حسب ثانیه (thread-safe).

    Example:
        >>> b = PreRollBuffer(capacity_seconds=1.5)
        >>> b.write(frame)
        >>> pre = b.snapshot()      # ابتدای فرمان از دست نمی‌رود
    """

    def __init__(
        self,
        capacity_seconds: float = 1.5,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.capacity_seconds = capacity_seconds
        self.capacity_bytes = int(
            capacity_seconds * sample_rate * SAMPLE_WIDTH * channels
        )
        self._chunks: Deque[bytes] = deque()
        self._size = 0
        self._lock = threading.Lock()
        self._total_written = 0

    # ------------------------------------------------------------------
    def write(self, data: bytes) -> None:
        """افزودن داده و حذف قدیمی‌ترین بایت‌ها در صورت سرریز."""
        if not data:
            return
        with self._lock:
            self._total_written += len(data)

            # اگر خودِ chunk از کل ظرفیت بزرگ‌تر بود، فقط انتهایش را نگه دار
            if len(data) >= self.capacity_bytes:
                self._chunks.clear()
                self._chunks.append(data[-self.capacity_bytes:])
                self._size = len(self._chunks[0])
                return

            self._chunks.append(data)
            self._size += len(data)

            while self._size > self.capacity_bytes and self._chunks:
                oldest = self._chunks[0]
                excess = self._size - self.capacity_bytes
                if len(oldest) <= excess:
                    self._chunks.popleft()
                    self._size -= len(oldest)
                else:
                    self._chunks[0] = oldest[excess:]
                    self._size -= excess

    def snapshot(self) -> bytes:
        """کپی محتوای فعلی بافر بدون تخلیهٔ آن."""
        with self._lock:
            return b"".join(self._chunks)

    def drain(self) -> bytes:
        """خواندن و تخلیهٔ بافر (اتمیک)."""
        with self._lock:
            data = b"".join(self._chunks)
            self._chunks.clear()
            self._size = 0
            return data

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._size = 0

    def tail(self, seconds: float) -> bytes:
        """فقط N ثانیهٔ انتهایی بافر."""
        want = int(seconds * self.sample_rate * SAMPLE_WIDTH * self.channels)
        data = self.snapshot()
        return data[-want:] if len(data) > want else data

    # ------------------------------------------------------------------
    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size

    @property
    def duration(self) -> float:
        return pcm16_duration(b"\x00" * self.size_bytes, self.sample_rate, self.channels)

    @property
    def is_full(self) -> bool:
        return self.size_bytes >= self.capacity_bytes

    @property
    def total_written_seconds(self) -> float:
        with self._lock:
            total = self._total_written
        return total / (self.sample_rate * SAMPLE_WIDTH * self.channels)

    def __len__(self) -> int:
        return self.size_bytes

    def __repr__(self) -> str:
        return (
            f"<PreRollBuffer {self.duration:.2f}s/"
            f"{self.capacity_seconds:.2f}s ({self.size_bytes}B)>"
        )


class FrameAccumulator:
    """
    تبدیل جریان نامنظم بایت (از WebSocket) به فریم‌های *دقیقاً هم‌اندازه*.

    ضروری است چون WebRTC VAD فقط فریم ۱۰/۲۰/۳۰ میلی‌ثانیه‌ای می‌پذیرد،
    در حالی که مرورگر ممکن است chunkهایی با اندازهٔ دلخواه بفرستد.
    """

    def __init__(self, frame_bytes: int) -> None:
        if frame_bytes <= 0:
            raise ValueError("frame_bytes باید مثبت باشد.")
        self.frame_bytes = frame_bytes
        self._buf = bytearray()

    def push(self, data: bytes) -> List[bytes]:
        """افزودن داده و برگرداندن فریم‌های کامل آماده‌شده."""
        if data:
            self._buf.extend(data)
        frames: List[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            frames.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return frames

    def iter_push(self, data: bytes) -> Iterator[bytes]:
        yield from self.push(data)

    def flush(self, pad: bool = True) -> Optional[bytes]:
        """باقیماندهٔ بافر؛ در صورت نیاز با سکوت به اندازهٔ فریم می‌رسد."""
        if not self._buf:
            return None
        rest = bytes(self._buf)
        self._buf.clear()
        if pad and len(rest) < self.frame_bytes:
            rest += b"\x00" * (self.frame_bytes - len(rest))
        return rest

    def reset(self) -> None:
        self._buf.clear()

    @property
    def pending_bytes(self) -> int:
        return len(self._buf)
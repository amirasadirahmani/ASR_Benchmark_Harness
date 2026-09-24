"""
سنجه‌های ارزیابی: WER، CER، RTF و آمار تکرار.

نکات متدولوژیک:
  * WER/CER با الگوریتم Levenshtein روی دنبالهٔ توکن/کاراکترِ *نرمال‌شده*
    محاسبه می‌شوند (پیاده‌سازی مستقل، بدون وابستگی خارجی مثل jiwer
    تا آفلاین بودن ۱۰۰٪ حفظ شود).
  * پیاده‌سازی Levenshtein از دو سطر استفاده می‌کند → حافظه O(min(n,m)).
  * برای backtrace (شمارش S/D/I) ماتریس کامل با نوع uint8 نگهداری می‌شود
    که برای طول جملات فرمان صوتی کاملاً کم‌هزینه است.
  * اگر متن مرجع وجود نداشته باشد، مقادیر None برمی‌گردند و لایهٔ نمایش
    پیام «جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت.» را نشان می‌دهد
    (بند ۱۵).
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.core.persian_normalizer import (
    NORMALIZER_VERSION,
    PersianNormalizer,
    get_normalizer,
)


class EditOperation(str, Enum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"


# ---------------------------------------------------------------------------
# الگوریتم پایه
# ---------------------------------------------------------------------------
def _levenshtein_counts(
    ref: Sequence[str], hyp: Sequence[str]
) -> Tuple[int, int, int, int]:
    """
    فاصلهٔ ویرایشی به همراه تفکیک عملیات.

    Returns:
        (substitutions, deletions, insertions, total_distance)
    """
    n, m = len(ref), len(hyp)

    if n == 0:
        return 0, 0, m, m
    if m == 0:
        return 0, n, 0, n

    # ماتریس backtrace: 0=match, 1=sub, 2=del, 3=ins
    _MATCH, _SUB, _DEL, _INS = 0, 1, 2, 3
    bt = [bytearray(m + 1) for _ in range(n + 1)]

    prev = list(range(m + 1))
    for j in range(1, m + 1):
        bt[0][j] = _INS

    for i in range(1, n + 1):
        curr = [i] + [0] * m
        bt[i][0] = _DEL
        ref_i = ref[i - 1]
        for j in range(1, m + 1):
            if ref_i == hyp[j - 1]:
                curr[j] = prev[j - 1]
                bt[i][j] = _MATCH
                continue
            sub = prev[j - 1] + 1
            dele = prev[j] + 1
            ins = curr[j - 1] + 1
            best = min(sub, dele, ins)
            curr[j] = best
            bt[i][j] = _SUB if best == sub else (_DEL if best == dele else _INS)
        prev = curr

    distance = prev[m]

    # backtrace
    s = d = ins_count = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == _MATCH:
            i -= 1
            j -= 1
        elif op == _SUB:
            s += 1
            i -= 1
            j -= 1
        elif op == _DEL:
            d += 1
            i -= 1
        else:
            ins_count += 1
            j -= 1

    return s, d, ins_count, distance


def _error_rate(distance: int, ref_len: int) -> Optional[float]:
    """نرخ خطا؛ اگر مرجع خالی باشد None."""
    if ref_len == 0:
        return None
    return distance / ref_len


# ---------------------------------------------------------------------------
# APIهای ساده (برای تست و استفادهٔ مستقیم)
# ---------------------------------------------------------------------------
def compute_wer(
    reference: str, hypothesis: str, normalizer: Optional[PersianNormalizer] = None
) -> Optional[float]:
    """نرخ خطای کلمه‌ای (0.0 تا ∞؛ None اگر مرجع خالی باشد)."""
    nz = normalizer or get_normalizer()
    ref = nz.tokenize(reference)
    hyp = nz.tokenize(hypothesis)
    _, _, _, dist = _levenshtein_counts(ref, hyp)
    return _error_rate(dist, len(ref))


def compute_cer(
    reference: str, hypothesis: str, normalizer: Optional[PersianNormalizer] = None
) -> Optional[float]:
    """نرخ خطای کاراکتری (فاصله‌ها نادیده گرفته می‌شوند)."""
    nz = normalizer or get_normalizer()
    ref = nz.characters(reference)
    hyp = nz.characters(hypothesis)
    _, _, _, dist = _levenshtein_counts(ref, hyp)
    return _error_rate(dist, len(ref))


def compute_rtf(processing_seconds: float, audio_seconds: float) -> Optional[float]:
    """
    Real-Time Factor = زمان پردازش ÷ طول صوت.
    RTF < 1 یعنی سریع‌تر از بلادرنگ.
    """
    if audio_seconds is None or audio_seconds <= 0:
        return None
    return processing_seconds / audio_seconds


# ---------------------------------------------------------------------------
# ساختار نتیجه
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionMetrics:
    """سنجه‌های کامل یک اجرا (یک مدل × یک نوبت)."""

    # --- متن ---
    reference_raw: Optional[str] = None
    reference_normalized: Optional[str] = None
    hypothesis_raw: str = ""
    hypothesis_normalized: str = ""

    # --- دقت ---
    wer: Optional[float] = None
    cer: Optional[float] = None
    accuracy: Optional[float] = None          # 1 - WER (کف صفر)
    exact_match: Optional[bool] = None

    substitutions: Optional[int] = None
    deletions: Optional[int] = None
    insertions: Optional[int] = None
    hits: Optional[int] = None
    reference_tokens: int = 0
    hypothesis_tokens: int = 0
    reference_chars: int = 0
    hypothesis_chars: int = 0

    # --- کارایی ---
    audio_duration: float = 0.0
    load_time: float = 0.0                    # زمان بارگذاری مدل
    warmup_time: float = 0.0
    inference_time: float = 0.0               # زمان خالص رونویسی
    total_time: float = 0.0
    rtf: Optional[float] = None
    latency_ms: Optional[float] = None

    # --- منابع ---
    peak_ram_mb: Optional[float] = None
    baseline_ram_mb: Optional[float] = None
    model_ram_mb: Optional[float] = None       # peak - baseline
    cpu_percent_avg: Optional[float] = None

    # --- وضعیت ---
    success: bool = True
    error: Optional[str] = None
    has_reference: bool = False
    no_reference_message: Optional[str] = None

    # --- ابرداده ---
    normalizer_version: str = NORMALIZER_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """تجمیع چند نوبت اجرا برای یک مدل (بند ۲۲)."""
    runs: int = 0
    successful_runs: int = 0

    wer_mean: Optional[float] = None
    wer_std: Optional[float] = None
    wer_min: Optional[float] = None
    wer_max: Optional[float] = None

    cer_mean: Optional[float] = None
    cer_std: Optional[float] = None

    rtf_mean: Optional[float] = None
    rtf_std: Optional[float] = None
    rtf_min: Optional[float] = None
    rtf_max: Optional[float] = None

    inference_time_mean: Optional[float] = None
    inference_time_median: Optional[float] = None
    load_time_mean: Optional[float] = None

    peak_ram_mb_max: Optional[float] = None
    peak_ram_mb_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
class MetricsEngine:
    """
    موتور محاسبهٔ سنجه‌ها.

    تنها نقطهٔ مجاز برای محاسبهٔ WER/CER/RTF در کل سامانه — تا تضمین شود
    همهٔ مدل‌ها با «یک خط‌کش» سنجیده می‌شوند (بند ۱۴).
    """

    def __init__(
        self,
        normalizer: Optional[PersianNormalizer] = None,
        no_reference_message: Optional[str] = None,
    ) -> None:
        self.normalizer = normalizer or get_normalizer()
        self._no_ref_msg = (
            no_reference_message
            or "جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت."
        )

    # ------------------------------------------------------------------
    def evaluate(
        self,
        hypothesis: str,
        reference: Optional[str] = None,
        *,
        audio_duration: float = 0.0,
        load_time: float = 0.0,
        warmup_time: float = 0.0,
        inference_time: float = 0.0,
        total_time: Optional[float] = None,
        peak_ram_mb: Optional[float] = None,
        baseline_ram_mb: Optional[float] = None,
        cpu_percent_avg: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionMetrics:
        """محاسبهٔ کامل سنجه‌های یک اجرا."""

        hyp_norm = self.normalizer.normalize(hypothesis)
        has_ref = bool(reference and self.normalizer.normalize(reference))

        m = TranscriptionMetrics(
            hypothesis_raw=hypothesis or "",
            hypothesis_normalized=hyp_norm,
            audio_duration=round(audio_duration, 4),
            load_time=round(load_time, 4),
            warmup_time=round(warmup_time, 4),
            inference_time=round(inference_time, 4),
            total_time=round(
                total_time if total_time is not None
                else (load_time + warmup_time + inference_time),
                4,
            ),
            peak_ram_mb=_round(peak_ram_mb, 2),
            baseline_ram_mb=_round(baseline_ram_mb, 2),
            cpu_percent_avg=_round(cpu_percent_avg, 2),
            success=success,
            error=error,
            has_reference=has_ref,
            normalizer_version=self.normalizer.version,
            metadata=metadata or {},
        )

        # --- منابع ---
        if peak_ram_mb is not None and baseline_ram_mb is not None:
            m.model_ram_mb = round(max(0.0, peak_ram_mb - baseline_ram_mb), 2)

        # --- کارایی ---
        m.rtf = _round(compute_rtf(inference_time, audio_duration), 4)
        m.latency_ms = round(inference_time * 1000, 2)

        hyp_tokens = hyp_norm.split() if hyp_norm else []
        m.hypothesis_tokens = len(hyp_tokens)
        m.hypothesis_chars = len(hyp_norm.replace(" ", ""))

        # --- دقت ---
        if not has_ref:
            # بند ۱۵: هیچ عدد ساختگی تولید نمی‌شود
            m.no_reference_message = self._no_ref_msg
            return m

        ref_norm = self.normalizer.normalize(reference)
        m.reference_raw = reference
        m.reference_normalized = ref_norm

        ref_tokens = ref_norm.split()
        m.reference_tokens = len(ref_tokens)
        m.reference_chars = len(ref_norm.replace(" ", ""))

        s, d, i, dist = _levenshtein_counts(ref_tokens, hyp_tokens)
        m.substitutions, m.deletions, m.insertions = s, d, i
        m.hits = max(0, len(ref_tokens) - s - d)
        m.wer = _round(_error_rate(dist, len(ref_tokens)), 4)

        ref_chars = self.normalizer.characters(reference)
        hyp_chars = self.normalizer.characters(hypothesis)
        _, _, _, cdist = _levenshtein_counts(ref_chars, hyp_chars)
        m.cer = _round(_error_rate(cdist, len(ref_chars)), 4)

        if m.wer is not None:
            m.accuracy = round(max(0.0, 1.0 - m.wer), 4)
        m.exact_match = (ref_norm == hyp_norm)

        return m

    # ------------------------------------------------------------------
    def aggregate(self, runs: List[TranscriptionMetrics]) -> AggregatedMetrics:
        """تجمیع چند نوبت اجرا (میانگین/انحراف معیار) — بند ۲۲."""
        agg = AggregatedMetrics(runs=len(runs))
        ok = [r for r in runs if r.success]
        agg.successful_runs = len(ok)
        if not ok:
            return agg

        def stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not values:
                return None, None
            mean = round(statistics.fmean(values), 4)
            std = round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
            return mean, std

        wers = [r.wer for r in ok if r.wer is not None]
        cers = [r.cer for r in ok if r.cer is not None]
        rtfs = [r.rtf for r in ok if r.rtf is not None]
        infs = [r.inference_time for r in ok]
        loads = [r.load_time for r in ok]
        rams = [r.peak_ram_mb for r in ok if r.peak_ram_mb is not None]

        agg.wer_mean, agg.wer_std = stats(wers)
        agg.cer_mean, agg.cer_std = stats(cers)
        agg.rtf_mean, agg.rtf_std = stats(rtfs)

        if wers:
            agg.wer_min, agg.wer_max = round(min(wers), 4), round(max(wers), 4)
        if rtfs:
            agg.rtf_min, agg.rtf_max = round(min(rtfs), 4), round(max(rtfs), 4)
        if infs:
            agg.inference_time_mean = round(statistics.fmean(infs), 4)
            agg.inference_time_median = round(statistics.median(infs), 4)
        if loads:
            agg.load_time_mean = round(statistics.fmean(loads), 4)
        if rams:
            agg.peak_ram_mb_max = round(max(rams), 2)
            agg.peak_ram_mb_mean = round(statistics.fmean(rams), 2)

        return agg

    # ------------------------------------------------------------------
    @staticmethod
    def rank(
        results: Dict[str, AggregatedMetrics],
        primary: str = "wer_mean",
        secondary: str = "rtf_mean",
    ) -> List[str]:
        """
        رتبه‌بندی مدل‌ها: ابتدا بر اساس WER (کمتر بهتر)، سپس RTF.
        مدل‌های بدون مقدار به انتهای فهرست می‌روند.
        """
        def key(item: Tuple[str, AggregatedMetrics]):
            _, a = item
            p = getattr(a, primary, None)
            s = getattr(a, secondary, None)
            return (
                p is None, p if p is not None else 0.0,
                s is None, s if s is not None else 0.0,
            )

        return [mid for mid, _ in sorted(results.items(), key=key)]


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)
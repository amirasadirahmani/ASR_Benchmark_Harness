"""
تست دود جامع — اعتبارسنجی فاز ۱ و ۲ بدون نیاز به مدل یا میکروفون.

اجرا:
    python -m tests.smoke_test
"""

from __future__ import annotations

import asyncio
import sys

import numpy as np

from backend.audio.audio_utils import (
    AudioArtifact, float32_to_bytes, make_silence, pcm16_duration,
    rms_dbfs, sha256_of_bytes,
)
from backend.audio.ring_buffer import FrameAccumulator, PreRollBuffer
from backend.audio.session import AudioSession
from backend.audio.vad import VADGate, create_vad
from backend.config.model_config import describe_environment, load_model_configs
from backend.config.settings import get_settings
from backend.core import MetricsEngine, get_normalizer

PASS, FAIL = "\033[92m✅\033[0m", "\033[91m❌\033[0m"
_results: list[tuple[bool, str]] = []


def check(ok: bool, title: str, detail: str = "") -> bool:
    _results.append((ok, title))
    print(f"  {PASS if ok else FAIL} {title}" + (f"  →  {detail}" if detail else ""))
    return ok


def section(t: str) -> None:
    print(f"\n\033[1m{'─' * 62}\n{t}\n{'─' * 62}\033[0m")


# ---------------------------------------------------------------------------
def speech_like(seconds: float = 1.5, sr: int = 16000) -> bytes:
    """
    سیگنال شبه‌گفتار: فرکانس پایه ۱۲۰ هرتز + هارمونیک‌ها + فورمنت‌ها
    + مدولاسیون دامنهٔ هجایی (~۴ هرتز) + کمی نویز.
    WebRTC VAD این را به‌عنوان گفتار می‌پذیرد.
    """
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = np.zeros(n, dtype=np.float32)

    f0 = 120.0
    for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15], start=1):
        sig += amp * np.sin(2 * np.pi * f0 * k * t)
    for f, amp in [(700, 0.5), (1220, 0.35), (2600, 0.2)]:  # فورمنت‌های /a/
        sig += amp * np.sin(2 * np.pi * f * t)

    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)     # ریتم هجایی
    sig *= envelope
    sig += np.random.randn(n).astype(np.float32) * 0.01
    sig /= (np.max(np.abs(sig)) + 1e-9)
    return float32_to_bytes(sig * 0.35)


# ═══════════════════════════════════════════════════════════════════════════
def test_config() -> None:
    section("۱) پیکربندی و محیط")
    s = get_settings()
    check(s.offline_mode is True, "حالت آفلاین فعال است")
    check(s.audio.sample_rate == 16000, "نرخ نمونه‌برداری", f"{s.audio.sample_rate} Hz")
    check(s.audio.frame_bytes == 640, "اندازهٔ فریم", f"{s.audio.frame_bytes} بایت / {s.audio.frame_ms}ms")
    check(s.vad.silence_duration == 2.0, "آستانهٔ سکوت", f"{s.vad.silence_duration}s")
    check(s.wake_word.word == "آرینا", "کلمهٔ بیدارباش", s.wake_word.word)
    check(s.server.host == "127.0.0.1", "اتصال فقط روی localhost")

    try:
        cfgs = load_model_configs()
        enabled = [c for c in cfgs if c.enabled]
        avail = [c for c in cfgs if c.exists()]
        check(len(cfgs) > 0, "رجیستری مدل‌ها", f"{len(cfgs)} مدل، {len(enabled)} فعال، {len(avail)} موجود روی دیسک")
    except Exception as e:
        check(False, "رجیستری مدل‌ها", str(e))

    env = describe_environment()
    print(f"     محیط: {env['machine']} | RAM {env.get('total_ram_mb', '?')} MB | {env['cpu_count']} هسته")


def test_normalizer() -> None:
    section("۲) نرمال‌ساز فارسی")
    n = get_normalizer()
    cases = [
        ("سلامِ  گرم! حالِ شما چطور اسـت؟", "سلام گرم حال شما چطور است"),
        ("كتاب‌هاي عربي", "کتاب های عربی"),
        ("۱۲۳ و ٤٥٦", "123 و 456"),
        ("«نقل‌قول»، و؛ نقطه.", "نقل قول و نقطه"),
        ("أحمد إسماعیل", "احمد اسماعیل"),
        ("", ""),
    ]
    for src, expected in cases:
        got = n.normalize(src)
        check(got == expected, f"«{src[:26] or '(خالی)'}»", f"«{got}»")

    check(n.version == "1.0.0", "نسخهٔ نرمال‌ساز", n.version)
    check(n.contains("آرینا چراغ را روشن کن", "آرینا"), "تشخیص کلمهٔ کامل")
    check(not n.contains("آرینام", "آرینا"), "عدم تطبیق جزئی کلمه")


def test_metrics() -> None:
    section("۳) سنجه‌های ارزیابی")
    e = MetricsEngine()

    m = e.evaluate("سلام حال شما چطوره", "سلام حال شما چطور است",
                   audio_duration=2.0, inference_time=0.5)
    check(m.wer == 0.4, "محاسبهٔ WER", f"{m.wer} (۲ خطا از ۵ کلمه)")
    check(m.rtf == 0.25, "محاسبهٔ RTF", f"{m.rtf} (0.5s ÷ 2.0s)")
    check(m.accuracy == 0.6, "دقت", str(m.accuracy))
    check(m.substitutions == 1 and m.deletions == 1, "تفکیک خطاها",
          f"S={m.substitutions} D={m.deletions} I={m.insertions}")

    m2 = e.evaluate("سلام دنیا", "سلام دنیا")
    check(m2.wer == 0.0 and m2.exact_match, "تطابق کامل → WER صفر")

    m3 = e.evaluate("چیزی گفتم", None)
    check(m3.wer is None and m3.has_reference is False, "بدون متن مرجع → WER تهی")
    check("جمله مبدا" in (m3.no_reference_message or ""), "پیام فارسی نبود مرجع")

    agg = e.aggregate([m, m2])
    check(agg.runs == 2 and agg.wer_mean == 0.2, "تجمیع چند اجرا", f"میانگین WER={agg.wer_mean}")


def test_buffers() -> None:
    section("۴) بافر و فریم‌بندی")
    b = PreRollBuffer(1.5)
    b.write(make_silence(3.0))
    d = pcm16_duration(b.snapshot())
    check(abs(d - 1.5) < 0.01, "سقف بافر Pre-Roll", f"{d:.2f}s از ۳s ورودی")
    check(b.is_full, "پر بودن بافر")
    b.clear()
    check(len(b) == 0, "پاک‌سازی بافر")

    acc = FrameAccumulator(640)
    frames = acc.push(b"\x00" * 1600)
    check(len(frames) == 2 and acc.pending_bytes == 320, "فریم‌بندی دقیق",
          f"{len(frames)} فریم کامل + {acc.pending_bytes} بایت باقیمانده")
    check(len(acc.flush()) == 640, "padding باقیمانده")


def test_vad() -> None:
    section("۵) موتور VAD")
    s = get_settings()
    vad = create_vad(s.vad, 16000, 20)
    is_webrtc = vad.name == "webrtc"
    check(is_webrtc, f"موتور فعال: {vad.name}",
          "" if is_webrtc else "⚠️ WebRTC نصب نشد — روی fallback")

    gate = VADGate(vad, frame_ms=20, silence_duration=0.4,
                   speech_start_frames=3, speech_timeout=0)
    audio = speech_like(1.2) + make_silence(0.8)
    events = [gate.push(audio[i:i + 640]).type.value
              for i in range(0, len(audio) - 640, 640)]

    check("speech_start" in events, "تشخیص شروع گفتار")
    check("utterance_end" in events, "تشخیص پایان فرمان پس از سکوت")
    check(gate.speech_duration > 0.5, "طول گفتار", f"{gate.speech_duration:.2f}s")


def test_artifact() -> None:
    section("۶) یکپارچگی فایل صوتی (بند ۱۰)")
    pcm = speech_like(2.0)
    a = AudioArtifact(pcm=pcm, session_id="test", utterance_id="u1")
    check(abs(a.duration - 2.0) < 0.01, "طول صوت", f"{a.duration:.2f}s")
    check(len(a.sha256) == 64, "اثر انگشت SHA-256", a.sha256[:16] + "…")
    check(a.sha256 == sha256_of_bytes(pcm), "تطابق hash با داده خام")

    b = AudioArtifact(pcm=pcm, session_id="other", utterance_id="u2")
    check(a.sha256 == b.sha256, "ثبات hash برای داده یکسان",
          "→ تضمین ورودی یکسان برای همهٔ مدل‌ها")
    check(-50 < rms_dbfs(pcm) < -5, "سطح انرژی معقول", f"{rms_dbfs(pcm):.1f} dBFS")


async def test_session() -> None:
    section("۷) خط لولهٔ کامل نشست")
    s = get_settings()
    s.wake_word.enabled = False
    s.vad.silence_duration = 0.4
    s.audio.save_recordings = False

    events: list[str] = []
    got: list[AudioArtifact] = []

    async def on_event(e): events.append(e["event"])
    async def on_utterance(a): got.append(a)

    sess = AudioSession(settings=s, on_event=on_event, on_utterance=on_utterance)
    await sess.start()
    print(f"     VAD={sess._vad_gate.vad.name}  |  Wake={sess._wake.name}")

    await sess.feed(speech_like(1.5))
    await sess.feed(make_silence(1.0))

    check("session_started" in events, "رویداد شروع نشست")
    check("speech_start" in events, "رویداد شروع گفتار")
    check("utterance_ready" in events, "رویداد آماده شدن فرمان")
    ok = check(len(got) == 1, "تولید دقیقاً یک artifact", f"{len(got)} عدد")

    if ok:
        a = got[0]
        check(1.0 < a.duration < 2.5, "طول فرمان نهایی", f"{a.duration:.2f}s")
        check(a.metadata.get("reason") == "utterance_end", "دلیل پایان",
              str(a.metadata.get("reason")))
    else:
        print("     وضعیت VAD:", sess._vad_gate.status())

    st = sess.status()
    check(st["stats"]["frames_received"] > 100, "تعداد فریم پردازش‌شده",
          str(st["stats"]["frames_received"]))
    sess.close()


async def test_preroll() -> None:
    section("۸) الصاق Pre-Roll پس از Wake Word (بند ۸)")
    s = get_settings()
    s.wake_word.enabled = True
    s.wake_word.engine = "manual"
    s.vad.silence_duration = 0.4
    s.audio.save_recordings = False

    got: list[AudioArtifact] = []
    async def on_event(e): pass
    async def on_utterance(a): got.append(a)

    sess = AudioSession(settings=s, on_event=on_event, on_utterance=on_utterance)
    await sess.start()
    check(sess.state.value == "waiting_wake", "حالت انتظار کلمهٔ بیدارباش")

    await sess.feed(speech_like(1.0))        # پیش از trigger → وارد Pre-Roll
    await sess.trigger_wake()                # فعال‌سازی دستی
    check(sess.state.value == "listening", "گذار به حالت ضبط")

    await sess.feed(speech_like(1.0))
    await sess.feed(make_silence(1.0))

    if check(len(got) == 1, "تولید artifact پس از Wake Word"):
        a = got[0]
        check(a.pre_roll_seconds > 0.5, "الصاق Pre-Roll",
              f"{a.pre_roll_seconds:.2f}s از صوت قبل از trigger حفظ شد")
        check(a.duration > 1.4, "طول کل (Pre-Roll + فرمان)", f"{a.duration:.2f}s")
    sess.close()


# ═══════════════════════════════════════════════════════════════════════════
async def main() -> int:
    print("\n\033[1m🎙  تست دود سامانه Benchmark آفلاین ASR فارسی\033[0m")
    test_config()
    test_normalizer()
    test_metrics()
    test_buffers()
    test_vad()
    test_artifact()
    await test_session()
    await test_preroll()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    section("خلاصه")
    print(f"  {passed}/{total} تست موفق")
    failed = [t for ok, t in _results if not ok]
    if failed:
        print("\n  موارد ناموفق:")
        for t in failed:
            print(f"    {FAIL} {t}")
        return 1
    print(f"\n  {PASS} \033[1mهمهٔ تست‌ها موفق — فاز ۱ و ۲ تأیید شد\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
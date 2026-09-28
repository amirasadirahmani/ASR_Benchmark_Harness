# Changelog

همهٔ تغییرات مهم این پروژه در این فایل ثبت می‌شوند.

## [1.0.0] - 2026-09-28

### Added

- دو حالت مستقل **Assistant Mode** و **Benchmark Mode**.
- رابط وب فارسی RTL با Bootstrap محلی و بدون CDN.
- Endpoint مستقل `POST /api/benchmark/run`.
- اسکریپت‌های:
  - `scripts/setup_models.py`
  - `scripts/offline_self_test.py`
  - `scripts/enroll_wakeword.py`
- Model Registry قابل توسعه برای تعداد نامحدود مدل.
- Worker Process مستقل برای هر مدل.
- اندازه‌گیری WER، CER، RTF، Timing و RAM.
- Canonical Audio با `PCM16 / mono / 16kHz`.
- Ring Buffer / Pre-roll برای حفظ Wake Word در Recording.
- VAD با توقف پیش‌فرض پس از ۲ ثانیه سکوت.
- Result storage به‌صورت JSON و CSV.
- `model_manifest.json` برای Model Lock و provenance.
- Regression Suite برای باگ‌های Integration.
- Specification نهایی در `docs/FINAL_PROPOSAL.md`.

### Changed

- Runtime به localhost-only و Offline-first تبدیل شد.
- Hugging Face model setup اکنون ابتدا immutable commit SHA را resolve می‌کند و سپس همان SHA را دانلود می‌کند.
- Vosk/Silero با SHA-256 قفل می‌شوند.
- Rotation ترتیب مدل‌ها بین اجراهای مستقل فعال شد.
- Audio Upload غیر-canonical قبل از Worker به نسخهٔ canonical تبدیل می‌شود.
- Session بعد از Benchmark دوباره به Listening برمی‌گردد.
- Preflight فقط مدل‌های واقعاً runnable را آماده گزارش می‌کند.
- مدل‌های نصب‌نشده در UI disabled می‌شوند.
- پیام بدون Reference دقیقاً از Backend در UI نمایش داده می‌شود.
- Offline Self-Test به دو سطح `Benchmark Ready` و `Assistant Ready` تفکیک شد.
- روی macOS، Vosk به نسخهٔ دارای wheel سازگار با Apple Silicon pin شد.

### Fixed

- `set_reference` و `apply_config_override` گمشده در AudioSession.
- wiring اشتباه Live Progress بین WebSocket و Orchestrator.
- bind شدن تصادفی روی `0.0.0.0`.
- ذخیره‌نشدن Recording به‌دلیل path اشتباه.
- reset شدن Rotation با ساخت Orchestrator جدید.
- باقی‌ماندن Session در حالت `PROCESSING`.
- ارسال WAV خام 44.1kHz/stereo به Worker.
- نشت فایل‌های Upload/Canonical/Warmup در `data/temp`.
- انتخاب «صفر مدل» که قبلاً به «همه مدل‌ها» تبدیل می‌شد.
- Self-Test ناقص Wake Word که fallback را Ready حساب می‌کرد.
- آلودگی Working Tree توسط `model_manifest.json` آزمایشی.
- setup موفق با `resolved_revision = null`.
- هشدار deprecated مربوط به `huggingface_hub.snapshot_download`.

### Validation

در نسخهٔ Code Freeze:

```text
pytest -q                  -> 21 passed
python -m tests.smoke_test -> 50/50
```

مرحلهٔ بعد، Integration Test واقعی روی MacBook M1 Pro با Base/Small، میکروفون واقعی و شبکهٔ خاموش است.

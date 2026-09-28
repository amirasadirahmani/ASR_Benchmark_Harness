# پروپوزال نهایی سامانهٔ آفلاین محک ASR فارسی

> **نسخه:** 1.0  
> **وضعیت:** Code Freeze / آماده برای Integration Test روی MacBook M1 Pro  
> **مرجع کد:** `main`  
> **تاریخ:** 2026-09-28

## 1. هدف

این پروژه یک ابزار **کاملاً محلی و آفلاین** برای ارزیابی علمی مدل‌های تشخیص گفتار فارسی است. سامانه یک Audio واحد را به چند مدل ASR می‌دهد و دقت، سرعت و مصرف منابع آن‌ها را تحت شرایط یکسان مقایسه می‌کند.

نسخهٔ 1.0 با دو مدل اصلی شروع می‌شود:

- `aictsharif/whisper-base-fa`
- `aictsharif/whisper-small-fa`

هر دو برای Runtime اصلی به **CTranslate2 / faster-whisper / INT8** تبدیل می‌شوند. معماری Registry-based است و افزودن مدل‌های بیشتر نباید نیازمند بازنویسی UI یا Benchmark Core باشد.

## 2. اهداف فنی

- مقایسهٔ WER و CER روی Audio یکسان.
- اندازه‌گیری `Inference Time`، `RTF`، `Peak RAM` و زمان Load.
- جداسازی کامل **Assistant Mode** از **Benchmark Mode**.
- اجرای آفلاین بعد از Setup اولیهٔ مدل‌ها.
- رابط وب فارسی، RTL و بدون CDN.
- اجرای هر مدل در Worker Process مستقل.
- ثبت نسخهٔ immutable مدل‌ها برای بازتولید Benchmark روی Raspberry Pi 5.

## 3. معماری

```text
Browser (Persian RTL UI)
        │
        │ REST + WebSocket
        ▼
FastAPI / API Layer
        │
        ├── Assistant Session Manager
        │       ├── Browser microphone
        │       ├── PCM16 / 16kHz / mono
        │       ├── Ring Buffer / Pre-roll
        │       ├── Wake Word Detector
        │       └── VAD / 2s silence
        │
        └── Benchmark Orchestrator
                ├── Model Registry
                ├── Rotation Scheduler
                ├── Worker Process: Base
                ├── Worker Process: Small
                └── Worker Process: Model N
                        │
                        ▼
                WER / CER / Time / RTF / RAM
                        │
                        ▼
                JSON + CSV Result Store
```

Frontend تعداد مدل‌ها را Hard-code نمی‌کند. کارت‌ها و ستون‌های جدول از Registry مدل‌ها ساخته می‌شوند.

## 4. حالت‌های اجرایی

### 4.1 Assistant Mode

1. Browser با `getUserMedia()` میکروفون را دریافت می‌کند.
2. صوت به PCM16 mono 16kHz تبدیل و از WebSocket به Backend فرستاده می‌شود.
3. Ring Buffer حدود 1.5 ثانیه Pre-roll نگه می‌دارد تا خود Wake Word حذف نشود.
4. Wake Word پیش‌فرض «آرینا» تشخیص داده می‌شود.
5. ضبط فرمان ادامه پیدا می‌کند.
6. پس از **2 ثانیه سکوت پیوسته** ضبط متوقف می‌شود.
7. همان Audio canonical به مدل‌های انتخاب‌شده داده می‌شود.
8. بعد از پایان Benchmark، Session دوباره به Listening برمی‌گردد.

### 4.2 Benchmark Mode

- ورودی یک فایل WAV است.
- Wake Word و VAD در معیار ASR دخالت ندارند.
- ورودی به PCM16 / mono / 16kHz canonical می‌شود.
- همان فایل canonical به همهٔ مدل‌ها داده می‌شود.
- API اصلی: `POST /api/benchmark/run`.

## 5. رابط کاربری

UI باید:

- فارسی و RTL باشد.
- Bootstrap را از فایل Local سرو کند.
- به فونت اینترنتی یا CDN وابسته نباشد.
- TextBox متن مرجع داشته باشد.
- متن شنیده‌شدهٔ هر مدل را نمایش دهد.
- Result Cardهای Dynamic تولید کند.
- جدول مقایسهٔ چندمدلی قابل مرتب‌سازی داشته باشد.
- وضعیت Listening / Recording / Processing / Completed را نمایش دهد.
- مدل نصب‌نشده را Disable کند.
- انتخاب صفر مدل را قبل از ارسال Request رد کند.

اگر Reference خالی باشد، ASR و Performance همچنان اجرا می‌شوند ولی WER/CER محاسبه نمی‌شود و UI باید این متن را قرمز نمایش دهد:

> جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت.

## 6. Audio Contract

| پارامتر | مقدار |
|---|---|
| Sample rate | 16000 Hz |
| Channels | Mono |
| Sample format | PCM16 |
| Frame | 20 ms |
| Pre-roll | 1.5 s |
| Silence timeout | 2.0 s |
| Max utterance | 30 s |

Workerها فقط Audio canonical را دریافت می‌کنند.

## 7. Wake Word و VAD

### Wake Word

- Phrase: `آرینا`
- Engine پیش‌فرض: Vosk فارسی
- Engineهای جایگزین: DTW / Manual
- مدل Baseline: `vosk-model-small-fa-0.5`

### VAD

- Runtime پیش‌فرض: WebRTC VAD
- Silence Duration: 2s
- Silero ONNX: موتور جایگزین
- EnergyVAD: fallback

## 8. مدل‌های نسخهٔ 1.0

| Model ID | Source | Runtime | Compute |
|---|---|---|---|
| `whisper-base-fa-ct2` | `aictsharif/whisper-base-fa` | faster-whisper / CT2 | INT8 CPU |
| `whisper-small-fa-ct2` | `aictsharif/whisper-small-fa` | faster-whisper / CT2 | INT8 CPU |

Adapter مربوط به Transformers مرجع و غیرفعال است. مدل Large-v3 نیز صرفاً نمونهٔ توسعهٔ Registry است و در نسخهٔ پایه فعال نیست.

## 9. Setup و Model Lock

Setup مدل‌ها تنها مرحله‌ای است که اینترنت نیاز دارد.

برای Hugging Face:

1. revision به SHA immutable resolve می‌شود.
2. اگر SHA resolve نشود، Setup Fail می‌شود.
3. Download دقیقاً از همان SHA انجام می‌شود.
4. مدل CT2 تبدیل و Quantize می‌شود.
5. SHA در `model_manifest.json` ثبت می‌شود.

نصب بعدی به‌طور پیش‌فرض از Lock قبلی استفاده می‌کند. فقط `--update-lock` نسخهٔ جدید می‌گیرد.

برای Vosk و Silero، SHA-256 فایل دانلودی ثبت و با Lock قبلی مقایسه می‌شود.

> SHA به‌تنهایی امکان بازسازی آفلاین فایل حذف‌شده از upstream را نمی‌دهد؛ برای بازتولید بلندمدت، snapshotهای مدل نیز باید آرشیو شوند.

## 10. روش Benchmark

- هر مدل در Worker Process مستقل اجرا می‌شود.
- مدل‌ها به‌صورت Sequential اجرا می‌شوند.
- ترتیب اجرا بین Runها Rotate/Shuffle می‌شود.
- Warm-up از Inference اصلی جداست.
- مقدار پیش‌فرض `runs_per_model = 3` است.
- هر مدل دقیقاً همان Audio canonical را می‌گیرد.
- `Inference Time` با Model Load Time مخلوط نمی‌شود.

## 11. متریک‌ها

### Accuracy

- WER
- CER
- Exact Match

### Performance

- Model Load Time
- Warm-up Time
- Inference Time
- Total Time
- RTF

### Resources

- Baseline RSS
- Peak RAM
- Model RAM
- CPU usage

### Auditability

- Audio SHA-256
- Model immutable revision/hash
- Execution order
- Settings snapshot
- Environment metadata

## 12. Normalization فارسی

Normalization فقط اختلاف‌های نوشتاری غیرمعنایی را حذف می‌کند:

- ی/ک عربی و فارسی
- Unicode normalization
- فاصله‌های اضافی
- نیم‌فاصله طبق Profile
- اعراب
- علائم نگارشی طبق تنظیمات

عبارات معنایی مثل «را» و «رو» نباید خودکار یکسان شوند.

## 13. آفلاین بودن

Runtime باید:

- فقط روی localhost اجرا شود.
- هیچ CDN یا API خارجی استفاده نکند.
- مدل‌ها را فقط از Local Disk Load کند.
- Hugging Face/Transformers را در Offline Mode قرار دهد.
- اتصال خارجی Python را با Offline Guard مسدود کند.

Python socket guard جایگزین sandbox شبکهٔ سیستم‌عامل نیست. Acceptance نهایی باید با Wi‑Fi/Ethernet/VPN خاموش انجام شود.

## 14. ذخیرهٔ نتایج

نتایج در `results/` نگهداری می‌شوند:

- JSON کامل هر Benchmark
- CSV تجمعی برای تحلیل آماری

هر Run باید شامل Reference، Transcription، WER/CER، Timing، RTF، RAM، Audio SHA و Model Version باشد.

Temp Upload، canonical temp و warmup باید پس از پایان یا خطا پاک شوند.

## 15. تست خودکار

معیار Code Freeze:

```bash
pytest -q
python -m tests.smoke_test
```

انتظار:

```text
21 passed
50/50
```

Regression Suite باگ‌های تاریخی مهم را پوشش می‌دهد:

- Rotation بین Orchestratorهای مستقل
- Canonical Audio
- Resume بعد از Benchmark
- Setup Model Lock
- Preflight/Dummy
- Temp Cleanup
- Handler-level integration

تست‌ها نباید `model_manifest.json` واقعی Working Tree را تغییر دهند.

## 16. معیارهای پذیرش نسخهٔ 1.0

- Base و Small با موفقیت دانلود، Lock و CT2/int8 شوند.
- Test Suite کامل سبز باشد.
- Offline Self-Test با شبکهٔ خاموش:
  - Benchmark Ready = بله
  - Assistant Ready = بله
- UI از `127.0.0.1` بدون اینترنت باز شود.
- Assistant Mode حداقل ۵ فرمان متوالی روی یک WebSocket اجرا کند.
- Wake Word «آرینا» با صدای واقعی کاربر کار کند.
- توقف ضبط پس از حدود ۲ ثانیه سکوت رخ دهد.
- Benchmark Mode همان WAV را برای Base و Small اجرا کند.
- نتایج WER/CER/RTF/RAM/زمان و model_versions ذخیره شوند.
- در حالت بدون Reference پیام دقیق قرمز نمایش داده شود.

## 17. راهنمای Integration Test روی MacBook M1 Pro

### 17.1 آماده‌سازی

```bash
uname -m
# انتظار: arm64

brew install python@3.11
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 17.2 تست Runtime

```bash
python - <<'PY'
import platform, ctranslate2, torch, onnxruntime
print("machine:", platform.machine())
print("python:", platform.python_version())
print("CT2 CPU types:", ctranslate2.get_supported_compute_types("cpu"))
print("Torch MPS available:", torch.backends.mps.is_available())
print("ONNX Runtime:", onnxruntime.__version__)
import vosk
print("Vosk: OK")
PY
```

در CT2 باید `int8` بین compute typeهای CPU دیده شود.

### 17.3 تست‌های کد

```bash
pytest -q
python -m tests.smoke_test
```

### 17.4 Setup مرحله‌ای مدل‌ها

```bash
python scripts/setup_models.py --list
python scripts/setup_models.py --model whisper-base-fa-ct2
python scripts/setup_models.py --model whisper-small-fa-ct2
python scripts/setup_models.py --wakeword vosk
python scripts/setup_models.py --vad silero
python scripts/setup_models.py --list
cat model_manifest.json
```

### 17.5 Self-Test آفلاین

پس از دانلود مدل‌ها، Wi‑Fi/Ethernet/VPN را قطع کنید:

```bash
python scripts/offline_self_test.py
echo $?
```

انتظار: exit code = 0.

### 17.6 اجرای UI

```bash
python -m backend.main
```

سپس:

<http://127.0.0.1:8000>

در macOS دسترسی Microphone مرورگر را در Privacy & Security بررسی کنید.

### 17.7 سناریوی Assistant Mode

1. Reference: «آرینا چراغ رو خاموش کن».
2. Base و Small را انتخاب کنید.
3. Session را شروع کنید.
4. «آرینا چراغ رو خاموش کن» را بگویید.
5. بیش از ۲ ثانیه سکوت کنید.
6. نتیجهٔ هر دو مدل را بررسی کنید.
7. بدون Reload صفحه حداقل ۵ فرمان متوالی اجرا کنید.
8. یک بار Reference را خالی بگذارید و پیام قرمز را بررسی کنید.
9. Manual Trigger را هم آزمایش کنید.

### 17.8 Benchmark واقعی

حداقل ۱۰ تا ۲۰ جمله را با شرایط صوتی ثابت آزمایش کنید. برای هر جمله فقط یک Recording تولید و همان Audio را برای هر دو مدل استفاده کنید.

نمونه‌ها:

- آرینا چراغ رو خاموش کن
- آرینا چراغ رو روشن کن
- آرینا صدا رو کم کن
- آرینا صدا رو زیاد کن
- آرینا امروز چندمه
- آرینا امروز چندشنبس
- آرینا چراغ رو بچرخون سمت چپ
- آرینا چراغ رو بچرخون سمت راست
- آرینا پشیمون شدم بیار سر جاش

برای تصمیم Raspberry Pi فقط WER ملاک نیست؛ WER/CER، RTF، Peak RAM و پایداری اجرای آفلاین باید با هم ارزیابی شوند.

## 18. توسعه‌های آینده

- افزودن مدل سوم تا N از طریق Registry/Adapter.
- اضافه‌کردن Intent Accuracy و Slot Accuracy پس از NLU.
- انتقال همان Dataset و Model Lock به Raspberry Pi 5.
- اضافه‌کردن Browser E2E tests.
- آرشیو artifactهای مدل برای بازتولید بلندمدت مستقل از upstream.

## 19. وضعیت سند

این فایل Specification مرجع نسخهٔ 1.0 است. رفع باگ‌هایی که Scope یا معماری را تغییر نمی‌دهند باید در CHANGELOG ثبت شوند و نیازمند بازنویسی Proposal نیستند.

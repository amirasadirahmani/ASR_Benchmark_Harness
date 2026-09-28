# آرینا — Offline Persian ASR Benchmark Harness

ابزار محلی و آفلاین برای مقایسهٔ علمی مدل‌های تشخیص گفتار فارسی از نظر **دقت، سرعت و مصرف حافظه**.

نسخهٔ 1.0 دو مسیر مستقل دارد:

- **Assistant Mode** — میکروفون زنده، Wake Word «آرینا»، VAD و توقف پس از ۲ ثانیه سکوت.
- **Benchmark Mode** — اجرای مستقیم یک WAV واحد روی همهٔ مدل‌ها، بدون دخالت Wake Word/VAD.

پس از Setup اولیهٔ مدل‌ها، Runtime پروژه بدون اینترنت کار می‌کند. UI و Bootstrap از فایل‌های Local سرو می‌شوند و هیچ CDN در Runtime لازم نیست.

## مستندات

- **Specification نهایی و راهنمای کامل M1 Pro:** [docs/FINAL_PROPOSAL.md](docs/FINAL_PROPOSAL.md)
- **تاریخچهٔ تغییرات:** [CHANGELOG.md](CHANGELOG.md)
- **Registry مدل‌ها:** [backend/config/models.yaml](backend/config/models.yaml)
- **تنظیمات برنامه:** [backend/config/app.yaml](backend/config/app.yaml)

## وضعیت نسخهٔ 1.0

- `pytest -q` → **21 passed**
- `python -m tests.smoke_test` → **50/50**
- Model Lock با immutable Hugging Face SHA
- Worker isolation برای هر مدل
- Canonical Audio: `PCM16 / mono / 16kHz`
- Rotation ترتیب مدل‌ها بین Runها
- UI فارسی RTL
- Runtime محلی و localhost-only

## مدل‌های اصلی

| ID | Source | Runtime | Compute |
|---|---|---|---|
| `whisper-base-fa-ct2` | `aictsharif/whisper-base-fa` | faster-whisper / CTranslate2 | INT8 CPU |
| `whisper-small-fa-ct2` | `aictsharif/whisper-small-fa` | faster-whisper / CTranslate2 | INT8 CPU |

Wake Word پیش‌فرض با Vosk فارسی و phrase «آرینا» اجرا می‌شود.

## نصب سریع — macOS Apple Silicon

Python 3.11 یا 3.12 توصیه می‌شود.

```bash
git clone git@github.com:amirasadirahmani/ASR_Benchmark_Harness.git
cd ASR_Benchmark_Harness

brew install python@3.11
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` برای macOS از `vosk==0.3.43` و برای سایر سیستم‌ها از `vosk==0.3.45` استفاده می‌کند.

## تست قبل از دانلود مدل

```bash
pytest -q
python -m tests.smoke_test
```

انتظار:

```text
21 passed
50/50
```

## Setup مدل‌ها — تنها مرحلهٔ آنلاین

برای اولین Setup بهتر است مرحله‌ای جلو بروید:

```bash
python scripts/setup_models.py --list

python scripts/setup_models.py --model whisper-base-fa-ct2
python scripts/setup_models.py --model whisper-small-fa-ct2
python scripts/setup_models.py --wakeword vosk
python scripts/setup_models.py --vad silero

python scripts/setup_models.py --list
cat model_manifest.json
```

یا همه با هم:

```bash
python scripts/setup_models.py --all
```

اسکریپت ابتدا Hugging Face revision را به SHA immutable resolve می‌کند و سپس دقیقاً همان SHA را دانلود می‌کند. اگر resolve ناموفق باشد Setup موفق اعلام نمی‌شود. Lock در `model_manifest.json` ثبت می‌شود.

برای ارتقای عمدی مدل‌ها:

```bash
python scripts/setup_models.py --all --update-lock
```

> داشتن SHA به‌تنهایی برای نصب مجدد آفلاین کافی نیست؛ فایل snapshot/model نیز باید از قبل روی سیستم یا cache موجود باشد.

## Self-Test آفلاین

```bash
python scripts/offline_self_test.py
echo $?
```

| Exit code | معنی |
|---:|---|
| `0` | Benchmark Ready + Assistant Ready |
| `2` | Benchmark Ready، Assistant Mode ناقص |
| `1` | Benchmark Mode هم آماده نیست |

برای Acceptance نهایی، Wi‑Fi/Ethernet/VPN را قطع کنید و همین تست را دوباره اجرا کنید.

## اجرا

```bash
python -m backend.main
```

سپس:

<http://127.0.0.1:8000>

سرور به‌صورت پیش‌فرض فقط روی localhost اجرا می‌شود.

## ساختار پروژه

```text
ASR_Benchmark_Harness/
├── backend/
│   ├── api/            # REST/WebSocket
│   ├── asr/            # Registry + adapters
│   ├── audio/          # Wake word, VAD, session, canonical audio
│   ├── benchmark/      # Orchestrator, worker, RAM monitor
│   ├── config/         # app.yaml + models.yaml
│   ├── core/           # Persian normalization + WER/CER
│   ├── storage/        # JSON/CSV results
│   └── main.py
├── frontend/
│   ├── css/
│   ├── js/
│   └── vendor/         # Bootstrap local
├── scripts/
│   ├── setup_models.py
│   ├── offline_self_test.py
│   └── enroll_wakeword.py
├── tests/
│   ├── smoke_test.py
│   ├── test_regressions.py
│   └── conftest.py
├── docs/
│   └── FINAL_PROPOSAL.md
├── CHANGELOG.md
├── pytest.ini
└── requirements.txt
```

## خروجی Benchmark

نتایج در `results/` ذخیره می‌شوند:

- JSON کامل هر Benchmark
- CSV تجمعی برای تحلیل آماری

متریک‌های اصلی:

- WER / CER
- Inference Time
- RTF
- Peak RAM
- Load/Warmup Time
- Audio SHA-256
- Model revision/hash
- Execution order

## افزودن مدل جدید

1. مدل را در `backend/config/models.yaml` ثبت کنید.
2. اگر Runtime جدید است، Adapter مربوط را در `backend/asr/adapters/` اضافه کنید.
3. Core Benchmark و UI نباید بر اساس تعداد مدل‌ها تغییر کنند.

## فایل‌های تولیدی

این مسیرها نباید commit شوند:

- `models/`
- `recordings/`
- `results/`
- `logs/`
- `data/temp/`
- `.cache/`
- `.pytest_cache/`

`model_manifest.json` استثناست: پس از Setup واقعی، برای reproducibility بهتر است نسخهٔ معتبر آن همراه کد نگهداری شود.

## مرحلهٔ بعد

مرحلهٔ بعدی **Integration Test واقعی روی MacBook M1 Pro** است: نصب Base/Small، تبدیل CT2/int8، قطع شبکه، اجرای Self-Test، تست Wake Word و VAD با میکروفون واقعی و Benchmark حداقل ۱۰ تا ۲۰ جمله.

پروتکل کامل در [docs/FINAL_PROPOSAL.md](docs/FINAL_PROPOSAL.md) آمده است.

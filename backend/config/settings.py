from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------------
# مسیرهای پایه
# ----------------------------------------------------------------------------
BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

_ENV_PREFIX = "ASRB_"          # مثال: ASRB_AUDIO__SAMPLE_RATE=8000
_ENV_NESTED_SEP = "__"


# ----------------------------------------------------------------------------
# بخش‌های تنظیمات
# ----------------------------------------------------------------------------
class ServerSettings(BaseModel):
    host: str = "127.0.0.1"           # عمداً localhost-only (بند ۴)
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    # اگر True باشد، سرور اجازه bind روی 0.0.0.0 را می‌دهد (پیش‌فرض: ممنوع)
    allow_external_bind: bool = False


class PathSettings(BaseModel):
    models_dir: Path = Path("models")
    recordings_dir: Path = Path("recordings")
    results_dir: Path = Path("results")
    frontend_dir: Path = Path("frontend")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path(".cache")

    def resolve_all(self, root: Path) -> "PathSettings":
        """تبدیل مسیرهای نسبی به مطلق نسبت به ریشه پروژه."""
        data = {}
        for name, value in self.model_dump().items():
            p = Path(value)
            data[name] = p if p.is_absolute() else (root / p)
        return PathSettings(**data)

    def ensure_dirs(self) -> None:
        for name, value in self.model_dump().items():
            if name == "frontend_dir":
                continue  # frontend باید از قبل وجود داشته باشد
            Path(value).mkdir(parents=True, exist_ok=True)


class AudioSettings(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2                 # int16
    frame_ms: int = 20                    # اندازه فریم ورودی از مرورگر
    max_command_seconds: float = 30.0     # سقف ایمنی طول فرمان
    min_command_seconds: float = 0.30     # کمتر از این => «گفتاری یافت نشد»
    pre_roll_seconds: float = 1.5         # بند ۸
    save_recordings: bool = True
    recording_format: Literal["wav"] = "wav"

    @field_validator("frame_ms")
    @classmethod
    def _valid_frame(cls, v: int) -> int:
        if v not in (10, 20, 30):
            raise ValueError("frame_ms باید 10، 20 یا 30 باشد (محدودیت WebRTC VAD)")
        return v

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.sample_width * self.channels


class VADSettings(BaseModel):
    engine: Literal["webrtc", "silero", "energy"] = "webrtc"
    aggressiveness: int = Field(2, ge=0, le=3)       # فقط webrtc
    silence_duration: float = 2.0                    # بند ۹ — SILENCE_DURATION
    speech_start_frames: int = 3                     # چند فریم متوالی = شروع گفتار
    energy_threshold_dbfs: float = -45.0             # فقط موتور energy    
    silero_model_path: Path = Path("models/vad/silero_vad.onnx")
    # اگر تا این مدت بعد از Wake Word هیچ گفتاری نیامد => timeout
    post_wake_speech_timeout: float = 6.0


class WakeWordSettings(BaseModel):
    enabled: bool = True
    word: str = "آرینا"                               # بند ۱۹ — WAKE_WORD
    engine: Literal["vosk", "dtw", "manual"] = "vosk"
    # لیست تلفظ‌های پذیرفته‌شده (بعد از نرمال‌سازی) برای موتور مبتنی بر متن
    aliases: list[str] = Field(default_factory=lambda: ["آرینا", "ارینا", "آرینه", "ارینا"])
    vosk_model_path: Path = Path("models/wakeword/vosk-model-small-fa")
    dtw_templates_dir: Path = Path("models/wakeword/dtw_templates")
    dtw_threshold: float = 0.42
    cooldown_seconds: float = 1.5                    # جلوگیری از trigger مکرر
    # تایم‌اوت انتظار برای Wake Word (بند ۲۳) — 0 یعنی بی‌نهایت
    listen_timeout: float = 0.0


class BenchmarkSettings(BaseModel):
    mode: Literal["sequential", "parallel"] = "sequential"   # بند ۱۲
    runs_per_utterance: int = 1          # در UI قابل تغییر؛ توصیه ۳ تا ۵ (بند ۲۲)
    warmup_enabled: bool = True
    warmup_audio_seconds: float = 1.0
    rotate_model_order: bool = True      # بند ۱۳ — جلوگیری از bias ترتیب
    rotation_strategy: Literal["rotate", "shuffle"] = "rotate"
    worker_timeout_seconds: float = 300.0
    isolate_workers: bool = True         # هر مدل در پروسه مستقل
    sample_ram_interval: float = 0.05    # فاصله نمونه‌برداری Peak RSS
    runs_per_model: int = 1               # تعداد تکرار برای میانگین‌گیری
    warmup_enabled: bool = True
    warmup_seconds: float = 1.0
    model_timeout: float = 300.0          # مهلت هر مدل (ثانیه)
    mp_context: str = "spawn"             # هرگز fork
    ram_sample_interval: float = 0.05
    isolated: bool = True

class NormalizerSettings(BaseModel):
    profile: Literal["default", "strict", "light"] = "default"
    zwnj_policy: Literal["keep", "space", "remove"] = "space"
    remove_punctuation: bool = True
    unify_alef_hamza: bool = True
    unify_alef_madda: bool = False       # آ -> ا (پیش‌فرض خاموش: تفاوت معنایی)
    digits_to: Literal["ascii", "persian", "keep"] = "ascii"
    remove_diacritics: bool = True


class UISettings(BaseModel):
    title: str = "سامانه مقایسه مدل‌های تشخیص گفتار فارسی"
    language: str = "fa"
    direction: Literal["rtl", "ltr"] = "rtl"
    no_reference_message: str = "جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت."
    show_raw_transcription: bool = True


class AppSettings(BaseModel):
    """ریشه تنظیمات."""
    offline_mode: bool = True            # بند ۱۹ — OFFLINE_MODE
    language: str = "fa"                 # بند ۱۹ — LANGUAGE
    strict_offline_guard: bool = True    # مسدودسازی فعال socket خارجی در runtime

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    wake_word: WakeWordSettings = Field(default_factory=WakeWordSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    normalizer: NormalizerSettings = Field(default_factory=NormalizerSettings)
    ui: UISettings = Field(default_factory=UISettings)

    models_config_file: Path = Path("backend/config/models.yaml")

    # ------------------------------------------------------------------
    def absolute(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


# ----------------------------------------------------------------------------
# بارگذاری / ادغام
# ----------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(raw: str) -> Any:
    """تبدیل مقدار رشته‌ای ENV به نوع پایتونی."""
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _env_overrides() -> Dict[str, Any]:
    """
    ASRB_OFFLINE_MODE=false
    ASRB_VAD__SILENCE_DURATION=1.5
    ASRB_WAKE_WORD__WORD=آرینا
    """
    result: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        path = key[len(_ENV_PREFIX):].lower().split(_ENV_NESTED_SEP)
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"ساختار فایل تنظیمات نامعتبر است: {path}")
    return data


_settings_cache: Optional[AppSettings] = None


def get_settings(config_path: Optional[Path] = None) -> AppSettings:
    """تنظیمات را (با کش) برمی‌گرداند."""
    global _settings_cache
    if _settings_cache is not None and config_path is None:
        return _settings_cache

    path = config_path or (BACKEND_DIR / "config" / "app.yaml")
    merged = _deep_merge(_load_yaml(path), _env_overrides())
    settings = AppSettings(**merged)

    # مسیرها را مطلق کن و بساز
    settings.paths = settings.paths.resolve_all(PROJECT_ROOT)
    settings.paths.ensure_dirs()

    if config_path is None:
        _settings_cache = settings
    return settings


def reload_settings(config_path: Optional[Path] = None) -> AppSettings:
    global _settings_cache
    _settings_cache = None
    return get_settings(config_path)


# ----------------------------------------------------------------------------
# گارد آفلاین (بند ۴)
# ----------------------------------------------------------------------------
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def ensure_offline_env(strict_socket_guard: bool = False) -> None:
    """
    باید در ابتدای main.py و ابتدای هر Worker Process صدا زده شود،
    *قبل* از import کتابخانه‌های HuggingFace.
    """
    for k, v in _OFFLINE_ENV.items():
        os.environ.setdefault(k, v)

    if strict_socket_guard:
        _install_socket_guard()


_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _install_socket_guard() -> None:
    """
    اتصال TCP به هر مقصدی جز localhost را در runtime مسدود می‌کند.
    این تضمین سخت‌افزاریِ «هیچ Runtime Dependency به اینترنت» است.
    """
    import socket

    if getattr(socket.socket, "_asrb_guarded", False):
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _check(address: Any) -> None:
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if host not in _ALLOWED_HOSTS:
                raise OSError(
                    f"[OFFLINE GUARD] اتصال شبکه‌ای به «{host}» در حالت آفلاین مسدود است."
                )

    def guarded_connect(self, address):          # type: ignore[no-untyped-def]
        _check(address)
        return original_connect(self, address)

    def guarded_connect_ex(self, address):       # type: ignore[no-untyped-def]
        _check(address)
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect          # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex    # type: ignore[assignment]
    socket.socket._asrb_guarded = True               # type: ignore[attr-defined]
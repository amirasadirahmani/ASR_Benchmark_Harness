from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config.settings import PROJECT_ROOT

# ---------------------------------------------------------------------------

DeviceType = Literal["auto", "cpu", "cuda", "mps"]
ComputeType = Literal[
    "int8", "int8_float16", "int8_bfloat16",
    "float16", "bfloat16", "float32", "default",
]


class ModelSource(BaseModel):
    """
    محل تهیه مدل.
    فقط توسط scripts/setup_models.py استفاده می‌شود؛ هرگز در Runtime.
    """
    type: Literal["huggingface", "local", "url"] = "local"
    repo_id: Optional[str] = None
    url: Optional[str] = None
    revision: Optional[str] = None
    convert: Literal["ct2", "none"] = "none"
    quantization: Optional[str] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "ModelSource":
        if self.type == "huggingface" and not self.repo_id:
            raise ValueError("برای منبع huggingface، مقدار repo_id الزامی است.")
        if self.type == "url" and not self.url:
            raise ValueError("برای منبع url، مقدار url الزامی است.")
        return self


class ModelConfig(BaseModel):
    """
    پیکربندی یک مدل ASR.

    افزودن مدل جدید = افزودن یک بلاک به backend/config/models.yaml
    (بند ۱۱ پروپوزال). هیچ تغییری در UI/Metrics/CSV/Orchestrator لازم نیست.
    """

    id: str
    display_name: str
    enabled: bool = True
    runtime: str                       # کلید ثبت‌شده در AdapterRegistry
    local_path: Path
    device: DeviceType = "auto"
    compute_type: ComputeType = "int8"
    order: int = 100
    source: ModelSource = Field(default_factory=ModelSource)
    params: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    # حدود ایمنی اختیاری برای هشدار روی RPi5
    expected_ram_mb: Optional[float] = None
    min_free_ram_mb: Optional[float] = None

    # -------------------------------------------------- validators
    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("شناسه مدل نمی‌تواند خالی باشد.")
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError(
                f"شناسه مدل «{v}» نامعتبر است؛ فقط حروف/عدد و - _ . مجاز است."
            )
        return v

    @field_validator("local_path", mode="before")
    @classmethod
    def _as_path(cls, v: Any) -> Path:
        return Path(v)

    # -------------------------------------------------- helpers
    @property
    def absolute_path(self) -> Path:
        """مسیر مطلق مدل روی دیسک."""
        p = self.local_path
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def exists(self) -> bool:
        """آیا فایل‌های مدل روی دیسک موجود است؟"""
        p = self.absolute_path
        if not p.exists():
            return False
        if p.is_dir():
            return any(p.iterdir())
        return True

    def resolve_device(self) -> str:
        """
        تبدیل device='auto' به دستگاه واقعی، بدون import کردن torch.

        استراتژی:
          - اگر کاربر صریحاً دستگاهی داده، همان برگردد.
          - در غیر این صورت 'cpu' (امن‌ترین و تنها گزینهٔ قطعی روی RPi5).
            آداپتورهایی که شتاب‌دهنده دارند (مثل transformers/mps) خودشان
            در زمان بارگذاری ارتقا می‌دهند.
        """
        if self.device != "auto":
            return self.device
        return "cpu"

    def effective_compute_type(self) -> str:
        """
        اصلاح compute_type ناسازگار با CPU.
        float16 روی CPU در CTranslate2 پشتیبانی نمی‌شود → به int8 تنزل می‌یابد.
        """
        device = self.resolve_device()
        if device == "cpu" and self.compute_type in ("float16", "bfloat16"):
            return "int8"
        return self.compute_type

    def safe_cpu_threads(self) -> int:
        """تعداد thread پیشنهادی؛ اگر در params نبود، بر اساس سخت‌افزار."""
        if "cpu_threads" in self.params:
            return int(self.params["cpu_threads"])
        try:
            import os
            cores = os.cpu_count() or 4
        except Exception:
            cores = 4
        return max(1, min(cores, 4))

    def to_public_dict(self) -> Dict[str, Any]:
        """نمایش امن برای ارسال به UI (بدون جزئیات مسیر محلی)."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "runtime": self.runtime,
            "device": self.resolve_device(),
            "compute_type": self.effective_compute_type(),
            "order": self.order,
            "available": self.exists(),
            "notes": self.notes,
        }

    def to_worker_payload(self) -> Dict[str, Any]:
        """
        دیکشنری کاملاً picklable برای ارسال به Worker Process.
        (spawn روی macOS نیاز دارد همه چیز serializable باشد)
        """
        return {
            "id": self.id,
            "display_name": self.display_name,
            "runtime": self.runtime,
            "model_path": str(self.absolute_path),
            "device": self.resolve_device(),
            "compute_type": self.effective_compute_type(),
            "params": dict(self.params),
            "cpu_threads": self.safe_cpu_threads(),
        }


# ---------------------------------------------------------------------------
# بارگذاری رجیستری
# ---------------------------------------------------------------------------
class ModelRegistryFile(BaseModel):
    models: List[ModelConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ModelRegistryFile":
        seen: set[str] = set()
        for m in self.models:
            if m.id in seen:
                raise ValueError(f"شناسه مدل تکراری در models.yaml: «{m.id}»")
            seen.add(m.id)
        return self


def load_model_configs(
    path: Optional[Path] = None,
    *,
    enabled_only: bool = False,
    available_only: bool = False,
) -> List[ModelConfig]:
    """
    رجیستری مدل‌ها را از YAML می‌خواند و بر اساس order مرتب برمی‌گرداند.

    Args:
        path: مسیر فایل YAML (پیش‌فرض backend/config/models.yaml)
        enabled_only: فقط مدل‌های enabled=true
        available_only: فقط مدل‌هایی که فایل‌هایشان روی دیسک موجود است
    """
    if path is None:
        path = PROJECT_ROOT / "backend" / "config" / "models.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"فایل رجیستری مدل‌ها یافت نشد: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(
            f"ساختار models.yaml نامعتبر است؛ کلید ریشه «models» الزامی است: {path}"
        )

    registry = ModelRegistryFile(**raw)
    models = sorted(registry.models, key=lambda m: (m.order, m.id))

    if enabled_only:
        models = [m for m in models if m.enabled]
    if available_only:
        models = [m for m in models if m.exists()]

    return models


def get_model_config(model_id: str, path: Optional[Path] = None) -> ModelConfig:
    """یک مدل مشخص را بر اساس شناسه برمی‌گرداند."""
    for m in load_model_configs(path):
        if m.id == model_id:
            return m
    raise KeyError(f"مدلی با شناسه «{model_id}» در رجیستری یافت نشد.")


def describe_environment() -> Dict[str, Any]:
    """اطلاعات محیط اجرا — برای درج در گزارش نتایج (بند ۲۱)."""
    import os
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_mb"] = round(vm.total / (1024 ** 2), 1)
        info["available_ram_mb"] = round(vm.available / (1024 ** 2), 1)
    except Exception:
        pass
    return info
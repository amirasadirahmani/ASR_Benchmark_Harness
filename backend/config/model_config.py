from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config.settings import PROJECT_ROOT

DeviceType = Literal["auto", "cpu", "cuda", "mps"]
ComputeType = Literal[
    "int8", "int8_float16", "int8_bfloat16",
    "float16", "bfloat16", "float32", "default",
]


class ModelSource(BaseModel):
    """محل تهیه مدل؛ فقط در setup استفاده می‌شود، نه Runtime."""
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
    """پیکربندی یک مدل ASR."""

    id: str
    display_name: str
    enabled: bool = True
    runtime: str
    local_path: Path
    device: DeviceType = "auto"
    compute_type: ComputeType = "int8"
    order: int = 100
    source: ModelSource = Field(default_factory=ModelSource)
    params: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    expected_ram_mb: Optional[float] = None
    min_free_ram_mb: Optional[float] = None

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

    @property
    def absolute_path(self) -> Path:
        p = self.local_path
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def exists(self) -> bool:
        p = self.absolute_path
        if not p.exists():
            return False
        if p.is_dir():
            return any(p.iterdir())
        return True

    _CPU_ONLY_RUNTIMES = frozenset({"faster_whisper", "dummy"})

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if self.runtime in self._CPU_ONLY_RUNTIMES:
            return "cpu"
        return "auto"

    def effective_compute_type(self) -> str:
        device = self.resolve_device()
        if device == "cpu" and self.compute_type in ("float16", "bfloat16"):
            return "int8"
        return self.compute_type

    def safe_cpu_threads(self) -> int:
        if "cpu_threads" in self.params:
            return int(self.params["cpu_threads"])
        try:
            import os
            cores = os.cpu_count() or 4
        except Exception:
            cores = 4
        return max(1, min(cores, 4))

    def to_public_dict(self) -> Dict[str, Any]:
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
    if path is None:
        from backend.config.settings import get_settings
        cfg_path = get_settings().models_config_file
        path = cfg_path if cfg_path.is_absolute() else PROJECT_ROOT / cfg_path
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
    for m in load_model_configs(path):
        if m.id == model_id:
            return m
    raise KeyError(f"مدلی با شناسه «{model_id}» در رجیستری یافت نشد.")


def describe_environment() -> Dict[str, Any]:
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

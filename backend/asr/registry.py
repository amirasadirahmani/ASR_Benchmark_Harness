"""
رجیستری Runtimeهای ASR — قلب گسترش‌پذیری سامانه.

افزودن Runtime جدید با register_runtime انجام می‌شود و سپس فقط runtime متناظر
در models.yaml تنظیم می‌شود.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

from backend.asr.base import ASRAdapterError, BaseASRAdapter
from backend.config.model_config import ModelConfig

_REGISTRY: Dict[str, Type[BaseASRAdapter]] = {}
_DISCOVERED = False


def register_runtime(name: str) -> Callable[[Type[BaseASRAdapter]], Type[BaseASRAdapter]]:
    key = name.strip().lower()

    def decorator(cls: Type[BaseASRAdapter]) -> Type[BaseASRAdapter]:
        if not issubclass(cls, BaseASRAdapter):
            raise TypeError(
                f"کلاس «{cls.__name__}» باید از BaseASRAdapter ارث‌بری کند."
            )
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(
                f"Runtime تکراری «{key}»: قبلاً توسط {_REGISTRY[key].__name__} ثبت شده است."
            )
        cls.runtime_name = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def discover_adapters(force: bool = False) -> None:
    global _DISCOVERED
    if _DISCOVERED and not force:
        return

    pkg_path = Path(__file__).parent / "adapters"
    if pkg_path.exists():
        for mod in pkgutil.iter_modules([str(pkg_path)]):
            if mod.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"backend.asr.adapters.{mod.name}")
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "آداپتور «%s» بارگذاری نشد: %s", mod.name, exc
                )
    _DISCOVERED = True


def available_runtimes() -> List[str]:
    discover_adapters()
    return sorted(_REGISTRY.keys())


def get_adapter_class(runtime: str) -> Type[BaseASRAdapter]:
    discover_adapters()
    key = runtime.strip().lower()
    if key not in _REGISTRY:
        raise ASRAdapterError(
            f"Runtime «{runtime}» ثبت نشده است. "
            f"Runtimeهای موجود: {', '.join(available_runtimes()) or '(هیچ)'}"
        )
    return _REGISTRY[key]


def create_adapter(config: ModelConfig) -> BaseASRAdapter:
    cls = get_adapter_class(config.runtime)
    return cls(
        model_id=config.id,
        model_path=config.absolute_path,
        device=config.resolve_device(),
        compute_type=config.effective_compute_type(),
        params=config.params,
        cpu_threads=config.safe_cpu_threads(),
    )


def create_adapter_from_payload(payload: Dict[str, Any]) -> BaseASRAdapter:
    cls = get_adapter_class(payload["runtime"])
    return cls(
        model_id=payload["id"],
        model_path=payload["model_path"],
        device=payload.get("device", "cpu"),
        compute_type=payload.get("compute_type", "int8"),
        params=payload.get("params") or {},
        cpu_threads=payload.get("cpu_threads", 4),
    )


def validate_registry(configs: List[ModelConfig]) -> Dict[str, List[str]]:
    discover_adapters()
    report: Dict[str, List[str]] = {
        "ok": [], "missing_runtime": [], "missing_files": [], "disabled": [],
    }
    for cfg in configs:
        if not cfg.enabled:
            report["disabled"].append(cfg.id)
            continue
        if cfg.runtime.strip().lower() not in _REGISTRY:
            report["missing_runtime"].append(cfg.id)
        elif cfg.runtime.strip().lower() != "dummy" and not cfg.exists():
            report["missing_files"].append(cfg.id)
        else:
            report["ok"].append(cfg.id)
    return report

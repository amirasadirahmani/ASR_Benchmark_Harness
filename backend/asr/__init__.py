"""لایهٔ تشخیص گفتار: آداپتورها، رجیستری و قرارداد مشترک."""

from backend.asr.base import (  # noqa: F401
    BaseASRAdapter,
    TranscriptionOutput,
    ASRAdapterError,
    ModelNotFoundError,
    RuntimeUnavailableError,
)
from backend.asr.registry import (  # noqa: F401
    register_runtime,
    discover_adapters,
    available_runtimes,
    get_adapter_class,
    create_adapter,
    create_adapter_from_payload,
    validate_registry,
)

__all__ = [
    "BaseASRAdapter",
    "TranscriptionOutput",
    "ASRAdapterError",
    "ModelNotFoundError",
    "RuntimeUnavailableError",
    "register_runtime",
    "discover_adapters",
    "available_runtimes",
    "get_adapter_class",
    "create_adapter",
    "create_adapter_from_payload",
    "validate_registry",
]
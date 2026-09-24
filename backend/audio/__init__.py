"""لایهٔ صوت: بافر حلقوی، VAD، Wake Word و مدیریت نشست ضبط."""

from backend.audio.audio_utils import (  # noqa: F401
    AudioArtifact,
    bytes_to_float32,
    float32_to_bytes,
    pcm16_duration,
    rms_dbfs,
    write_wav,
    read_wav,
    sha256_of_bytes,
    make_silence,
    trim_pcm,
)
from backend.audio.ring_buffer import PreRollBuffer, FrameAccumulator  # noqa: F401
from backend.audio.vad import (  # noqa: F401
    BaseVAD,
    WebRTCVAD,
    EnergyVAD,
    SileroVAD,
    VADGate,
    VADEvent,
    VADEventType,
    create_vad,
)

__all__ = [
    "AudioArtifact", "bytes_to_float32", "float32_to_bytes", "pcm16_duration",
    "rms_dbfs", "write_wav", "read_wav", "sha256_of_bytes", "make_silence", "trim_pcm",
    "PreRollBuffer", "FrameAccumulator",
    "BaseVAD", "WebRTCVAD", "EnergyVAD", "SileroVAD",
    "VADGate", "VADEvent", "VADEventType", "create_vad",
]
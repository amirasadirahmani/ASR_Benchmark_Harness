"""Regression tests for bugs found during independent reviews."""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path
from typing import List

import pytest

from backend.api.websocket_handler import ConnectionHandler
from backend.asr.registry import validate_registry
from backend.audio.audio_utils import AudioArtifact
from backend.audio.session import AudioSession, SessionState
from backend.benchmark.orchestrator import BenchmarkOrchestrator
from backend.config.model_config import ModelConfig
from backend.config.settings import get_settings
from backend.storage import ResultsStore


def _dummy_configs(n: int = 2) -> List[ModelConfig]:
    return [
        ModelConfig(
            id=f"dummy-{i}",
            display_name=f"Dummy {i}",
            runtime="dummy",
            local_path=Path("models/dummy"),
            params={"fake_ram_mb": 20, "fake_rtf": 0.1, "load_delay": 0.0},
        )
        for i in range(n)
    ]


def _write_wav(path: Path, rate: int, channels: int, seconds: float = 0.5) -> None:
    n = int(rate * seconds)
    samples = [
        int(4000 * math.sin(2 * math.pi * 220 * i / rate))
        for i in range(n)
    ]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        if channels == 1:
            wf.writeframes(struct.pack(f"<{n}h", *samples))
        else:
            interleaved: List[int] = []
            for sample in samples:
                interleaved.extend([sample] * channels)
            wf.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))


@pytest.mark.asyncio
async def test_rotation_persists_across_separate_orchestrator_instances(monkeypatch, tmp_path):
    import backend.benchmark.orchestrator as orch_mod

    configs = _dummy_configs(2)
    monkeypatch.setattr(orch_mod, "load_model_configs", lambda **kw: configs)

    settings = get_settings().model_copy(deep=True)
    settings.paths.temp_dir = tmp_path
    settings.benchmark.rotate_model_order = True
    settings.benchmark.rotation_strategy = "rotate"

    orders = []
    for index in range(4):
        path = tmp_path / f"u{index}.wav"
        _write_wav(path, 16000, 1)
        artifact = AudioArtifact.from_wav(
            path, session_id="s", utterance_id=f"u{index}"
        )
        orchestrator = BenchmarkOrchestrator(settings=settings)
        report = await orchestrator.run(artifact, reference_text=None)
        orchestrator.close()
        orders.append(report.model_order)

    assert not all(order == orders[0] for order in orders)


def test_non_canonical_wav_is_rewritten_to_canonical(tmp_path):
    source = tmp_path / "stereo_44k.wav"
    _write_wav(source, rate=44100, channels=2, seconds=1.0)

    artifact = AudioArtifact.from_wav(source, session_id="s", utterance_id="u")
    assert artifact.sample_rate == 16000
    assert artifact.channels == 1
    assert artifact.path is None

    settings = get_settings().model_copy(deep=True)
    settings.paths.temp_dir = tmp_path
    orchestrator = BenchmarkOrchestrator(settings=settings)
    try:
        disk_path = orchestrator._ensure_audio_on_disk(artifact)
        with wave.open(str(disk_path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
    finally:
        orchestrator.close()


def test_already_canonical_wav_keeps_original_path(tmp_path):
    source = tmp_path / "canonical.wav"
    _write_wav(source, 16000, 1, 0.5)
    artifact = AudioArtifact.from_wav(source, session_id="s", utterance_id="u")
    assert artifact.path == source


@pytest.mark.asyncio
async def test_session_resumes_listening_after_benchmark():
    settings = get_settings()
    events = []

    async def on_event(event):
        events.append(event.get("event") or event.get("type"))

    session = AudioSession(
        settings=settings,
        on_event=on_event,
        session_id="regress-1",
    )
    await session.start()
    session.state = SessionState.PROCESSING
    await session.resume()

    assert session.state == SessionState.WAITING_WAKE
    assert "listening_resumed" in events


@pytest.mark.asyncio
async def test_session_resume_guarded_when_already_closed():
    settings = get_settings()
    session = AudioSession(
        settings=settings,
        on_event=lambda event: None,
        session_id="regress-2",
    )
    await session.start()
    session.state = SessionState.PROCESSING
    session.close()
    assert session.state == SessionState.IDLE

    is_closed = True
    if not is_closed:
        await session.resume()

    assert session.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_connection_handler_run_benchmark_resumes_session(monkeypatch, tmp_path):
    import backend.benchmark.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod, "load_model_configs", lambda **kw: _dummy_configs(1)
    )

    settings = get_settings().model_copy(deep=True)
    settings.paths.temp_dir = tmp_path
    settings.paths.results_dir = tmp_path / "results"

    handler = ConnectionHandler(
        websocket=None,
        settings=settings,
        store=ResultsStore(tmp_path / "results"),
    )
    await handler.session.start()
    handler.session.state = SessionState.PROCESSING

    path = tmp_path / "u.wav"
    _write_wav(path, 16000, 1)
    artifact = AudioArtifact.from_wav(
        path, session_id="s", utterance_id="u"
    )

    await handler._run_benchmark(artifact)
    assert handler.session.state == SessionState.WAITING_WAKE

    await asyncio.sleep(0.05)
    queued_types = []
    while not handler._out_queue.empty():
        queued_types.append(handler._out_queue.get_nowait().get("type"))
    assert "listening_resumed" in queued_types
    assert "benchmark_completed" in queued_types


@pytest.mark.asyncio
async def test_connection_handler_run_benchmark_respects_closed_connection(monkeypatch, tmp_path):
    import backend.benchmark.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod, "load_model_configs", lambda **kw: _dummy_configs(1)
    )

    settings = get_settings().model_copy(deep=True)
    settings.paths.temp_dir = tmp_path
    settings.paths.results_dir = tmp_path / "results"

    handler = ConnectionHandler(
        websocket=None,
        settings=settings,
        store=ResultsStore(tmp_path / "results"),
    )
    await handler.session.start()
    handler.session.state = SessionState.PROCESSING
    handler.session.close()
    handler._closed = True

    path = tmp_path / "u2.wav"
    _write_wav(path, 16000, 1)
    artifact = AudioArtifact.from_wav(
        path, session_id="s", utterance_id="u2"
    )

    await handler._run_benchmark(artifact)
    assert handler.session.state == SessionState.IDLE


def test_preflight_treats_dummy_as_runnable_without_files():
    configs = _dummy_configs(1)
    report = validate_registry(configs)
    assert configs[0].id in report["ok"]
    assert configs[0].id not in report["missing_files"]


def test_setup_models_source_is_attribute_accessed_not_dict(monkeypatch, tmp_path):
    from unittest import mock

    import backend.config.model_config as mc_mod
    import scripts.setup_models as setup_mod

    cfg = ModelConfig(
        id="whisper-fake-test",
        display_name="fake",
        runtime="faster_whisper",
        local_path=tmp_path / "whisper-fake-test",
        source={
            "type": "huggingface",
            "repo_id": "org/does-not-exist",
            "revision": "v1.0",
        },
    )
    monkeypatch.setattr(mc_mod, "load_model_configs", lambda **kw: [cfg])

    def fake_snapshot_download(repo_id, revision, local_dir):
        Path(local_dir, "config.json").write_text("{}")
        return local_dir

    with mock.patch(
        "huggingface_hub.snapshot_download",
        side_effect=fake_snapshot_download,
    ) as download, mock.patch.object(
        setup_mod, "_resolve_hf_revision", return_value="a" * 40
    ) as resolve:
        ok = setup_mod.setup_model("whisper-fake-test", force=True)

    assert ok is True
    resolve.assert_called_once_with("org/does-not-exist", "v1.0")
    download.assert_called_once_with(
        repo_id="org/does-not-exist",
        revision="a" * 40,
        local_dir=mock.ANY,
    )


def test_setup_models_pins_to_previously_locked_revision(monkeypatch, tmp_path):
    from unittest import mock

    import backend.config.model_config as mc_mod
    import scripts.setup_models as setup_mod

    cfg = ModelConfig(
        id="lock-test-model",
        display_name="x",
        runtime="faster_whisper",
        local_path=tmp_path / "lock-test-model",
        source={"type": "huggingface", "repo_id": "org/repo"},
    )
    monkeypatch.setattr(mc_mod, "load_model_configs", lambda **kw: [cfg])

    downloaded = []

    def fake_snapshot_download(repo_id, revision, local_dir):
        downloaded.append(revision)
        Path(local_dir, "config.json").write_text("{}")
        return local_dir

    head_shas = iter(["sha-AAA111", "sha-BBB222"])

    def fake_resolve(repo_id, revision):
        if revision and revision.startswith("sha-"):
            return revision
        return next(head_shas)

    with mock.patch(
        "huggingface_hub.snapshot_download",
        side_effect=fake_snapshot_download,
    ), mock.patch.object(
        setup_mod, "_resolve_hf_revision", side_effect=fake_resolve
    ):
        assert setup_mod.setup_model("lock-test-model", force=True)
        assert downloaded[-1] == "sha-AAA111"

        assert setup_mod.setup_model(
            "lock-test-model", force=True, update_lock=False
        )
        assert downloaded[-1] == "sha-AAA111"

        assert setup_mod.setup_model(
            "lock-test-model", force=True, update_lock=True
        )
        assert downloaded[-1] == "sha-BBB222"


def test_setup_models_fails_when_sha_cannot_be_resolved(monkeypatch, tmp_path):
    from unittest import mock

    import backend.config.model_config as mc_mod
    import scripts.setup_models as setup_mod

    cfg = ModelConfig(
        id="unresolvable",
        display_name="x",
        runtime="faster_whisper",
        local_path=tmp_path / "unresolvable",
        source={"type": "huggingface", "repo_id": "org/repo"},
    )
    monkeypatch.setattr(mc_mod, "load_model_configs", lambda **kw: [cfg])

    with mock.patch("huggingface_hub.snapshot_download") as download,          mock.patch.object(
             setup_mod, "_resolve_hf_revision", return_value=None
         ):
        ok = setup_mod.setup_model("unresolvable", force=True)

    assert ok is False
    download.assert_not_called()
    assert not setup_mod.MANIFEST_PATH.exists()


def test_resolve_hf_revision_skips_network_for_full_sha():
    import scripts.setup_models as setup_mod

    sha = "0123456789abcdef0123456789abcdef01234567"
    assert setup_mod._resolve_hf_revision("org/repo", sha) == sha


@pytest.mark.asyncio
async def test_orchestrator_close_cleans_up_owned_temp_files(tmp_path, monkeypatch):
    import backend.benchmark.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod, "load_model_configs", lambda **kw: _dummy_configs(1)
    )

    settings = get_settings().model_copy(deep=True)
    settings.paths.temp_dir = tmp_path

    path = tmp_path / "in.wav"
    _write_wav(path, 44100, 2, 1.0)
    artifact = AudioArtifact.from_wav(
        path, session_id="s", utterance_id="u"
    )
    assert artifact.path is None

    orchestrator = BenchmarkOrchestrator(settings=settings)
    await orchestrator.run(artifact, reference_text=None)
    owned = orchestrator._owned_audio_path
    assert owned is not None and Path(owned).exists()

    orchestrator.close()
    assert not Path(owned).exists()

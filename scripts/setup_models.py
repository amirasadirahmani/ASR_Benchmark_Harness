#!/usr/bin/env python3
"""
اسکریپت یک‌بارهٔ آنلاین برای آماده‌سازی مدل‌های محلی.

استفاده:
    python scripts/setup_models.py --all
    python scripts/setup_models.py --model whisper-base-fa-ct2
    python scripts/setup_models.py --wakeword vosk
    python scripts/setup_models.py --vad silero
    python scripts/setup_models.py --list
    python scripts/setup_models.py --all --update-lock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "model_manifest.json"


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"models": {}, "wakeword_vosk": None, "vad_silero": None}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _resolve_hf_revision(repo_id: str, revision: Optional[str]) -> Optional[str]:
    if revision and _FULL_SHA_RE.match(revision):
        return revision
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, revision=revision)
        return info.sha or None
    except Exception as exc:
        _print_err(f"resolve کردن SHA برای «{repo_id}» ناموفق بود: {exc}")
        return None


SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
VOSK_FA_URL = "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip"


def _print_step(msg: str) -> None:
    print(f"\n\033[1;36m▶ {msg}\033[0m")


def _print_ok(msg: str) -> None:
    print(f"  \033[1;32m✓\033[0m {msg}")


def _print_err(msg: str) -> None:
    print(f"  \033[1;31m✗ {msg}\033[0m", file=sys.stderr)


def _download(url: str, dest: Path, *, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _print_step(f"دانلود {label} از:\n  {url}")

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 // total_size
        print(
            f"\r  {pct:3d}%  ({done // 1024} / {total_size // 1024} KB)",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print()
        _print_ok(f"{label} دانلود شد → {dest}")
    except Exception as exc:
        print()
        raise RuntimeError(
            f"دانلود {label} ناموفق بود ({exc}).\n"
            f"    آدرس را در مرورگر باز کنید و صحتش را بررسی کنید: {url}"
        ) from exc


def _record_model_manifest(
    model_id: str,
    repo_id: str,
    requested_revision: Optional[str],
    resolved_revision: str,
    convert: str,
    quantization: Optional[str],
) -> None:
    manifest = _load_manifest()
    manifest.setdefault("models", {})[model_id] = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "convert": convert,
        "quantization": quantization,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    _print_ok(f"نسخهٔ HuggingFace ثبت شد: {repo_id} @ {resolved_revision[:12]}")


def setup_model(
    model_id: str, *, force: bool = False, update_lock: bool = False
) -> bool:
    from backend.config.model_config import load_model_configs

    configs = {config.id: config for config in load_model_configs(enabled_only=False)}
    cfg = configs.get(model_id)
    if cfg is None:
        _print_err(f"مدل «{model_id}» در models.yaml تعریف نشده است.")
        return False

    if cfg.runtime == "dummy":
        _print_ok(f"«{model_id}» مدل Dummy است؛ نیازی به دانلود ندارد.")
        return True

    local_path = (
        PROJECT_ROOT / cfg.local_path
        if not Path(cfg.local_path).is_absolute()
        else Path(cfg.local_path)
    )
    if local_path.exists() and any(local_path.iterdir()) and not force:
        _print_ok(
            f"«{model_id}» از قبل در {local_path} موجود است "
            "(برای اجبار به دانلود دوباره: --force)"
        )
        return True

    source = cfg.source
    if source.type != "huggingface":
        _print_err(f"«{model_id}» نوع منبع پشتیبانی‌نشده: {source.type}")
        return False

    repo_id = source.repo_id
    if not repo_id:
        _print_err(f"«{model_id}» فاقد source.repo_id در models.yaml است.")
        return False

    effective_revision = source.revision
    if effective_revision is None and not update_lock:
        locked = _load_manifest().get("models", {}).get(model_id)
        locked_revision = locked.get("resolved_revision") if locked else None
        if locked_revision:
            effective_revision = locked_revision
            _print_step(
                f"استفاده از نسخهٔ قفل‌شدهٔ قبلی: {locked_revision[:12]} "
                "(برای دریافت جدیدترین HEAD: --update-lock)"
            )

    # ابتدا SHA immutable resolve می‌شود؛ بدون SHA معتبر Setup ادامه نمی‌یابد.
    resolved_revision = _resolve_hf_revision(repo_id, effective_revision)
    if not resolved_revision:
        _print_err(
            f"«{model_id}»: SHA کامیت immutable برای «{repo_id}» "
            f"(revision={effective_revision or 'HEAD'}) resolve نشد؛ "
            "بدون آن Lock تکرارپذیر ممکن نیست، پس Setup متوقف شد."
        )
        return False

    _print_step(f"آماده‌سازی «{model_id}» ← {repo_id}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _print_err("کتابخانهٔ huggingface_hub نصب نیست.")
        return False

    with tempfile.TemporaryDirectory(prefix="asrb_dl_") as tmp:
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                revision=resolved_revision,
                local_dir=tmp,
            )
        except Exception as exc:
            _print_err(
                f"دانلود «{repo_id}» از HuggingFace Hub ناموفق بود: {exc}"
            )
            return False

        _print_ok(f"snapshot دانلود شد ← {snapshot_dir}")
        convert = source.convert
        local_path.mkdir(parents=True, exist_ok=True)

        if convert == "none":
            for item in Path(snapshot_dir).iterdir():
                target = local_path / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            _print_ok(f"مدل خام (transformers) در {local_path} قرار گرفت.")
            _record_model_manifest(
                model_id, repo_id, effective_revision, resolved_revision, convert, None
            )
            return True

        if convert == "ct2":
            try:
                from ctranslate2.converters import TransformersConverter
            except ImportError:
                _print_err("کتابخانهٔ ctranslate2 نصب نیست.")
                return False

            quantization = source.quantization or "int8"
            _print_step(f"تبدیل به CTranslate2 (quantization={quantization}) ...")
            try:
                converter = TransformersConverter(snapshot_dir)
                if local_path.exists():
                    shutil.rmtree(local_path)
                converter.convert(
                    str(local_path), quantization=quantization, force=True
                )
            except Exception as exc:
                _print_err(f"تبدیل CTranslate2 ناموفق بود: {exc}")
                return False

            _print_ok(f"مدل CTranslate2 در {local_path} ساخته شد.")
            _record_model_manifest(
                model_id,
                repo_id,
                effective_revision,
                resolved_revision,
                convert,
                quantization,
            )
            return True

        _print_err(f"نوع تبدیل ناشناخته: {convert}")
        return False


def setup_wakeword_vosk(
    *,
    url: Optional[str] = None,
    force: bool = False,
    update_lock: bool = False,
) -> bool:
    from backend.config.settings import get_settings

    settings = get_settings()
    dest = Path(settings.wake_word.vosk_model_path)
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest

    if dest.exists() and any(dest.iterdir()) and not force:
        _print_ok(f"مدل Vosk از قبل در {dest} موجود است.")
        return True

    zip_url = url or VOSK_FA_URL
    with tempfile.TemporaryDirectory(prefix="asrb_vosk_") as tmp:
        zip_path = Path(tmp) / "vosk-fa.zip"
        try:
            _download(zip_url, zip_path, label="مدل Vosk فارسی")
        except RuntimeError as exc:
            _print_err(str(exc))
            return False

        zip_sha256 = _sha256_file(zip_path)
        locked = _load_manifest().get("wakeword_vosk")
        if (
            locked
            and locked.get("sha256")
            and locked["sha256"] != zip_sha256
            and not update_lock
        ):
            _print_err(
                "هش فایل دانلودشده با نسخهٔ قفل‌شدهٔ قبلی یکی نیست. "
                "اگر عمداً نسخهٔ جدید را می‌خواهید از --update-lock استفاده کنید."
            )
            return False

        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            top = names[0].split("/")[0] if names else None
            archive.extractall(tmp)

        extracted = Path(tmp) / top if top else None
        if not extracted or not extracted.exists():
            _print_err("ساختار فایل استخراج‌شده غیرمنتظره است.")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(extracted), str(dest))
        _print_ok(f"مدل Vosk در {dest} نصب شد.")

        manifest = _load_manifest()
        manifest["wakeword_vosk"] = {
            "url": zip_url,
            "sha256": zip_sha256,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest)
        return True


def setup_vad_silero(
    *,
    url: Optional[str] = None,
    force: bool = False,
    update_lock: bool = False,
) -> bool:
    from backend.config.settings import get_settings

    settings = get_settings()
    dest = Path(settings.vad.silero_model_path)
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest

    if dest.exists() and not force:
        _print_ok(f"مدل Silero VAD از قبل در {dest} موجود است.")
        return True

    effective_url = url or SILERO_VAD_URL
    with tempfile.TemporaryDirectory(prefix="asrb_silero_") as tmp:
        tmp_dest = Path(tmp) / "silero_vad.onnx"
        try:
            _download(effective_url, tmp_dest, label="مدل Silero VAD (ONNX)")
        except RuntimeError as exc:
            _print_err(str(exc))
            return False

        file_sha256 = _sha256_file(tmp_dest)
        locked = _load_manifest().get("vad_silero")
        if (
            locked
            and locked.get("sha256")
            and locked["sha256"] != file_sha256
            and not update_lock
        ):
            _print_err(
                "هش فایل دانلودشده با نسخهٔ قفل‌شدهٔ قبلی یکی نیست. "
                "اگر عمداً نسخهٔ جدید را می‌خواهید از --update-lock استفاده کنید."
            )
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_dest, dest)

    manifest = _load_manifest()
    manifest["vad_silero"] = {
        "url": effective_url,
        "sha256": file_sha256,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    return True


def list_status() -> None:
    from backend.config.model_config import load_model_configs
    from backend.config.settings import get_settings

    settings = get_settings()
    _print_step("وضعیت مدل‌های ASR:")
    for cfg in load_model_configs(enabled_only=False):
        local_path = (
            PROJECT_ROOT / cfg.local_path
            if not Path(cfg.local_path).is_absolute()
            else Path(cfg.local_path)
        )
        available = cfg.runtime == "dummy" or (
            local_path.exists() and any(local_path.iterdir())
        )
        mark = "✓ موجود" if available else "✗ نیاز به دانلود"
        flag = "" if cfg.enabled else "  (غیرفعال)"
        print(f"  [{mark}] {cfg.id}{flag}  → {local_path}")

    vosk_path = Path(settings.wake_word.vosk_model_path)
    if not vosk_path.is_absolute():
        vosk_path = PROJECT_ROOT / vosk_path
    vosk_ok = vosk_path.exists() and any(vosk_path.iterdir())
    print(f"\n  wakeword/vosk: {'✓ موجود' if vosk_ok else '✗ نیاز به دانلود'} → {vosk_path}")

    vad_path = Path(settings.vad.silero_model_path)
    if not vad_path.is_absolute():
        vad_path = PROJECT_ROOT / vad_path
    print(f"  vad/silero:    {'✓ موجود' if vad_path.exists() else '✗ نیاز به دانلود'} → {vad_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", metavar="MODEL_ID")
    parser.add_argument("--wakeword", choices=["vosk"])
    parser.add_argument("--vosk-url", default=None)
    parser.add_argument("--vad", choices=["silero"])
    parser.add_argument("--vad-url", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_status()
        return 0

    if not any([args.model, args.wakeword, args.vad, args.all]):
        parser.print_help()
        return 1

    ok = True
    if args.all:
        from backend.config.model_config import load_model_configs
        for cfg in load_model_configs(enabled_only=True):
            ok &= setup_model(
                cfg.id, force=args.force, update_lock=args.update_lock
            )
        ok &= setup_wakeword_vosk(
            force=args.force, update_lock=args.update_lock
        )
        ok &= setup_vad_silero(
            force=args.force, update_lock=args.update_lock
        )
    else:
        for model_id in args.model or []:
            ok &= setup_model(
                model_id, force=args.force, update_lock=args.update_lock
            )
        if args.wakeword == "vosk":
            ok &= setup_wakeword_vosk(
                url=args.vosk_url,
                force=args.force,
                update_lock=args.update_lock,
            )
        if args.vad == "silero":
            ok &= setup_vad_silero(
                url=args.vad_url,
                force=args.force,
                update_lock=args.update_lock,
            )

    if ok:
        _print_ok(
            "همهٔ مراحل با موفقیت انجام شد. اکنون می‌توانید کاملاً آفلاین اجرا کنید."
        )
        print("    python scripts/offline_self_test.py")
        return 0

    _print_err("برخی مراحل ناموفق بودند؛ پیام‌های بالا را بررسی کنید.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

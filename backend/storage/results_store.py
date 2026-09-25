"""
ذخیره‌سازی نتایج Benchmark.

طراحی:
    * هر benchmark یک فایل JSON کامل (شامل segments، raw، همه چیز) —
      برای بازبینی و اشکال‌زدایی.
    * یک CSV تجمیعی append-only که هر سطرش یک (benchmark, model) است —
      برای تحلیل آماری در Excel/pandas بدون پارس JSON.

⚠️ چرا append-only و نه بازنویسی کامل؟
    اگر سرور در میانهٔ نوشتن CSV کرش کند، append-only حداکثر یک سطر را
    از دست می‌دهد؛ بازنویسی کامل می‌تواند کل تاریخچه را خراب کند.

Schema CSV پایدار است (بند ۲۲): ستون‌های جدید فقط در انتها اضافه می‌شوند،
هیچ ستونی حذف یا جابه‌جا نمی‌شود، تا نمودارها/اسکریپت‌های قدیمی نشکنند.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ترتیب ستون‌ها ثابت است — فقط در انتها اضافه کن، هرگز حذف/جابه‌جا نکن.
CSV_COLUMNS = [
    "benchmark_id", "session_id", "utterance_id", "created_at",
    "audio_sha256", "audio_duration",
    "model_id", "display_name", "runtime", "device", "compute_type",
    "success", "error_type", "error",
    "reference_text", "hypothesis_text", "hypothesis_normalized",
    "wer", "cer", "accuracy", "exact_match",
    "substitutions", "deletions", "insertions",
    "load_time", "warmup_time", "inference_time", "total_time", "rtf",
    "peak_ram_mb", "model_ram_mb", "cpu_percent_avg",
    "runs", "rank", "worker_pid",
]


@dataclass
class StoredPaths:
    json_path: Path
    csv_path: Path


class ResultsStore:
    """
    نویسندهٔ نتایج Benchmark به دیسک.

    Example:
        store = ResultsStore(results_dir)
        paths = store.save(report)
    """

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.json_dir = self.results_dir / "runs"
        self.csv_path = self.results_dir / "benchmark_history.csv"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_csv_header()

    # ------------------------------------------------------------------
    def _ensure_csv_header(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
            return
        # اعتبارسنجی سازگاری schema با فایل موجود
        try:
            with self.csv_path.open("r", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
            missing = [c for c in CSV_COLUMNS if c not in header]
            if missing:
                logger.warning(
                    "CSV موجود ستون‌های جدید %s را ندارد — مقادیرشان خالی "
                    "نوشته می‌شود. برای schema کامل، فایل را بایگانی و از نو بساز.",
                    missing,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("خواندن هدر CSV ناموفق: %s", exc)

    # ------------------------------------------------------------------
    def save(self, report: Any) -> StoredPaths:
        """
        ذخیرهٔ کامل یک گزارش Benchmark.

        `report` یک BenchmarkReport (dataclass) یا dict معادل آن است.
        """
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)

        with self._lock:
            json_path = self._write_json(data)
            self._append_csv(data)

        return StoredPaths(json_path=json_path, csv_path=self.csv_path)

    # ------------------------------------------------------------------
    def _write_json(self, data: Dict[str, Any]) -> Path:
        bid = data.get("benchmark_id", "unknown")
        ts = _safe_ts(data.get("created_at"))
        path = self.json_dir / f"{ts}_{bid}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            tmp.replace(path)   # نوشتن اتمی — از فایل نیمه‌نوشته جلوگیری می‌کند
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def _append_csv(self, data: Dict[str, Any]) -> None:
        rows = self._to_rows(data)
        if not rows:
            return
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writerows(rows)

    # ------------------------------------------------------------------
    def _to_rows(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تبدیل یک BenchmarkReport به سطرهای CSV (یکی برای هر مدل)."""
        rows: List[Dict[str, Any]] = []
        audio = data.get("audio") or {}

        for r in data.get("results", []):
            metrics = r.get("metrics") or {}
            row = {
                "benchmark_id": data.get("benchmark_id"),
                "session_id": data.get("session_id"),
                "utterance_id": data.get("utterance_id"),
                "created_at": data.get("created_at"),
                "audio_sha256": audio.get("sha256"),
                "audio_duration": audio.get("duration"),
                "model_id": r.get("model_id"),
                "display_name": r.get("display_name"),
                "runtime": r.get("runtime"),
                "device": r.get("device"),
                "compute_type": r.get("compute_type"),
                "success": r.get("success"),
                "error_type": r.get("error_type"),
                "error": _flatten(r.get("error")),
                "reference_text": data.get("reference_text"),
                "hypothesis_text": r.get("text"),
                "hypothesis_normalized": r.get("text_normalized"),
                "wer": metrics.get("wer"),
                "cer": metrics.get("cer"),
                "accuracy": metrics.get("accuracy"),
                "exact_match": metrics.get("exact_match"),
                "substitutions": metrics.get("substitutions"),
                "deletions": metrics.get("deletions"),
                "insertions": metrics.get("insertions"),
                "load_time": r.get("load_time"),
                "warmup_time": r.get("warmup_time"),
                "inference_time": r.get("inference_time"),
                "total_time": r.get("total_time"),
                "rtf": r.get("rtf"),
                "peak_ram_mb": r.get("peak_ram_mb"),
                "model_ram_mb": r.get("model_ram_mb"),
                "cpu_percent_avg": r.get("cpu_percent_avg"),
                "runs": r.get("runs"),
                "rank": r.get("rank"),
                "worker_pid": r.get("worker_pid"),
            }
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """فهرست آخرین benchmarkها برای نمایش در UI (تاریخچه)."""
        files = sorted(self.json_dir.glob("*.json"), reverse=True)[:limit]
        out = []
        for f in files:
            try:
                with f.open(encoding="utf-8") as fh:
                    d = json.load(fh)
                out.append({
                    "benchmark_id": d.get("benchmark_id"),
                    "created_at": d.get("created_at"),
                    "models_succeeded": d.get("models_succeeded"),
                    "models_failed": d.get("models_failed"),
                    "best_model": d.get("best_model"),
                    "has_reference": d.get("has_reference"),
                    "reference_text": d.get("reference_text"),
                    "file": f.name,
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("خواندن %s ناموفق: %s", f, exc)
        return out

    def load(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """بارگذاری کامل یک گزارش با شناسه (برای نمایش جزئیات)."""
        for f in self.json_dir.glob(f"*_{benchmark_id}.json"):
            try:
                with f.open(encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.warning("خواندن %s ناموفق: %s", f, exc)
        return None


# ---------------------------------------------------------------------------
def _safe_ts(iso_str: Any) -> str:
    try:
        dt = datetime.fromisoformat(str(iso_str))
    except Exception:  # noqa: BLE001
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


def _flatten(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    return s.replace("\n", " ").replace("\r", " ")[:500]
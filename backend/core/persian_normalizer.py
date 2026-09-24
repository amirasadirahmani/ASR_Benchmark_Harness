"""
نرمال‌ساز متن فارسی — نسخه‌دار و قطعی (Deterministic).

اصل حاکم (بند ۱۴ پروپوزال):
    «متن مرجع» و «خروجی همهٔ مدل‌ها» باید *دقیقاً* از یک مسیر نرمال‌سازی
    یکسان عبور کنند. هر تغییری در این فایل باید NORMALIZER_VERSION را
    افزایش دهد تا نتایج قدیمی و جدید قابل تفکیک باشند.

مراحل (به ترتیب):
    1.  حذف کاراکترهای کنترلی و Unicode نامرئی
    2.  یکسان‌سازی حروف عربی → فارسی (ي→ی، ك→ک، ة→ه، …)
    3.  حذف اعراب و تشدید (Harakat)
    4.  حذف کشیده (ـ / Tatweel)
    5.  یکسان‌سازی همزه روی الف (أ/إ/ٱ → ا)
    6.  سیاست نیم‌فاصله (ZWNJ): keep | space | remove
    7.  یکسان‌سازی ارقام (فارسی/عربی → ASCII)
    8.  حذف علائم نگارشی فارسی و لاتین
    9.  یکسان‌سازی انواع فاصله‌ها → فاصله عادی
    10. فشرده‌سازی فاصله‌های متوالی + trim
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.config.settings import NormalizerSettings, get_settings

# ---------------------------------------------------------------------------
# ⚠️ با هر تغییر در منطق نرمال‌سازی، این نسخه را افزایش بده.
NORMALIZER_VERSION = "1.0.0"
# ---------------------------------------------------------------------------


# --- جدول یکسان‌سازی حروف ---------------------------------------------------
_CHAR_MAP: Dict[str, str] = {
    # عربی → فارسی
    "\u064A": "\u06CC",   # ي → ی
    "\u0649": "\u06CC",   # ى → ی
    "\u06D2": "\u06CC",   # ے → ی
    "\u0643": "\u06A9",   # ك → ک
    "\u06AA": "\u06A9",   # ڪ → ک
    "\u0629": "\u0647",   # ة → ه
    "\u06C0": "\u0647",   # ۀ → ه
    "\u06C1": "\u0647",   # ہ → ه
    "\u0624": "\u0648",   # ؤ → و
    "\u0626": "\u06CC",   # ئ → ی
    # فاصله‌های غیرعادی → فاصله معمولی
    "\u00A0": " ",        # NBSP
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
    "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ",
    "\u2008": " ", "\u2009": " ", "\u200A": " ",
    "\u202F": " ", "\u205F": " ", "\u3000": " ",
    "\t": " ", "\r": " ", "\n": " ",
}

# همزه روی الف (اختیاری، طبق تنظیمات)
_ALEF_HAMZA_MAP: Dict[str, str] = {
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
    "\u0671": "\u0627",   # ٱ → ا
}
_ALEF_MADDA_MAP: Dict[str, str] = {
    "\u0622": "\u0627",   # آ → ا
}

# --- ارقام ------------------------------------------------------------------
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"

_DIGITS_TO_ASCII = str.maketrans(
    _PERSIAN_DIGITS + _ARABIC_DIGITS,
    _ASCII_DIGITS + _ASCII_DIGITS,
)
_DIGITS_TO_PERSIAN = str.maketrans(
    _ASCII_DIGITS + _ARABIC_DIGITS,
    _PERSIAN_DIGITS + _PERSIAN_DIGITS,
)

# --- اعراب، کشیده، کاراکترهای نامرئی ---------------------------------------
_DIACRITICS = re.compile(
    "["
    "\u064B-\u065F"   # فتحه، ضمه، کسره، تنوین، شدّه، سکون …
    "\u0670"          # الف خنجری
    "\u06D6-\u06ED"   # علائم قرآنی
    "\u0640"          # ـ کشیده (Tatweel)
    "]"
)

_INVISIBLE = re.compile(
    "["
    "\u200B"          # ZWSP
    "\u200E\u200F"    # LRM / RLM
    "\u202A-\u202E"   # embedding/override
    "\u2066-\u2069"   # isolates
    "\uFEFF"          # BOM
    "\u00AD"          # soft hyphen
    "]"
)

_ZWNJ = "\u200C"

# --- علائم نگارشی -----------------------------------------------------------
_PUNCTUATION = re.compile(
    "["
    r"\.\,\;\:\!\?\'\"\`\^\~\*\_\-\–\—\(\)\[\]\{\}\<\>\/\\\|\+\=\&\%\$\#\@"
    "\u060C"          # ،
    "\u061B"          # ؛
    "\u061F"          # ؟
    "\u066A-\u066D"   # ٪ ٫ ٬ ٭
    "\u06D4"          # ۔
    "\u00AB\u00BB"    # « »
    "\u2010-\u2027"   # انواع خط تیره و نقل‌قول
    "\u2030-\u205E"
    "]"
)

_MULTI_SPACE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
@dataclass
class NormalizationResult:
    """نتیجهٔ نرمال‌سازی به همراه ابرداده برای شفافیت و ممیزی."""
    original: str
    normalized: str
    tokens: List[str] = field(default_factory=list)
    char_count: int = 0
    token_count: int = 0
    version: str = NORMALIZER_VERSION
    profile: str = "default"

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "normalizer_version": self.version,
            "normalizer_profile": self.profile,
        }


# ---------------------------------------------------------------------------
class PersianNormalizer:
    """
    نرمال‌ساز متن فارسی، thread-safe و بدون وضعیت داخلی متغیر.

    Example:
        >>> n = PersianNormalizer()
        >>> n.normalize("سلام! حالِ شما چطور اسـت؟ ۱۲۳")
        'سلام حال شما چطور است 123'
    """

    def __init__(self, settings: Optional[NormalizerSettings] = None) -> None:
        self.settings = settings or get_settings().normalizer
        self._apply_profile()
        self._char_table = self._build_char_table()

    # ------------------------------------------------------------ setup
    def _apply_profile(self) -> None:
        """پروفایل‌های آماده، تنظیمات جزئی را override می‌کنند."""
        p = self.settings.profile
        if p == "light":
            # حداقل دخالت: فقط یکسان‌سازی حروف و فاصله
            self.settings = self.settings.model_copy(
                update={
                    "remove_punctuation": False,
                    "remove_diacritics": True,
                    "zwnj_policy": "keep",
                    "digits_to": "keep",
                    "unify_alef_hamza": False,
                }
            )
        elif p == "strict":
            # سخت‌گیرانه: حداکثر یکسان‌سازی برای WER عادلانه‌تر
            self.settings = self.settings.model_copy(
                update={
                    "remove_punctuation": True,
                    "remove_diacritics": True,
                    "zwnj_policy": "space",
                    "digits_to": "ascii",
                    "unify_alef_hamza": True,
                    "unify_alef_madda": True,
                }
            )

    def _build_char_table(self) -> dict:
        mapping = dict(_CHAR_MAP)
        if self.settings.unify_alef_hamza:
            mapping.update(_ALEF_HAMZA_MAP)
        if self.settings.unify_alef_madda:
            mapping.update(_ALEF_MADDA_MAP)
        return str.maketrans(mapping)

    # ------------------------------------------------------------ API
    def normalize(self, text: Optional[str]) -> str:
        """متن نرمال‌شده را برمی‌گرداند (رشتهٔ خالی برای ورودی تهی)."""
        if not text:
            return ""

        # 1) نرمال‌سازی یونیکد (NFC) + حذف کاراکترهای نامرئی/کنترلی
        s = unicodedata.normalize("NFC", text)
        s = _INVISIBLE.sub("", s)
        s = "".join(
            ch for ch in s
            if ch == _ZWNJ or not unicodedata.category(ch).startswith("C")
        )

        # 2) یکسان‌سازی حروف و فاصله‌ها
        s = s.translate(self._char_table)

        # 3) حذف اعراب و کشیده
        if self.settings.remove_diacritics:
            s = _DIACRITICS.sub("", s)

        # 4) سیاست نیم‌فاصله
        policy = self.settings.zwnj_policy
        if policy == "space":
            s = s.replace(_ZWNJ, " ")
        elif policy == "remove":
            s = s.replace(_ZWNJ, "")
        # "keep" → دست‌نخورده

        # 5) ارقام
        if self.settings.digits_to == "ascii":
            s = s.translate(_DIGITS_TO_ASCII)
        elif self.settings.digits_to == "persian":
            s = s.translate(_DIGITS_TO_PERSIAN)

        # 6) علائم نگارشی → فاصله (نه حذف کامل، تا کلمات نچسبند)
        if self.settings.remove_punctuation:
            s = _PUNCTUATION.sub(" ", s)

        # 7) فشرده‌سازی فاصله
        s = _MULTI_SPACE.sub(" ", s).strip()
        return s

    def tokenize(self, text: Optional[str]) -> List[str]:
        """توکن‌سازی روی متن *نرمال‌شده* — مبنای محاسبهٔ WER."""
        norm = self.normalize(text)
        return norm.split() if norm else []

    def characters(self, text: Optional[str], keep_spaces: bool = False) -> List[str]:
        """دنبالهٔ کاراکترها روی متن نرمال‌شده — مبنای محاسبهٔ CER."""
        norm = self.normalize(text)
        if keep_spaces:
            return list(norm)
        return [c for c in norm if not c.isspace()]

    def analyze(self, text: Optional[str]) -> NormalizationResult:
        """نرمال‌سازی به‌همراه ابردادهٔ کامل."""
        norm = self.normalize(text)
        tokens = norm.split() if norm else []
        return NormalizationResult(
            original=text or "",
            normalized=norm,
            tokens=tokens,
            char_count=len(norm.replace(" ", "")),
            token_count=len(tokens),
            version=NORMALIZER_VERSION,
            profile=self.settings.profile,
        )

    def equals(self, a: Optional[str], b: Optional[str]) -> bool:
        """مقایسهٔ دو متن پس از نرمال‌سازی — برای تشخیص Wake Word."""
        return self.normalize(a) == self.normalize(b)

    def contains(self, haystack: Optional[str], needle: Optional[str]) -> bool:
        """آیا `needle` (نرمال‌شده) به‌صورت کلمهٔ کامل در متن وجود دارد؟"""
        h = self.normalize(haystack)
        n = self.normalize(needle)
        if not h or not n:
            return False
        return f" {n} " in f" {h} "

    # ------------------------------------------------------------ meta
    @property
    def version(self) -> str:
        return NORMALIZER_VERSION

    def fingerprint(self) -> dict:
        """امضای پیکربندی — برای درج در CSV و تضمین تکرارپذیری."""
        return {
            "normalizer_version": NORMALIZER_VERSION,
            "profile": self.settings.profile,
            "zwnj_policy": self.settings.zwnj_policy,
            "remove_punctuation": self.settings.remove_punctuation,
            "remove_diacritics": self.settings.remove_diacritics,
            "unify_alef_hamza": self.settings.unify_alef_hamza,
            "unify_alef_madda": self.settings.unify_alef_madda,
            "digits_to": self.settings.digits_to,
        }


# --- singleton ---------------------------------------------------------------
_default_normalizer: Optional[PersianNormalizer] = None


def get_normalizer(settings: Optional[NormalizerSettings] = None) -> PersianNormalizer:
    """نمونهٔ مشترک نرمال‌ساز (برای تضمین یکسانی مطلق در کل سامانه)."""
    global _default_normalizer
    if settings is not None:
        return PersianNormalizer(settings)
    if _default_normalizer is None:
        _default_normalizer = PersianNormalizer()
    return _default_normalizer
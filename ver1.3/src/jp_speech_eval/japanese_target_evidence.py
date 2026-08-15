from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Sequence

from .text_frontend import build_text_info, kata_normalize, split_mora
from .verified_targets import lookup_verified_target


_FORBIDDEN_PHONE_OVERRIDE_TOKENS = frozenset(
    {
        "PAD",
        "UNK",
        "SOS",
        "EOS",
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<blank>",
        "pau",
        "sil",
    }
)


@dataclass(frozen=True)
class JapaneseTargetEvidence:
    """Free target-side Japanese linguistic evidence with provenance.

    The object describes the expected target pronunciation. It is not acoustic
    evidence that the learner actually realized that pronunciation.

    Phone-source policy is intentionally conditional:

    * ordinary automatic OpenJTalk targets keep surface text as the phone
      frontend so punctuation/context and model-label allophones match the
      frozen pyopenjtalk-plus contract;
    * verified/manual readings may drive G2P when a lexical reading must be
      corrected;
    * a reviewed ``phones_override`` is authoritative when even kana text can be
      morphologically re-analysed by the frontend. This is intended for audited
      research anchors, not casual product text.
    """

    surface_text: str
    frontend_input: str
    reading_kana: str
    reading_source: str
    phones: List[str]
    moras: List[str]
    fullcontext_labels: List[str]
    accent_source: str
    accent_phrases: List[Dict[str, Any]]
    verified_target_used: bool
    frontend_distribution: str
    frontend_version: Optional[str]
    frontend_ambiguous: bool
    marine_available: bool
    marine_used: bool
    marine_fullcontext_labels: Optional[List[str]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_pyopenjtalk():
    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError("pyopenjtalk-compatible frontend is required by the free Japanese frontend") from exc
    return pyopenjtalk


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def detect_pyopenjtalk_distribution() -> tuple[str, Optional[str], bool, List[str]]:
    """Identify which distribution currently owns the ``pyopenjtalk`` import."""
    plus_version = _installed_version("pyopenjtalk-plus")
    base_version = _installed_version("pyopenjtalk")
    warnings: List[str] = []
    if plus_version and base_version:
        warnings.append("both_pyopenjtalk_and_pyopenjtalk_plus_installed")
        return "ambiguous", plus_version, True, warnings
    if plus_version:
        return "pyopenjtalk-plus", plus_version, False, warnings
    if base_version:
        warnings.append("gop_backend_training_frontend_is_pyopenjtalk_plus_but_runtime_is_base_pyopenjtalk")
        return "pyopenjtalk", base_version, False, warnings
    warnings.append("pyopenjtalk_distribution_metadata_unavailable")
    return "unknown", None, False, warnings


def _g2p_phones(pyopenjtalk: Any, text: str) -> List[str]:
    output = pyopenjtalk.g2p(text, kana=False, join=True)
    if isinstance(output, str):
        return [phone for phone in output.split() if phone]
    return [str(phone) for phone in output if str(phone)]


def _validated_phone_override(phones: Sequence[str]) -> List[str]:
    values: List[str] = []
    for raw in phones:
        phone = str(raw).strip()
        if not phone:
            raise ValueError("phones_override contains an empty phone token")
        if any(character.isspace() for character in phone):
            raise ValueError(f"phones_override token contains whitespace: {phone!r}")
        if phone in _FORBIDDEN_PHONE_OVERRIDE_TOKENS:
            raise ValueError(f"phones_override contains nonsegmental/control token: {phone!r}")
        values.append(phone)
    if not values:
        raise ValueError("phones_override must contain at least one phone")
    return values


def build_japanese_target_evidence(
    text: str,
    *,
    reading_override: Optional[str] = None,
    phones_override: Optional[Sequence[str]] = None,
    use_marine_shadow: bool = False,
) -> JapaneseTargetEvidence:
    """Build Japanese reading/phone/accent metadata without paid services.

    Manual reading overrides deliberately suppress automatic accent
    interpretation. A reviewed pronunciation can be trusted for phones without
    pretending that the surface lexical analysis also provides a verified pitch
    pattern.

    For an ordinary automatic target, surface-context phone G2P remains the
    pinned baseline. Converting an already-resolved kana string back through the
    text frontend is *not* assumed to preserve an exact intended phone sequence:
    the frontend may still perform morphological/contextual analysis. Therefore
    audited research anchors can provide ``phones_override``. When supplied,
    that explicit phone sequence is authoritative and G2P-derived phone output
    is retained only implicitly as frontend metadata, never as the scoring
    target.
    """
    pyopenjtalk = _load_pyopenjtalk()
    verified = lookup_verified_target(text)
    warnings: List[str] = []
    frontend_distribution, frontend_version, frontend_ambiguous, frontend_warnings = (
        detect_pyopenjtalk_distribution()
    )
    warnings.extend(frontend_warnings)

    if reading_override:
        frontend_input = kata_normalize(str(reading_override))
        reading_kana = frontend_input
        reading_source = "manual_override"
        moras = split_mora(reading_kana)
        phones = _g2p_phones(pyopenjtalk, reading_kana)
        fullcontext_labels = list(pyopenjtalk.extract_fullcontext(frontend_input, run_marine=False))
        accent_source = "unavailable_for_manual_reading_without_verified_accent"
        accent_phrases: List[Dict[str, Any]] = []
        verified_target_used = False
        warnings.append("manual_reading_override_disables_automatic_accent_target")
        warnings.append("resolved_reading_drives_phone_sequence")
    else:
        frontend_input = text
        info = build_text_info(text)
        reading_kana = info.kana
        reading_source = "verified_target" if verified else "pyopenjtalk_g2p"
        moras = list(info.moras)
        if verified:
            phones = _g2p_phones(pyopenjtalk, reading_kana)
            warnings.append("verified_reading_drives_phone_sequence")
        else:
            phones = _g2p_phones(pyopenjtalk, text)
            warnings.append("automatic_surface_context_drives_phone_sequence")
        fullcontext_labels = list(pyopenjtalk.extract_fullcontext(text, run_marine=False))
        accent_source = str(info.pitch_target_source)
        accent_phrases = list(info.accent_phrases)
        verified_target_used = bool(verified)

    if phones_override is not None:
        phones = _validated_phone_override(phones_override)
        warnings.append("manual_phone_override_drives_phone_sequence")
        warnings.append("phone_override_is_authoritative_over_text_frontend_g2p")

    marine_available = False
    try:
        marine_available = find_spec("marine") is not None
    except Exception:
        marine_available = False
    marine_used = False
    marine_labels: Optional[List[str]] = None
    if use_marine_shadow:
        if not marine_available:
            warnings.append("marine_requested_but_not_installed")
        else:
            try:
                marine_labels = list(pyopenjtalk.extract_fullcontext(frontend_input, run_marine=True))
                marine_used = True
            except Exception as exc:
                warnings.append(f"marine_shadow_failed:{type(exc).__name__}")

    return JapaneseTargetEvidence(
        surface_text=text,
        frontend_input=frontend_input,
        reading_kana=reading_kana,
        reading_source=reading_source,
        phones=phones,
        moras=moras,
        fullcontext_labels=fullcontext_labels,
        accent_source=accent_source,
        accent_phrases=accent_phrases,
        verified_target_used=bool(verified_target_used),
        frontend_distribution=frontend_distribution,
        frontend_version=frontend_version,
        frontend_ambiguous=frontend_ambiguous,
        marine_available=marine_available,
        marine_used=marine_used,
        marine_fullcontext_labels=marine_labels,
        warnings=warnings,
    )

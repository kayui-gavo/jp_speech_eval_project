from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from shutil import which
from typing import Dict, List


@dataclass(frozen=True)
class FreeToolCapability:
    """Runtime-discoverable capability with explicit product/licensing policy.

    This registry is descriptive only.  Discovering a tool never enables it,
    downloads a model, or changes the user-facing ProductScore path.
    """

    tool: str
    role: str
    installed: bool
    license: str
    product_policy: str
    runtime: str
    requires_model_download: bool
    default_enabled: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def discover_free_assessment_stack() -> Dict[str, FreeToolCapability]:
    """Return the free/local stack without making network calls.

    `installed` means only that the Python package/CLI is discoverable.  It
    does not claim that optional model weights are cached or scientifically
    validated for ProductScore use.
    """

    return {
        "pyopenjtalk": FreeToolCapability(
            tool="pyopenjtalk",
            role="japanese_text_frontend_g2p_fullcontext",
            installed=_module_available("pyopenjtalk"),
            license="MIT wrapper + Modified BSD OpenJTalk",
            product_policy="allowed",
            runtime="local_cpu",
            requires_model_download=False,
            default_enabled=True,
            notes="Already a base dependency; target reading/accent metadata are evidence, not pronunciation ground truth.",
        ),
        "marine": FreeToolCapability(
            tool="marine",
            role="optional_japanese_accent_estimation_shadow",
            installed=_module_available("marine"),
            license="Apache-2.0",
            product_policy="shadow_only_until_validated",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="Optional accent estimator used only as a second target-side opinion; never treated as ground truth.",
        ),
        "faster_whisper": FreeToolCapability(
            tool="faster-whisper",
            role="japanese_asr_content_and_free_speech_transcript",
            installed=_module_available("faster_whisper"),
            license="MIT",
            product_policy="allowed",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=True,
            notes="Existing ASR backbone. Model weights must be cached/provisioned by deployment; discovery does not download them.",
        ),
        "beatrice_phone_ctc": FreeToolCapability(
            tool="prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 via transformers",
            role="japanese_phone_ctc_gop_shadow_primary",
            installed=_module_available("transformers"),
            license="Apache-2.0 model card; preserve upstream/training-data provenance",
            product_policy="shadow_only_until_japanese_l2_validated",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="94.4M Japanese phoneme CTC candidate. First GOP backend because it is substantially lighter than WavLM-large/dual-CTC alternatives. Native phone recognition is not L2 score validity.",
        ),
        "sakasegawa_dual_ctc": FreeToolCapability(
            tool="sakasegawa/japanese-wav2vec2-large-hiragana-ctc + hiragana-asr phone head",
            role="japanese_dual_ctc_gop_shadow_comparison",
            installed=_module_available("transformers"),
            license="Apache-2.0 repository/model metadata; preserve ReazonSpeech provenance",
            product_policy="research_shadow_comparison",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="315.6M dual CTC with a dedicated intermediate phoneme head and pyopenjtalk-style 42-phone vocabulary. Strong scientific comparison backend, too heavy to assume as product default.",
        ),
        "wavlm": FreeToolCapability(
            tool="microsoft/wavlm-large via transformers",
            role="ssl_pronunciation_similarity_shadow",
            installed=_module_available("transformers"),
            license="upstream model/code license must be preserved with deployment provenance",
            product_policy="shadow_only_until_human_calibrated",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="Existing v3 evidence backbone; no /100 mapping and no automatic download in normal tests.",
        ),
        "whisperx": FreeToolCapability(
            tool="WhisperX + Japanese wav2vec2 alignment model",
            role="optional_word_level_japanese_forced_alignment",
            installed=_module_available("whisperx"),
            license="BSD-2-Clause code; default Japanese alignment model Apache-2.0",
            product_policy="optional_shadow_benchmark",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="Useful candidate for word timestamps; not a phone-correctness score and not auto-installed.",
        ),
        "narabas": FreeToolCapability(
            tool="darashi/narabas",
            role="optional_japanese_phone_boundary_benchmark",
            installed=_module_available("narabas"),
            license="MIT repository; model/data provenance must still be recorded",
            product_policy="research_batch_candidate",
            runtime="local_model",
            requires_model_download=True,
            default_enabled=False,
            notes="Experimental Japanese Wav2Vec2 forced aligner using pyopenjtalk phone sequences. Useful boundary comparator, not pronunciation correctness by itself.",
        ),
        "mfa": FreeToolCapability(
            tool="Montreal Forced Aligner",
            role="optional_batch_phone_boundary_reference",
            installed=which("mfa") is not None,
            license="MIT package; model-specific license/attribution must be checked",
            product_policy="research_batch_candidate",
            runtime="local_cli",
            requires_model_download=True,
            default_enabled=False,
            notes="Promising Japanese forced-alignment benchmark backend, but too operationally heavy to become a silent runtime dependency.",
        ),
        "opensmile": FreeToolCapability(
            tool="openSMILE open-source edition",
            role="generic_paralinguistic_feature_extraction",
            installed=_module_available("opensmile"),
            license="audEERING Research License",
            product_policy="excluded_from_commercial_product_default",
            runtime="local_cpu",
            requires_model_download=False,
            default_enabled=False,
            notes="Open-source edition explicitly restricts commercial product use; do not add to the product runtime without a suitable license.",
        ),
        "parselmouth": FreeToolCapability(
            tool="praat-parselmouth",
            role="optional_praat_acoustic_cross_check",
            installed=_module_available("parselmouth"),
            license="GPL-3.0-or-later",
            product_policy="research_or_legal_review",
            runtime="local_cpu",
            requires_model_download=False,
            default_enabled=False,
            notes="Scientifically useful Praat-equivalent algorithms, but keep out of the default product dependency set until distribution/deployment obligations are reviewed.",
        ),
    }


def recommended_free_routes() -> Dict[str, List[str]]:
    """Architecture recommendation only; it does not execute or reweight scores."""

    return {
        "fixed_reading_default": [
            "pyopenjtalk_target_frontend",
            "faster_whisper_content_verification",
            "local_cached_dtw_timing",
            "local_f0",
        ],
        "fixed_reading_shadow": [
            "beatrice_phone_ctc_gop",
            "sakasegawa_dual_ctc_phone_gop_comparison",
            "wavlm_multi_reference_pronunciation",
            "whisperx_word_alignment_if_cached",
            "narabas_phone_alignment_batch_if_installed",
            "mfa_phone_alignment_batch_if_installed",
            "marine_accent_estimation_if_explicitly_enabled",
        ],
        "free_speaking_default": [
            "faster_whisper_japanese_asr",
            "local_pause_continuity",
            "local_recording_quality",
            "local_f0_phrase_intonation_shadow",
        ],
        "free_speaking_shadow": [
            "confirmed_transcript_to_pyopenjtalk_phones_then_phone_ctc_gop",
            "wavlm_if_reference_strategy_is_explicitly_available",
        ],
        "never_auto_enable": [
            "paid_cloud_pronunciation_apis",
            "opensmile_commercial_product_use",
            "automatic_optional_model_downloads",
        ],
    }


def free_only_policy_ok() -> bool:
    stack = discover_free_assessment_stack()
    forbidden = {
        "excluded_from_commercial_product_default",
    }
    return all(
        not capability.default_enabled or capability.product_policy not in forbidden
        for capability in stack.values()
    )

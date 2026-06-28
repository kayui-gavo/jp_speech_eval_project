from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version() -> str:
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def build_release(output_dir: Path) -> Path:
    version = _version()
    commit = _git_value("rev-parse", "HEAD")
    short_commit = commit[:7]
    output_dir.mkdir(parents=True, exist_ok=True)

    _run(sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output_dir))

    wheel = max(output_dir.glob(f"jp_speech_eval-{version}-*.whl"), key=lambda p: p.stat().st_mtime)
    sdist = max(output_dir.glob(f"jp_speech_eval-{version}.tar.gz"), key=lambda p: p.stat().st_mtime)

    bundle_name = f"jp_speech_eval_handoff_v{version}_{short_commit}"
    archive = output_dir / f"{bundle_name}.zip"

    with tempfile.TemporaryDirectory(prefix="jp_speech_eval_release_") as temp:
        bundle = Path(temp) / bundle_name
        _copy_file(ROOT / "docs" / "integration_handoff_zh.md", bundle / "README_INTEGRATION_ZH.md")
        _copy_file(ROOT / "docs" / "package_usage_zh.md", bundle / "docs" / "package_usage_zh.md")
        _copy_file(ROOT / "docs" / "python_package_api.md", bundle / "docs" / "python_package_api.md")
        _copy_file(ROOT / "docs" / "api_user_facing_contract.md", bundle / "docs" / "api_user_facing_contract.md")
        _copy_tree(ROOT / "examples", bundle / "examples")
        _copy_file(ROOT / "requirements.txt", bundle / "requirements-full.txt")
        _copy_file(wheel, bundle / "wheels" / wheel.name)
        _copy_file(sdist, bundle / "source" / sdist.name)

        for suffix in (".json", ".npz", ".ref.wav"):
            source = ROOT / "cache" / f"ramen_kudasai{suffix}"
            if source.exists():
                _copy_file(source, bundle / "sample_assets" / "cache" / source.name)
        sample_wav = ROOT / "data" / "ramen.wav"
        if sample_wav.exists():
            _copy_file(sample_wav, bundle / "sample_assets" / "data" / sample_wav.name)

        metadata = {
            "package": "jp-speech-eval",
            "version": version,
            "git_commit": commit,
            "python": ">=3.10,<3.13",
            "primary_modes": ["asr_confirmed_weak_reference", "reference"],
            "learner_ui_contract": "EvaluationResponse.user_facing",
        }
        (bundle / "VERSION.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        checksums = "".join(f"{_sha256(path)}  {path.relative_to(bundle)}\n" for path in files)
        (bundle / "SHA256SUMS").write_text(checksums, encoding="utf-8")

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    zip_file.write(path, path.relative_to(bundle.parent))

    print(archive)
    return archive


def smoke_release(archive: Path) -> None:
    """Install the wheel outside the source tree and run the sample API."""

    with tempfile.TemporaryDirectory(prefix="jp_speech_eval_wheel_smoke_") as temp:
        temp_root = Path(temp)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(temp_root / "bundle")
        bundle = next((temp_root / "bundle").iterdir())
        wheel = next((bundle / "wheels").glob("jp_speech_eval-*.whl"))
        site = temp_root / "site"
        _run(sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(wheel))

        smoke_code = """
import json
import sys
from pathlib import Path

site, bundle = map(Path, sys.argv[1:3])
sys.path.insert(0, str(site))

import jp_speech_eval
from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient
from jp_speech_eval.pitch_naturalness_v2 import default_config_path

assert jp_speech_eval.__version__ == "1.6.0"
assert str(jp_speech_eval.__file__).startswith(str(site))
assert default_config_path().exists()
assert "resources" in default_config_path().parts

client = SpeechEvaluationClient(
    SpeechEvalConfig(cache_path=str(bundle / "sample_assets" / "cache" / "ramen_kudasai"))
)
response = client.evaluate(
    EvaluationRequest(
        audio_path=str(bundle / "sample_assets" / "data" / "ramen.wav"),
        mode="reference",
        target_text="ラーメンをください",
    )
)
assert response.ok, response.error
dimensions = response.user_facing.get("dimension_scores", {})
assert set(dimensions) == {"pronunciation", "rhythm", "fluency", "pitch"}
assert all(isinstance(value, int) for value in dimensions.values())
print(json.dumps({"version": jp_speech_eval.__version__, "dimensions": dimensions}, ensure_ascii=False))
"""
        env = dict(os.environ)
        env["MPLCONFIGDIR"] = str(temp_root / "matplotlib")
        subprocess.run(
            [sys.executable, "-c", smoke_code, str(site), str(bundle)],
            cwd=temp_root,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wheel, sdist and the integration handoff bundle.")
    parser.add_argument("--output-dir", default="dist", help="Output directory relative to ver1.3.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip isolated wheel installation smoke test.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    archive = build_release(output)
    if not args.skip_smoke:
        smoke_release(archive)


if __name__ == "__main__":
    main()

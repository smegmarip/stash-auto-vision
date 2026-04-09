"""
Faces Service Bootstrap

Generic one-shot bootstrap entry point that runs before uvicorn starts.
Currently responsible for ensuring required ONNX model weights are present
in the model cache (downloading from HuggingFace when missing); designed
as a single CLI surface so future bootstrap concerns (cache warming,
dependency probing, etc.) can slot in alongside without a rewrite.

CLI:
    python -m app.bootstrap              # run all bootstrap steps
    python -m app.bootstrap --models     # run only the model-download step

The Dockerfile invokes this with no arguments before execing uvicorn,
so every container start goes through the full bootstrap. Individual
steps are idempotent and cheap when the cache is already populated —
a hot restart with all models present completes in well under a second.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("faces-bootstrap")

# ---------------------------------------------------------------------------
# Defaults — used when the corresponding env var is empty or missing
# ---------------------------------------------------------------------------
DEFAULT_HF_REPO = "smegmarip/face-recognition"
DEFAULT_HF_OCCLUSION_MODEL = "models/occlusion_classifier.onnx"
DEFAULT_HF_TOPIQ_MODEL = "models/topiq_nr.onnx"
DEFAULT_HF_CLIPIQA_MODEL = "models/clipiqa_plus.onnx"
DEFAULT_MODEL_CACHE_DIR = "/app/models"

# Manifest file written by the models step so the runtime (and the health
# endpoint) can report which model files are actually available without
# re-probing the filesystem on every request.
MANIFEST_FILENAME = "bootstrap_manifest.json"


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Declarative spec for a single model file to resolve during bootstrap."""

    name: str                       # short ID used in logs and the manifest
    local_filename: str             # filename inside the cache directory
    required: bool                  # required = blocks health endpoint when missing
    env_hf_file: str                # env var holding the HF-repo-relative path
    env_local_path: str             # env var holding an absolute local-override path
    default_hf_file: str            # hardcoded default HF-repo-relative path


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="occlusion",
        local_filename="occlusion_classifier.onnx",
        required=True,
        env_hf_file="FACES_HF_OCCLUSION_MODEL",
        env_local_path="FACES_LOCAL_OCCLUSION_PATH",
        default_hf_file=DEFAULT_HF_OCCLUSION_MODEL,
    ),
    ModelSpec(
        name="topiq",
        local_filename="topiq_nr.onnx",
        required=False,
        env_hf_file="FACES_HF_TOPIQ_MODEL",
        env_local_path="FACES_LOCAL_TOPIQ_PATH",
        default_hf_file=DEFAULT_HF_TOPIQ_MODEL,
    ),
    ModelSpec(
        name="clipiqa",
        local_filename="clipiqa_plus.onnx",
        required=False,
        env_hf_file="FACES_HF_CLIPIQA_MODEL",
        env_local_path="FACES_LOCAL_CLIPIQA_PATH",
        default_hf_file=DEFAULT_HF_CLIPIQA_MODEL,
    ),
]


@dataclass
class ModelResolution:
    """Result of resolving a single model spec."""

    name: str
    required: bool
    local_path: Optional[str] = None     # absolute path to the resolved file
    source: str = "unresolved"           # "local_override" | "hf_configured" | "hf_default" | "missing"
    error: Optional[str] = None          # populated when source == "missing"


@dataclass
class BootstrapManifest:
    """Manifest written to the cache dir after the models step runs."""

    cache_dir: str
    required_ready: bool = True          # False if any required model failed to resolve
    models: Dict[str, ModelResolution] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "cache_dir": self.cache_dir,
                "required_ready": self.required_ready,
                "models": {k: asdict(v) for k, v in self.models.items()},
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    """Read an env var, treating whitespace-only values as empty."""
    value = os.environ.get(name, default)
    return value.strip() if value else ""


def _resolve_hf_repo() -> str:
    """Resolve the HF repo ID, falling back to the hardcoded default."""
    return _env("FACES_HF_REPO") or DEFAULT_HF_REPO


def _hf_token() -> Optional[str]:
    """Optional HF token (for gated or private forks). Returns None when empty."""
    token = _env("FACES_HF_TOKEN")
    return token or None


def _try_hf_download(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    local_filename: str,
    token: Optional[str],
) -> Optional[str]:
    """Download a single file from HF Hub into the cache dir.

    Returns the absolute path on success, None on any failure. The download
    lands at ``cache_dir/local_filename`` regardless of how the file is named
    inside the HF repo, so the runtime always looks at the same path.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.error("huggingface_hub not installed: %s", exc)
        return None

    try:
        logger.info("HF download: %s / %s", repo_id, filename)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    except Exception as exc:
        logger.warning("HF download failed (%s / %s): %s", repo_id, filename, exc)
        return None

    # hf_hub_download returns a path inside its own cache (symlink-based).
    # Copy into our cache dir under the expected runtime filename so the
    # service finds it at a stable, predictable location.
    target_path = cache_dir / local_filename
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Use a real copy (not symlink) so the named volume is self-contained
        # and survives HF cache eviction on the host.
        import shutil
        shutil.copy2(downloaded_path, target_path)
    except Exception as exc:
        logger.error("Failed to copy downloaded file into cache dir: %s", exc)
        return None

    logger.info("HF download OK: %s → %s", filename, target_path)
    return str(target_path)


def _resolve_one_model(spec: ModelSpec, cache_dir: Path, token: Optional[str]) -> ModelResolution:
    """Apply the per-model resolution order: local override → HF → HF defaults."""
    result = ModelResolution(name=spec.name, required=spec.required)

    target_in_cache = cache_dir / spec.local_filename

    # --- 1. Local override -------------------------------------------------
    local_override = _env(spec.env_local_path)
    if local_override:
        candidate = Path(local_override)
        if candidate.is_file():
            logger.info("%s: using local override at %s", spec.name, candidate)
            result.local_path = str(candidate)
            result.source = "local_override"
            return result
        else:
            logger.warning(
                "%s: %s is set but file does not exist (%s); falling back to HF",
                spec.name, spec.env_local_path, candidate,
            )

    # --- 1b. Already cached ------------------------------------------------
    # If a prior bootstrap run already populated the volume, skip the download.
    if target_in_cache.is_file():
        logger.info("%s: already cached at %s", spec.name, target_in_cache)
        result.local_path = str(target_in_cache)
        result.source = "cache_hit"
        return result

    # --- 2. Explicit HF config ---------------------------------------------
    repo_id = _resolve_hf_repo()
    configured_file = _env(spec.env_hf_file)
    if configured_file:
        path = _try_hf_download(repo_id, configured_file, cache_dir, spec.local_filename, token)
        if path:
            result.local_path = path
            result.source = "hf_configured"
            return result
        logger.warning("%s: configured HF download failed, trying defaults", spec.name)

    # --- 3. HF defaults ----------------------------------------------------
    path = _try_hf_download(
        DEFAULT_HF_REPO, spec.default_hf_file, cache_dir, spec.local_filename, token,
    )
    if path:
        result.local_path = path
        result.source = "hf_default"
        return result

    # --- 4. All resolution paths failed ------------------------------------
    result.source = "missing"
    result.error = f"Could not resolve {spec.name} model from local override, configured HF, or HF defaults"
    if spec.required:
        logger.error("%s (REQUIRED): %s", spec.name, result.error)
    else:
        logger.warning("%s (optional): %s", spec.name, result.error)
    return result


# ---------------------------------------------------------------------------
# Bootstrap steps
# ---------------------------------------------------------------------------
def run_models_step() -> BootstrapManifest:
    """Ensure all declared model files are available in the cache dir.

    Writes a manifest file alongside the models so the runtime health
    endpoint can report availability without re-probing the filesystem.
    """
    cache_dir = Path(_env("FACES_MODEL_CACHE_DIR") or DEFAULT_MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model cache directory: %s", cache_dir)

    token = _hf_token()
    manifest = BootstrapManifest(cache_dir=str(cache_dir))

    for spec in MODEL_SPECS:
        resolution = _resolve_one_model(spec, cache_dir, token)
        manifest.models[spec.name] = resolution
        if spec.required and resolution.source == "missing":
            manifest.required_ready = False

    manifest_path = cache_dir / MANIFEST_FILENAME
    try:
        manifest_path.write_text(manifest.to_json())
        logger.info("Wrote bootstrap manifest: %s", manifest_path)
    except Exception as exc:
        logger.warning("Could not write bootstrap manifest (%s): %s", manifest_path, exc)

    if manifest.required_ready:
        logger.info("Bootstrap models step: all required models ready")
    else:
        logger.error(
            "Bootstrap models step: one or more REQUIRED models missing — "
            "service will start but health endpoint will report unhealthy"
        )

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    level_name = _env("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Bootstrap CLI entry point. Returns a process exit code.

    Exit codes:
        0  - all requested steps completed (optional-model failures are
             warnings, not errors)
        1  - a required model could not be resolved (the service will still
             start — the error is surfaced via /faces/health)
    """
    parser = argparse.ArgumentParser(
        prog="faces-bootstrap",
        description="Run faces-service bootstrap steps (models download, etc).",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Run only the model-download step (skip all other bootstrap steps).",
    )
    args = parser.parse_args(argv)

    _configure_logging()

    run_all = not args.models

    logger.info("faces-service bootstrap starting (mode=%s)", "all" if run_all else "models")

    manifest = run_models_step()

    if run_all:
        # Placeholder for future bootstrap steps.
        # e.g. cache warming, dependency probing, preflight checks.
        pass

    # Non-zero exit code if required models are missing, so operators can
    # detect bootstrap failure in CI. The Dockerfile CMD intentionally does
    # NOT propagate this exit code — the service still starts so operators
    # can hit /faces/health for a structured error message.
    return 0 if manifest.required_ready else 1


if __name__ == "__main__":
    sys.exit(main())

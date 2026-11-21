"""Central model registry (model_registry/*.yaml).

Each YAML file describes one model: task, repositories, pinned weights, licences,
hardware needs, formats, benchmarks and limitations. Business logic refers to models
only by registry id; adapters (later milestones) bind ids to implementations.

Licence fields are *declared* from documentation until ``--verify-registry`` checks the
public Hugging Face / GitHub APIs and records the result in
``model_registry/_verification.json``. Where sources disagree or are ambiguous the
platform asks the user for the authoritative document rather than guessing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import ConfigError
from ..core.network_policy import DEFAULT_FORBIDDEN_HOST_PATTERNS, assert_allowed

Permission = Literal["yes", "no", "restricted", "unknown"]
Usage = Literal["personal_learning", "non_commercial", "commercial"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WeightsSpec(_Strict):
    hf_repo: str | None = None
    url: str | None = None
    files: list[str] = Field(default_factory=list)
    pinned_revision: str | None = Field(
        default=None,
        description="Revision used for artifact fingerprints. Changing it invalidates dependent artifacts.",
    )
    sha256: dict[str, str] = Field(default_factory=dict, description="file → sha256 once downloaded and verified")
    gated: bool = False


class LicenceSpec(_Strict):
    code: str | None = None
    weights: str | None = None
    personal_use: Permission = "unknown"
    commercial_use: Permission = "unknown"
    attribution_required: bool | None = None
    notes: str = ""
    sources: list[str] = Field(default_factory=list)
    declared_from: Literal["documentation_unverified", "verified_api", "user_provided_document"] = (
        "documentation_unverified"
    )


class HardwareSpec(_Strict):
    runtimes: list[Literal["cuda", "mps", "cpu", "mlx"]] = Field(default_factory=list)
    min_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    notes: str = ""


class BenchmarkEntry(_Strict):
    date: date
    hardware: str
    dataset: str
    metrics: dict[str, float | str]
    notes: str = ""


class ModelRecord(_Strict):
    schema_version: Literal[1] = 1
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]*$")
    name: str
    tasks: list[str]
    status: Literal["legacy", "candidate", "selected", "excluded"] = "candidate"
    version: str | None = None
    repository: str | None = None
    weights: WeightsSpec = Field(default_factory=WeightsSpec)
    licence: LicenceSpec = Field(default_factory=LicenceSpec)
    supported_languages: list[str] | Literal["multilingual", "n/a"] = "n/a"
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    input_format: str = ""
    output_format: str = ""
    inference_parameters: dict[str, Any] = Field(default_factory=dict)
    benchmarks: list[BenchmarkEntry] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return f"{self.model_id}@{self.weights.pinned_revision or 'unpinned'}"


class PolicyDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def check_usage_policy(record: ModelRecord, usage: Usage) -> PolicyDecision:
    """Is this model usable under the project's licensing.usage?"""
    reasons: list[str] = []
    warnings: list[str] = []
    lic = record.licence
    if record.status == "excluded":
        reasons.append("model is marked excluded in the registry")
    if lic.personal_use == "no":
        reasons.append("licence forbids even personal use")
    if usage == "commercial":
        if lic.commercial_use == "no":
            reasons.append(f"weights licence ({lic.weights}) does not permit commercial use")
        elif lic.commercial_use in ("restricted", "unknown"):
            warnings.append(f"commercial use is '{lic.commercial_use}' — review before commercial use")
    if lic.personal_use == "unknown":
        warnings.append("personal-use permission not yet confirmed")
    if lic.declared_from == "documentation_unverified":
        warnings.append("licence not yet verified (run --verify-registry or provide the licence document)")
    if record.weights.gated:
        warnings.append("gated weights: accept the terms on the model page with your personal account")
    return PolicyDecision(allowed=not reasons, reasons=reasons, warnings=warnings)


class ModelRegistry:
    def __init__(self, records: dict[str, ModelRecord], directory: Path | None = None) -> None:
        self.records = records
        self.directory = directory

    @classmethod
    def load(cls, directory: Path) -> ModelRegistry:
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigError(f"model registry directory not found: {directory}")
        records: dict[str, ModelRecord] = {}
        for path in sorted(directory.glob("*.yaml")):
            try:
                record = ModelRecord.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            except (yaml.YAMLError, ValidationError) as exc:
                raise ConfigError(f"invalid model registry entry {path.name}: {exc}") from exc
            if path.stem != record.model_id:
                raise ConfigError(f"{path.name}: file name must match model_id '{record.model_id}'")
            records[record.model_id] = record
        return cls(records, directory)

    def get(self, model_id: str) -> ModelRecord:
        try:
            return self.records[model_id]
        except KeyError:
            raise ConfigError(
                f"model '{model_id}' is not in the registry ({self.directory}). Add model_registry/{model_id}.yaml."
            ) from None

    def __iter__(self):
        return iter(self.records.values())

    def verification_path(self) -> Path | None:
        return self.directory / "_verification.json" if self.directory else None

    def load_verification(self) -> dict[str, Any]:
        path = self.verification_path()
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}


# ---------------------------------------------------------------------------
# Verification against public APIs (hosts checked by the network policy)
# ---------------------------------------------------------------------------

_HF_API = "https://huggingface.co/api/models/{repo}"
_GH_API = "https://api.github.com/repos/{owner}/{repo}"


def _get_json(url: str, forbidden: list[str], timeout: float = 20.0) -> dict[str, Any]:
    assert_allowed(url, forbidden, what="verification URL")
    request = urllib.request.Request(url, headers={"User-Agent": "video-platform-registry-verifier/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return json.loads(response.read().decode("utf-8"))


def _github_slug(repository: str | None) -> tuple[str, str] | None:
    if not repository or "github.com/" not in repository:
        return None
    parts = repository.split("github.com/", 1)[1].strip("/").split("/")
    return (parts[0], parts[1].removesuffix(".git")) if len(parts) >= 2 else None


def verify_record(record: ModelRecord, forbidden: list[str] | None = None) -> dict[str, Any]:
    forbidden = forbidden or list(DEFAULT_FORBIDDEN_HOST_PATTERNS)
    result: dict[str, Any] = {"checked_at": datetime.now(tz=timezone.utc).isoformat(), "issues": []}

    if record.weights.hf_repo:
        url = _HF_API.format(repo=record.weights.hf_repo)
        try:
            data = _get_json(url, forbidden)
            card = data.get("cardData") or {}
            tags = [t for t in data.get("tags", []) if isinstance(t, str) and t.startswith("license:")]
            result["huggingface"] = {
                "url": f"https://huggingface.co/{record.weights.hf_repo}",
                "license": card.get("license") or (tags[0].split(":", 1)[1] if tags else None),
                "license_name": card.get("license_name"),
                "license_link": card.get("license_link"),
                "sha": data.get("sha"),
                "gated": data.get("gated"),
                "last_modified": data.get("lastModified"),
            }
            hf_license = result["huggingface"]["license"]
            if hf_license in (None, "other", "unknown"):
                result["issues"].append("HF licence is missing or 'other' — read the model card / LICENSE manually")
            if bool(data.get("gated")) != record.weights.gated:
                result["issues"].append(f"registry gated={record.weights.gated} but HF reports gated={data.get('gated')}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            result["huggingface"] = {"url": url, "error": str(exc)}
            result["issues"].append("Hugging Face API unreachable — verify manually or provide the model card")

    slug = _github_slug(record.repository)
    if slug:
        url = _GH_API.format(owner=slug[0], repo=slug[1])
        try:
            data = _get_json(url, forbidden)
            licence = data.get("license") or {}
            result["github"] = {
                "url": record.repository,
                "license_spdx": licence.get("spdx_id"),
                "license_name": licence.get("name"),
                "archived": data.get("archived"),
                "pushed_at": data.get("pushed_at"),
            }
            if licence.get("spdx_id") in (None, "NOASSERTION"):
                result["issues"].append("GitHub reports no standard licence — read the LICENSE file manually")
            if data.get("archived"):
                result["issues"].append("GitHub repository is archived (unmaintained)")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            result["github"] = {"url": url, "error": str(exc)}
            result["issues"].append("GitHub API unreachable — verify manually or provide the LICENSE")
    return result


def verify_registry(registry: ModelRegistry, forbidden: list[str] | None = None) -> dict[str, Any]:
    results = {record.model_id: verify_record(record, forbidden) for record in registry}
    path = registry.verification_path()
    if path:
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results

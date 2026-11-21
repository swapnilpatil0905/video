"""Unified command-line interface (``python run.py`` / ``./run.sh``).

Exit codes: 0 ok · 2 configuration error · 3 illegal state · 4 service unavailable ·
5 stage failure · 6 not implemented yet (later milestone).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config.loader import LoadedConfig, find_platform_root, load_project_config
from .config.schema import ProjectConfig
from .core.errors import ConfigError, IllegalStateError, NotImplementedYet, PlatformError
from .core.log import configure_logging, get_logger
from .core.network_policy import DEFAULT_FORBIDDEN_HOST_PATTERNS, preflight
from .approval.state import ApprovalStore
from .migration.legacy import LegacyImporter, build_manifest, compare_manifests
from .orchestration.runner import Runner
from .runtime.launcher import RuntimeRunner
from .models.registry import ModelRegistry, check_usage_policy, verify_registry
from .orchestration.graph import StageGraph, load_pipeline
from .orchestration.planner import Planner
from .storage.index import ArtifactIndex
from .storage.store import ArtifactStore, store_from_config

log = get_logger("video_platform")

MODES = ("analyze", "sample", "analyze-and-sample", "full", "worker")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="AI video remake / localization platform")
    p.add_argument("--version", action="version", version=f"video-platform {__version__}")
    p.add_argument("--project", type=Path, help="path to projects/<id>/config.yaml")
    p.add_argument("--mode", choices=MODES, help="pipeline mode to execute (or to --plan)")
    p.add_argument("--email", help="approval recipient (overrides notifications.approval_email)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set sample.duration_seconds=60 (repeatable)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--log-level", default="INFO")

    g = p.add_argument_group("inspection")
    g.add_argument("--validate-only", action="store_true", help="validate configuration, models and inputs")
    g.add_argument("--print-config", action="store_true", help="print the effective merged configuration")
    g.add_argument("--config-schema", action="store_true", help="print the configuration JSON schema")
    g.add_argument("--status", action="store_true", help="project, artifact and plan status")
    g.add_argument("--plan", action="store_true", help="show what --mode/--stage would execute (cache-aware)")
    g.add_argument("--estimate", action="store_true", help="resource and cost estimate")
    g.add_argument("--preflight", action="store_true", help="check package indexes / endpoints against network policy")

    g = p.add_argument_group("project setup and migration")
    g.add_argument("--init", type=Path, metavar="PROJECT_DIR", help="create a new project from the template")
    g.add_argument("--migrate-legacy", type=Path, nargs="?", const=Path("."), metavar="LEGACY_ROOT",
                   help="import the pre-platform analysis workspace as cached artifacts (default: platform root)")
    g.add_argument("--verify-manifest", action="store_true", help="verify legacy files are unchanged since migration")
    g.add_argument("--verify-registry", action="store_true",
                   help="check model licences against public Hugging Face / GitHub APIs")

    g = p.add_argument_group("execution")
    g.add_argument("--stage", help="run (or --plan) a single stage and its prerequisites")
    g.add_argument("--resume", action="store_true", help="resume interrupted work")
    g.add_argument("--regenerate-shot", metavar="SHOT_ID")
    g.add_argument("--regenerate-dialogue", metavar="SEGMENT_ID")

    g = p.add_argument_group("approval (email workflow; see docs/specs/email-approval-workflow.md)")
    g.add_argument("--approve-sample", action="store_true", help="approve the current sample (audited)")
    g.add_argument("--reject-sample", action="store_true", help="stop: do not start full production (audited)")
    g.add_argument("--profile", help="which trailer/profile to approve (e.g. high or balanced)")
    g.add_argument("--notify", choices=("sample", "failure"), help="email a notification (SMTP settings from .env)")
    g.add_argument("--message", help="text for --notify failure")
    g.add_argument("--log", help="log file whose tail is included in --notify failure")
    g.add_argument("--send-approval-email", action="store_true")
    g.add_argument("--approval-status", action="store_true")
    g.add_argument("--resend-approval", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    loaded: LoadedConfig
    graph: StageGraph
    registry: ModelRegistry
    index: ArtifactIndex
    store: ArtifactStore
    planner: Planner

    @property
    def config(self) -> ProjectConfig:
        return self.loaded.config


def build_context(args: argparse.Namespace) -> Context:
    if args.project is None:
        raise ConfigError("--project is required for this command")
    loaded = load_project_config(args.project, overrides=args.overrides, email=args.email)
    root = loaded.platform_root
    graph = load_pipeline(root / "configs" / "pipeline.yaml")
    registry = ModelRegistry.load(root / "model_registry")
    index = ArtifactIndex(loaded.state_dir / "artifacts.sqlite")
    store = store_from_config(loaded)
    planner = Planner(graph, loaded, registry, index)
    return Context(loaded, graph, registry, index, store, planner)


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str) if args.json else text)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def model_checks(ctx: Context) -> list[dict[str, Any]]:
    results = []
    for name in ctx.graph.order:
        spec = ctx.graph.get(name)
        for model_id in ctx.planner.model_ids(spec):
            record = ctx.registry.get(model_id)
            decision = check_usage_policy(record, ctx.config.licensing.usage)
            results.append({"stage": name, "model": model_id, "allowed": decision.allowed,
                            "reasons": decision.reasons, "warnings": decision.warnings})
    return results


def cmd_validate(args: argparse.Namespace, ctx: Context) -> int:
    cfg = ctx.config
    problems: list[str] = []
    checks = model_checks(ctx)
    problems += [f"model {c['model']} ({c['stage']}): {r}" for c in checks for r in c["reasons"]]
    source = ctx.loaded.source_video_path
    if not source.exists():
        problems.append(f"source video not found: {source} (run --migrate-legacy for project_001, or copy it there)")
    payload = {
        "project": cfg.project.id,
        "config": str(ctx.loaded.config_path),
        "valid": not problems,
        "source_video": str(source),
        "source_exists": source.exists(),
        "languages": f"{cfg.languages.source} → {cfg.languages.target}",
        "licensing_usage": cfg.licensing.usage,
        "models": checks,
        "problems": problems,
    }
    lines = [
        f"Project {cfg.project.id} ({cfg.project.project_type.value}): {cfg.languages.source} → {cfg.languages.target}",
        f"  config        {ctx.loaded.config_path}",
        f"  source video  {source} [{'found' if source.exists() else 'MISSING'}]",
        f"  usage policy  {cfg.licensing.usage}",
        "  models:",
    ]
    for c in checks:
        mark = "ok " if c["allowed"] else "BLOCKED"
        lines.append(f"    [{mark}] {c['stage']:<22} {c['model']}")
        lines += [f"           ! {w}" for w in c["warnings"]]
        lines += [f"           x {r}" for r in c["reasons"]]
    lines.append("Configuration is valid." if not problems else "Problems:\n  - " + "\n  - ".join(problems))
    _emit(args, payload, "\n".join(lines))
    return 0 if not problems else 2


def _plan_targets(args: argparse.Namespace, ctx: Context) -> list[str] | None:
    if args.stage:
        ctx.graph.get(args.stage)
        return [args.stage]
    if args.mode:
        if args.mode == "worker":
            raise NotImplementedYet("--mode worker (queue worker for the email/VPS setup)", "M5-email")
        return ctx.graph.mode_targets(args.mode)
    return None


def _plan_text(items: list[Any]) -> str:
    lines = [f"{'stage':<24} {'status':<8} {'milestone':<9} artifact_id / reason"]
    for item in items:
        lines.append(f"{item.stage:<24} {item.status:<8} {item.milestone:<9} {item.artifact_id}")
        if item.status != "cached":
            lines.append(f"{'':<43}↳ {item.reason}")
    cached = sum(i.status == "cached" for i in items)
    lines.append(f"\n{cached}/{len(items)} stages cached; {len(items) - cached} would run.")
    return "\n".join(lines)


def cmd_plan(args: argparse.Namespace, ctx: Context) -> int:
    items = ctx.planner.plan(_plan_targets(args, ctx))
    _emit(args, [i.as_dict() for i in items], _plan_text(items))
    return 0


SAMPLE_STAGE, FULL_STAGE = "sample_production", "full_production"


def _approvals(ctx: Context) -> ApprovalStore:
    return ApprovalStore(ctx.loaded.state_dir / "approval.sqlite")


def _actor() -> str:
    return f"cli:{os.environ.get('USER', 'unknown')}"


def _current_sample(ctx: Context) -> tuple[str, dict[str, str]] | None:
    """(sample artifact id, {profile: trailer sha256}) of the current valid sample, if any."""
    artifact_id = ctx.planner.expected_ids()[SAMPLE_STAGE]
    record = ctx.index.find_valid(artifact_id)
    if record is None:
        return None
    trailers = {}
    for ref in record.outputs:
        parts = (ref.uri or "").rsplit("/", 2)
        if len(parts) == 3 and parts[2] == "trailer.mp4" and ref.sha256:
            trailers[parts[1]] = ref.sha256
    return (artifact_id, trailers) if trailers else None


def _approved(ctx: Context, approvals: ApprovalStore, profile: str | None) -> bool:
    current = _current_sample(ctx)
    if current is None or not profile or profile not in current[1]:
        return False
    return approvals.is_approved(ctx.loaded.project_id, current[0], current[1][profile])


def cmd_execute(args: argparse.Namespace, ctx: Context) -> int:
    targets = _plan_targets(args, ctx)
    items = ctx.planner.plan(targets)
    pending = [i.stage for i in items if i.status == "pending"]
    if not pending:
        _emit(args, {"executed": [], "cached": [i.stage for i in items]},
              f"Nothing to do: all {len(items)} stages are cached (no transcription, diarization or detection re-run).")
        return 0
    approvals = _approvals(ctx)
    pid = ctx.loaded.project_id
    actor = _actor()

    def before_stage(spec: Any, artifact_id: str) -> None:
        state = approvals.state(pid)
        if spec.name == FULL_STAGE:
            if not _approved(ctx, approvals, ctx.config.production.profile):
                raise IllegalStateError(
                    "full production requires an approved sample. Review "
                    f"{ctx.loaded.project_dir / 'sample' / 'trailer.mp4'} then run --approve-sample (or --reject-sample)."
                )
            if state in ("APPROVED", "PRODUCTION_COMPLETED", "FAILED"):
                approvals.transition(pid, "PRODUCTION_QUEUED", actor=actor, reason="approved sample")
            if approvals.state(pid) == "PRODUCTION_QUEUED":
                approvals.transition(pid, "PRODUCTION_RUNNING", actor=actor)
        elif spec.name in ("sample_selection", SAMPLE_STAGE):
            if state != "SAMPLE_GENERATING":
                approvals.transition(pid, "SAMPLE_GENERATING", actor=actor)
        elif state not in ("ANALYZING", "SAMPLE_GENERATING"):
            approvals.transition(pid, "ANALYZING", actor=actor)

    runtime = RuntimeRunner(ctx.config.runtime, ctx.loaded.state_dir / "logs")
    try:
        executed = Runner(ctx, runtime, before_stage=before_stage).run(targets)
    except IllegalStateError:
        raise
    except PlatformError:
        if approvals.state(pid) in ("ANALYZING", "SAMPLE_GENERATING", "PRODUCTION_RUNNING"):
            approvals.transition(pid, "FAILED", actor=actor, reason="stage failure")
        raise
    state = approvals.state(pid)
    if state == "ANALYZING":
        approvals.transition(pid, "ANALYSIS_COMPLETED", actor=actor)
    if SAMPLE_STAGE in executed:
        if approvals.state(pid) == "ANALYZING":
            approvals.transition(pid, "SAMPLE_GENERATING", actor=actor)
        approvals.transition(pid, "SAMPLE_COMPLETED", actor=actor)
        approvals.transition(pid, "APPROVAL_PENDING", actor=actor, reason="awaiting review (CLI approval)")
    if FULL_STAGE in executed:
        approvals.transition(pid, "PRODUCTION_COMPLETED", actor=actor)
    lines = [f"Executed: {', '.join(executed)}", f"Project state: {approvals.state(pid)}"]
    if SAMPLE_STAGE in executed:
        out = ctx.loaded.project_dir / "sample"
        lines += [f"Sample trailer: {out / 'trailer.mp4'}", f"Reports:        {out}",
                  "Review it, then approve with --approve-sample (or reject with --reject-sample)."]
    if FULL_STAGE in executed:
        lines.append(f"Final video: {ctx.loaded.project_dir / 'output' / 'final.mp4'}")
    _emit(args, {"executed": executed, "state": approvals.state(pid)}, "\n".join(lines))
    return 0


def cmd_decide(args: argparse.Namespace, ctx: Context, decision: str) -> int:
    if decision == "PROCEED" and not ctx.config.notifications.approval.allow_cli_override:
        raise IllegalStateError("CLI approval is disabled (notifications.approval.allow_cli_override: false)")
    current = _current_sample(ctx)
    if current is None:
        raise IllegalStateError("no current sample trailer to decide on; run --mode analyze-and-sample first")
    sample_id, trailers = current
    profile = args.profile or (next(iter(trailers)) if len(trailers) == 1 else None)
    if profile is None and decision == "STOP":
        profile = next(iter(trailers))
    if profile not in trailers:
        raise IllegalStateError(f"choose which trailer to approve: --profile one of {sorted(trailers)} "
                                f"(e.g. bash start.sh approve {sorted(trailers)[0]})")
    state = _approvals(ctx).decide(ctx.loaded.project_id, sample_artifact_id=sample_id, trailer_sha256=trailers[profile],
                                   decision=decision, actor=_actor(), profile=profile)
    _emit(args, {"state": state, "sample": sample_id, "profile": profile},
          f"Recorded {decision} for the '{profile}' trailer of sample {sample_id} → state {state}")
    return 0


def cmd_approval_status(args: argparse.Namespace, ctx: Context) -> int:
    approvals = _approvals(ctx)
    pid = ctx.loaded.project_id
    current = _current_sample(ctx)
    approved_profile = approvals.approved_profile(pid, current[0]) if current else None
    payload = {"state": approvals.state(pid), "current_sample": current[0] if current else None,
               "trailers": sorted(current[1]) if current else [], "approved_profile": approved_profile,
               "approved": _approved(ctx, approvals, approved_profile), "history": approvals.history(pid)}
    text = [f"State: {payload['state']}", f"Current sample: {payload['current_sample']} (approved: {payload['approved']})"]
    text += [f"  {h}" for h in payload["history"][:10]]
    _emit(args, payload, "\n".join(text))
    return 0


def cmd_regenerate(args: argparse.Namespace, ctx: Context, kind: str, ident: str) -> int:
    pid = ctx.loaded.project_id
    if kind == "shot":
        records = [r for r in ctx.index.find_items(pid, "clip:") if ident in r.work_item]
    else:
        records = ctx.index.find_items(pid, f"line:{ident}")
    if not records:
        raise IllegalStateError(f"no generated items found for {kind} {ident}")
    for record in records:
        ctx.index.set_status(record.artifact_id, "superseded")
    for stage in (SAMPLE_STAGE, FULL_STAGE):
        record = ctx.index.find_valid(ctx.planner.expected_ids()[stage])
        if record:
            ctx.index.set_status(record.artifact_id, "superseded")
    _emit(args, {"superseded_items": [r.work_item for r in records]},
          f"Marked {len(records)} item(s) for regeneration: {[r.work_item for r in records][:10]}. "
          "Re-run the sample or full mode; everything else is reused. (A regenerated sample needs re-approval.)")
    return 0


def _email_preview(trailer: Path, workdir: Path) -> Path | None:
    """Small H.264 copy of the trailer that fits an email attachment."""
    from .notify.mailer import MAX_ATTACHMENT_BYTES
    from .stages import media

    workdir.mkdir(parents=True, exist_ok=True)
    for height, crf in ((540, 30), (360, 33), (270, 36)):
        out = workdir / f"{trailer.stem}_preview_{height}p.mp4"
        media.ffmpeg(["-i", str(trailer), "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast",
                      "-crf", str(crf), "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)])
        if out.stat().st_size <= MAX_ATTACHMENT_BYTES * 0.9:
            return out
    return None


def cmd_notify(args: argparse.Namespace, ctx: Context | None) -> int:
    from .notify.mailer import MailNotConfigured, send

    name = ctx.config.project.name if ctx else "video project"
    try:
        if args.notify == "failure":
            tail = ""
            if args.log and Path(args.log).exists():
                tail = "\n".join(Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            send(f"AI Video — run FAILED — {name}",
                 f"The pipeline stopped with an error.\n\n{args.message or ''}\n\nLast log lines:\n{tail}\n\n"
                 "Fix (or send me the log), then run: bash start.sh   (finished work is reused)")
            return 0
        if ctx is None:
            raise ConfigError("--notify sample needs a valid --project")
        current = _current_sample(ctx)
        if current is None:
            raise IllegalStateError("no finished sample trailer to report")
        sample_dir = ctx.loaded.project_dir / "sample"
        lines = [f"Your sample trailer for '{name}' is ready.", ""]
        attachments: list[Path] = []
        for profile in sorted(current[1]):
            folder = sample_dir / profile
            summary = json.loads((folder / "production_summary.json").read_text(encoding="utf-8"))
            quality = json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))["objective"]
            cost = json.loads((folder / "cost_report.json").read_text(encoding="utf-8"))
            lines += [
                f"[{profile}] {summary['duration_s']:.0f} s · {summary['characters_generated']} characters · "
                f"{summary['voices']} voices · quality: {summary['quality_summary']}",
                f"  identity consistency (cos, mean/min): {quality.get('identity_cos_mean')} / {quality.get('identity_cos_min')}",
                f"  lip-sync coverage: {quality.get('lipsync_coverage')} · clip status: {quality.get('clip_status')}",
                f"  GPU time this run: {cost.get('this_run_generation_hours')} h on {cost.get('gpu')}",
                "",
            ]
            trailer = sample_dir / f"trailer_{profile}.mp4"
            preview = _email_preview(trailer, ctx.loaded.state_dir / "work" / "email") if trailer.exists() else None
            attachments += [p for p in (preview, folder / "thumbnail.jpg") if p]
        lines += [
            f"Full-quality files on the machine: {sample_dir}",
            "  trailer_<profile>.mp4, <profile>/trailer_with_subtitles.mp4, character_references/, voice_preview/, reports",
            "",
            "The attached video is a compressed preview. Download the sample/ folder before destroying the instance.",
        ]
        send(f"AI Video — Sample trailer ready — {name}", "\n".join(lines), attachments)
        _emit(args, {"sent": True}, "Notification email sent.")
        return 0
    except MailNotConfigured as exc:
        print(f"[notify] skipped: {exc}")
        return 0


def cmd_estimate(args: argparse.Namespace, ctx: Context) -> int:
    record = ctx.index.find_valid(ctx.planner.expected_ids()[SAMPLE_STAGE])
    if record is None:
        raise IllegalStateError("no sample yet: the estimate is measured on the sample trailer run")
    ref = next(r for r in record.outputs if r.uri.endswith("/comparison.json"))
    cost = json.loads(ctx.store.read_bytes(ctx.store.key_from_uri(ref.uri)))
    _emit(args, cost, json.dumps(cost, indent=2))
    return 0


def cmd_status(args: argparse.Namespace, ctx: Context) -> int:
    cfg = ctx.config
    summary = ctx.index.summary(cfg.project.id)
    items = ctx.planner.plan() if ctx.loaded.source_video_path.exists() else []
    payload = {
        "project": cfg.project.id,
        "name": cfg.project.name,
        "type": cfg.project.project_type.value,
        "languages": [cfg.languages.source, cfg.languages.target],
        "source_video": str(ctx.loaded.source_video_path),
        "source_exists": ctx.loaded.source_video_path.exists(),
        "artifacts": {stage: dict(counts) for stage, counts in summary.items()},
        "plan": [i.as_dict() for i in items],
        "budget": cfg.budget.model_dump(),
        "approval_state": "not available until M5",
    }
    lines = [
        f"Project   {cfg.project.id} — {cfg.project.name} ({cfg.project.project_type.value})",
        f"Languages {cfg.languages.source} → {cfg.languages.target}",
        f"Source    {ctx.loaded.source_video_path} [{'found' if payload['source_exists'] else 'MISSING'}]",
        f"Budget    total={cfg.budget.max_total_spend_usd or 'open'}  sample={cfg.budget.max_sample_spend_usd or 'open'}",
        "",
    ]
    lines.append(_plan_text(items) if items else "No plan: source video missing (run --migrate-legacy or add input).")
    _emit(args, payload, "\n".join(lines))
    return 0


def cmd_migrate(args: argparse.Namespace, ctx: Context) -> int:
    legacy_root = args.migrate_legacy
    if not legacy_root.is_absolute():
        legacy_root = (ctx.loaded.platform_root / legacy_root).resolve()
    importer = LegacyImporter(ctx.loaded, ctx.graph, ctx.planner, ctx.index, ctx.store, legacy_root)
    report = importer.import_all()
    lines = [
        f"Legacy root  {report.legacy_root}",
        f"Manifest     {report.manifest_path} ({report.manifest_file_count} files; "
        + {None: "created", True: "unchanged", False: "CHANGED"}[report.manifest_unchanged] + ")",
        "",
    ]
    for item in report.stages:
        lines.append(f"  {item.stage:<24} {item.status:<24} {item.artifact_id}  {item.detail}")
    _emit(args, report.as_dict(), "\n".join(lines))
    failed = any(s.status.startswith("skipped") for s in report.stages)
    return 0 if not failed and report.manifest_unchanged is not False else 3


def cmd_verify_manifest(args: argparse.Namespace, ctx: Context) -> int:
    path = ctx.loaded.project_dir / "migration" / "manifest_v0.json"
    if not path.exists():
        raise IllegalStateError("no migration manifest yet; run --migrate-legacy first")
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_manifest(Path(expected["legacy_root"]), ctx.index)
    diff = compare_manifests(expected, actual)
    ok = not any(diff.values())
    _emit(args, {"unchanged": ok, **diff},
          f"Legacy workspace unchanged ({expected['file_count']} files)." if ok else f"Legacy workspace CHANGED: {diff}")
    return 0 if ok else 3


def cmd_verify_registry(args: argparse.Namespace) -> int:
    root = find_platform_root(Path.cwd())
    registry = ModelRegistry.load(root / "model_registry")
    results = verify_registry(registry)
    lines = []
    for model_id, result in results.items():
        hf = result.get("huggingface", {})
        gh = result.get("github", {})
        lines.append(f"{model_id:<42} HF={hf.get('license', hf.get('error', '-'))!s:<24} GH={gh.get('license_spdx', gh.get('error', '-'))}")
        lines += [f"    ! {issue}" for issue in result["issues"]]
    lines.append(f"\nWrote {registry.verification_path()}")
    _emit(args, results, "\n".join(lines))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    patterns: list[str] = list(DEFAULT_FORBIDDEN_HOST_PATTERNS)
    urls: list[tuple[str, str]] = []
    if args.project:
        loaded = load_project_config(args.project, overrides=args.overrides, email=args.email)
        cfg = loaded.config
        patterns = list(cfg.packages.forbid_hosts_matching)
        urls = [("packages.pip_index_url", cfg.packages.pip_index_url)]
        urls += [("packages.container_registries", r) for r in cfg.packages.container_registries]
        urls += [("packages.model_hubs", h) for h in cfg.packages.model_hubs]
        if cfg.storage.endpoint:
            urls.append(("storage.endpoint", cfg.storage.endpoint))
    findings = preflight(patterns=patterns, configured_urls=urls)
    failed = [f for f in findings if f.status == "fail"]
    _emit(args, [f.__dict__ for f in findings],
          "\n".join(f"[{f.status:^4}] {f.check:<32} {f.detail}" for f in findings)
          + ("\nPreflight FAILED." if failed else "\nPreflight passed."))
    return 2 if failed else 0


def cmd_init(args: argparse.Namespace) -> int:
    target: Path = args.init.resolve()
    project_id = target.name
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{1,63}", project_id):
        raise ConfigError(f"project directory name '{project_id}' must be a lowercase slug (a-z, 0-9, _ or -)")
    if (target / "config.yaml").exists():
        raise IllegalStateError(f"{target / 'config.yaml'} already exists")
    root = find_platform_root(Path.cwd())
    template = root / "projects" / "project_template"
    target.mkdir(parents=True, exist_ok=True)
    (target / "input").mkdir(exist_ok=True)
    text = (template / "config.yaml").read_text(encoding="utf-8")
    text = text.replace("__PROJECT_ID__", project_id).replace("__PROJECT_NAME__", project_id)
    (target / "config.yaml").write_text(text, encoding="utf-8")
    shutil.copy2(template / "README.md", target / "README.md")
    _emit(args, {"created": str(target)}, f"Created {target}. Edit config.yaml and put the source video in input/.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_lines=False)
    try:
        if args.config_schema:
            print(json.dumps(ProjectConfig.model_json_schema(), indent=2, ensure_ascii=False))
            return 0
        if args.init:
            return cmd_init(args)
        if args.preflight:
            return cmd_preflight(args)
        if args.verify_registry:
            return cmd_verify_registry(args)

        if args.notify:
            try:
                notify_ctx = build_context(args) if args.project else None
            except PlatformError:
                notify_ctx = None
            if args.notify == "sample" and notify_ctx is None:
                raise ConfigError("--notify sample needs a valid --project")
            return cmd_notify(args, notify_ctx)
        for flag in ("send_approval_email", "resend_approval"):
            if getattr(args, flag):
                raise NotImplementedYet("--" + flag.replace("_", "-") + " (needs the VPS approval service)", "M5-email")

        ctx = build_context(args)
        if args.print_config:
            print(json.dumps(ctx.config.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return 0
        if args.validate_only:
            return cmd_validate(args, ctx)
        if args.migrate_legacy is not None:
            return cmd_migrate(args, ctx)
        if args.verify_manifest:
            return cmd_verify_manifest(args, ctx)
        if args.approve_sample:
            return cmd_decide(args, ctx, "PROCEED")
        if args.reject_sample:
            return cmd_decide(args, ctx, "STOP")
        if args.approval_status:
            return cmd_approval_status(args, ctx)
        if args.regenerate_shot:
            return cmd_regenerate(args, ctx, "shot", args.regenerate_shot)
        if args.regenerate_dialogue:
            return cmd_regenerate(args, ctx, "line", args.regenerate_dialogue)
        if args.estimate:
            return cmd_estimate(args, ctx)
        if (args.resume and not (args.mode or args.stage)) or args.mode == "full":
            approvals = _approvals(ctx)
            current = _current_sample(ctx)
            profile = ctx.config.production.profile or (
                approvals.approved_profile(ctx.loaded.project_id, current[0]) if current else None)
            if args.resume and not args.mode:
                args.mode = "full" if _approved(ctx, approvals, profile) else "analyze-and-sample"
            if args.mode == "full" and profile and not ctx.config.production.profile:
                args.overrides = [*args.overrides, f"production.profile={profile}"]
                ctx = build_context(args)  # full production uses the approved profile
        if args.status:
            return cmd_status(args, ctx)
        if args.plan:
            return cmd_plan(args, ctx)
        if args.mode or args.stage:
            return cmd_execute(args, ctx)
        return cmd_status(args, ctx)
    except PlatformError as exc:
        log.error("%s", exc)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

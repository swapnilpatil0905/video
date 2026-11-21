# AI Video Remake / Localization Platform

This is a personal self-improvement learning project. It is a reusable, config-driven platform that analyses a source video, generates a representative sample trailer for approval, and then produces a fully localized remake. The first project is a Chinese drama remade in Hindi (`projects/project_001`).

- Architecture and plan: [`docs/specs/platform-technical-proposal.md`](docs/specs/platform-technical-proposal.md)
- Approval workflow: [`docs/specs/email-approval-workflow.md`](docs/specs/email-approval-workflow.md)
- Milestone reports: [`docs/milestones/`](docs/milestones/)

## Rules this repo follows

- Open-source software and open-weight models only. No paid AI APIs.
- **Public sources only.** Packages come from public PyPI. `run.sh` overrides any machine-wide pip/uv index for this project only; it never edits global settings. Container images come from docker.io, ghcr.io, nvcr.io or quay.io. Weights come from huggingface.co or github.com. `--preflight` checks all of this.
- GPU work runs on a **personal Vast.ai** account. Spend caps are optional (`budget.*`, open by default).
- Secrets live only in environment variables or `.env` (gitignored), never in config files. The loader rejects literal secrets.

## Run it on Vast.ai

1. Rent 1× RTX PRO 6000 (or H100/A100 80 GB) with the NVIDIA CUDA template and a 400 GB disk.
2. Upload `dist/video-platform-0.4.0.zip` and unzip it.
3. Create `.env` (HF token and Gmail app password).
4. Run `bash start.sh notify-test`, then `nohup bash start.sh > /workspace/run.log 2>&1 &`.
5. You get an email when the trailer is ready.

Step-by-step: [`docs/RUNBOOK.md`](docs/RUNBOOK.md) → *Installation*.

## Developer commands

```bash
./run.sh --bootstrap                     # creates .venv-platform from https://pypi.org/simple
./run.sh --preflight                     # verify no forbidden package indexes / endpoints
./run.sh --test                          # unit tests

P=projects/project_001/config.yaml
./run.sh --project $P --validate-only    # config, models, licences, source video
./run.sh --project $P --migrate-legacy   # adopt the pre-platform analysis outputs (hardlinks, no re-run)
./run.sh --project $P --status           # artifacts + cache-aware plan
./run.sh --project $P --mode analyze     # "Nothing to do" once legacy results are imported
./run.sh --project $P --verify-manifest  # legacy files unchanged?
./run.sh --verify-registry               # model licences vs public HF / GitHub APIs
./run.sh --init projects/project_002     # new project from the template
```

Exit codes: `0` ok · `2` config error · `3` illegal state · `4` service unavailable · `5` stage failure · `6` not implemented yet (later milestone).

## Layout

```
run.py / run.sh          CLI entry points
configs/                 default.yaml, pipeline.yaml (stage DAG), hardware.yaml, profiles.yaml
model_registry/          one YAML per model (task, weights, licence, hardware, limitations)
projects/<id>/           config.yaml (+ input/, artifacts/, state/, migration/ — gitignored)
src/video_platform/      core, config, schemas, storage, orchestration, models, ingestion, migration, cli
tests/unit/              pytest suite
pipeline/                legacy pre-platform scripts (kept for provenance; superseded stage by stage)
```

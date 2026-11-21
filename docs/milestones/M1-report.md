# Milestone 1 report: inspection, architecture, skeleton

| | |
|---|---|
| Date | 2026-11-21 |
| Status | **Code complete. Tests written but NOT yet executed.** See "Blocked: test execution". |

## Implemented

| Area | Files | Notes |
|---|---|---|
| CLI | `run.py`, `run.sh`, `src/video_platform/cli.py` | All Part 9 commands are wired up. M1 implements `--validate-only`, `--print-config`, `--config-schema`, `--status`, `--plan`, `--mode analyze` (cache-aware), `--init`, `--migrate-legacy`, `--verify-manifest`, `--verify-registry` and `--preflight`. Later-milestone commands exit with code 6 and name their milestone. |
| Package isolation | `pip.conf`, `uv.toml`, `run.sh`, `core/network_policy.py` | Public PyPI is used for this project only. The preflight fails on any configured forbidden index, registry or endpoint. |
| Config | `config/schema.py`, `config/loader.py`, `configs/default.yaml` | Pydantic schema with `extra=forbid`. Precedence: defaults → project → `--email` → `--set`. Supports `${VAR}`, `${VAR:-default}` and `secret://env/NAME`. Literal secrets are rejected and secrets are held in `SecretStr`. Cross-checks cover the aspect ratio, the target language, forbidden hosts and "never clone source voices". |
| Schemas | `schemas/common.py`, `media.py`, `audio.py`, `characters.py` | Versioned contracts with stable IDs (`SH_`, `SEG_`, `SPK_`, `SRC_CHAR_`). Timestamps are in seconds on the source timeline. |
| Storage | `storage/store.py`, `storage/index.py` | Local store (hardlink import) and an S3/MinIO store (boto3 is an optional extra). SQLite artifact index with a file-hash cache. |
| Orchestration | `orchestration/graph.py`, `planner.py`, `configs/pipeline.yaml` | DAG validation (cycles, unknown deps). Deterministic artifact IDs from stage + version + upstream IDs + params hash + pinned model revision. A change re-plans only the affected stage and its descendants. |
| Model registry | `models/registry.py`, `model_registry/*.yaml` (27 entries) | Licence policy check per `licensing.usage`. `--verify-registry` queries the public HF and GitHub APIs and writes `model_registry/_verification.json`. |
| Benchmark harness | `models/benchmark.py`, `models/adapter.py` | Skeleton plus the adapter contract. Real cases arrive in M2. |
| Ingestion helper | `ingestion/probe.py` | ffprobe → `VideoMetadata`. Handles rotation and `0/0` frame rates, and records duration-mismatch warnings. |
| Legacy migration | `migration/legacy.py` | sha256 manifest of the legacy workspace. Hardlinks the legacy files in. Converts them to platform schemas. Adopts a stage **only if** the configured params equal the legacy script constants; otherwise that stage and its descendants are re-planned. |
| Projects | `projects/project_001/config.yaml`, `projects/project_template/` | project_001 pins the analysis params to the legacy values, so the legacy outputs become cache hits. |
| Tests | `tests/unit/*` (8 files, about 50 tests) | Covers the config, hashing, network policy, graph, storage, registry, planner, and an end-to-end legacy migration on a miniature workspace using the real file formats. |

## Verified so far

- **All sources byte-compile** under Python 3.11 (`compileall`).
- **Stdlib-only modules smoke-tested:** hashing, network policy, secret detection, storage-key validation, hardlink store.
- **Preflight on this Mac:**
  - Raw environment: 3 failures (machine-wide pip/uv settings pointed at a non-public mirror).
  - With the `run.sh` overrides: all checks pass.
- **Duplicate source deleted** after the two sha256 values matched (`69eb4aed…7862f`).

## Blocked: test execution

The pytest suite and the real `--migrate-legacy` run on project_001 need pydantic, PyYAML and pytest, which could not be installed in the development sandbox. They run on the GPU machine: `./run.sh --bootstrap && ./run.sh --test`.

## Acceptance criteria (proposal §20, M1)

| Criterion | Status |
|---|---|
| `--validate-only` passes for project_001 and fails clearly on malformed configs | Implemented and tested (≥10 negative cases). **Not yet executed.** |
| `--status` lists imported legacy artifacts as valid cache hits, and re-running analysis triggers zero transcription, diarization or detection work | Implemented and tested on a miniature workspace. **Real project_001 run pending.** |
| The legacy sha256 manifest is unchanged after migration | `--verify-manifest`. **Pending the run.** |
| Every registered model has a verified code and weight licence with source URLs | Sources are recorded for all 27 entries. **Verification pending network access** (`--verify-registry`). IndicF5 and MuseTalk weights are `unknown`, so please share their licence pages if you have them. |
| Unit tests cover time handling, schemas, config, hashing and the cache | Written. **Not yet executed.** |

## Known issues and limitations

- The analysis stage *executors* are M2. M1 can only plan them and adopt the legacy outputs.
- The legacy transcript has 3 zero or negative-length segments. They are flagged `invalid_timestamp`, not altered.
- Legacy speaker↔character results are imported as `uncertain` co-occurrence priors only. Active speaker detection is M2.
- Legacy `frame_sampling` used one midpoint frame per shot. It is kept for cache purposes; M2 adds multi-frame sampling as a new configuration.
- The model `pinned_revision` for legacy models is `legacy-unpinned`. `--verify-registry` records upstream SHAs separately, so pinning does not invalidate the legacy cache.
- GPU requirements for M1: none. Everything in M1 is CPU-only.

## Next: Milestone 2 (needs your approval)

Video and audio analysis executors on Vast.ai, including active speaker detection, a calibration set, multi-frame sampling and VLM tagging.

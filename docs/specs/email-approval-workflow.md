# Spec: Email-Based Automated Approval and Production Trigger

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-11-21 |
| Supersedes | Earlier requirement for manual approval via terminal / interactive CLI prompt |
| Context | Personal self-improvement learning project. **Open-source software and public sources only.** Control plane on a **personal VPS**, GPUs on the owner's **personal Vast.ai account**. |
| Scope | Approval service, notification system, project state machine, production job queue, CLI commands, review assets |

---

## 1. Summary

After the platform finishes analysis and generates a sample trailer, it publishes review assets to persistent storage and emails the configured recipient. The email has a secure review link and two actions, **PROCEED** and **STOP**. Following an action link opens a confirmation page. The decision only takes effect when the recipient submits that page as a token-authorized `POST`. A PROCEED decision creates exactly one production job, in the same database transaction. A GPU worker picks the job up whenever GPU capacity is available. Production reuses all completed analysis and approved sample assets.

Once the user starts the pipeline, nobody needs to touch the terminal again. No GPU instance stays running while the platform waits for a decision.

### 1.1 Goals

- Run end to end from one command to the final video link, with no further terminal interaction.
- Make sure email scanners, link previews and `GET` requests can never authorize production.
- Create production jobs exactly once, even with concurrent clicks, retries or restarts.
- Keep the approval service fully independent of GPU workers.
- Make every state change and decision auditable.
- Keep secrets out of config files and source control.

### 1.2 Non-goals

- Multi-user role-based access control. Only an optional email-OTP gate is in scope (see `require_authentication`).
- Approval through channels other than email, such as Slack or SMS. The provider interface must not rule these out.
- Editing or revising a sample from the review page.
- Billing and payment collection. The cost estimate is informational only.

### 1.3 Current-state notes (this repository)

The pipeline today is a set of standalone scripts in `pipeline/` with hardcoded relative paths (`transcripts/`, `diarization/`, `scenes/`, `faces/`, `characters/`, `timeline/`, `verification/`). It has no `run.py`, no `projects/` layout and no per-stage manifests. This feature depends on the following prerequisite work, which belongs in this spec only as far as approval needs it (see §13):

- A `run.py` orchestrator with a project config and a per-project workspace.
- Stage wrappers around the existing scripts that record their inputs, outputs and content hashes.

---

## 2. Architecture

```
                  ┌────────────────────────────── always-on, CPU-only ───────────────────────────────┐
                  │                                                                                  │
 run.py ──admin──▶│  Approval Service (FastAPI)      Notification Dispatcher      Expiry Sweeper     │
 (CLI)    API     │   • /review, /approval pages      • email outbox → provider    • APPROVAL_PENDING │
                  │   • decision POST                 • retries / backoff            → EXPIRED        │
                  │   • admin API                                                                    │
                  │             │                              │                         │           │
                  │             └──────────────┬───────────────┴─────────────────────────┘           │
                  │                            ▼                                                     │
                  │                    PostgreSQL (state, approvals, jobs, outbox, audit)            │
                  └────────────────────────────┬─────────────────────────────────────────────────────┘
                                               │  job queue (SKIP LOCKED leasing)
                                               ▼
               GPU Launcher (Vast.ai API) ──starts──▶ GPU Worker(s) ──HTTPS──▶ Worker API + MinIO (S3 API)
               watches queue depth                 run stages, heartbeat lease    trailers, thumbnails, outputs
```

| Component | Responsibility | Runs on |
|---|---|---|
| `run.py` | Starts projects and runs analysis and sample stages. Calls the admin API to register review assets and request approval. Hosts the admin CLI commands. | Whichever machine starts the job (GPU for analysis/sample) |
| Approval Service | Serves the HTTPS review page, confirmation pages and decision `POST`, and the admin API. Validates state transitions. Creates jobs atomically. | Small always-on CPU container or VM |
| Notification Dispatcher | Drains `email_outbox` and sends through the configured provider, retrying on failure. | Same process as the service (background task) or a separate worker |
| Expiry Sweeper | Moves approval requests past their deadline to `APPROVAL_EXPIRED` and queues the expiry email. | Periodic task, every 5 minutes |
| PostgreSQL | Source of truth for project state, approvals, jobs, outbox and audit log. | Managed DB recommended |
| Object Storage | Stores trailer, thumbnail, reports and final video. Serves media through short-lived presigned HTTPS URLs. | **MinIO** on the personal VPS (any S3-compatible endpoint works) |
| GPU Launcher | When the queue holds `QUEUED` jobs and no worker is alive, starts a Vast.ai instance through the **personal** account API key, within the spend caps. Can be set to `manual`. | Background task in the control plane |
| GPU Worker | Leases a job, resumes the pipeline from the approved snapshot, heartbeats, and exits after an idle timeout. | Ephemeral GPU instance |

**Key invariant:** the GPU worker plays no part in recording a decision. If every GPU instance is stopped, PROCEED still records the decision and queues the job. The job sits in `QUEUED` until a worker leases it.

The job queue lives in PostgreSQL and uses `SELECT … FOR UPDATE SKIP LOCKED`. **Postgres is never exposed to the internet.** Vast.ai workers lease, heartbeat and complete jobs through an authenticated HTTPS **worker API** on the approval service (`POST /worker/lease`, `/worker/{job}/heartbeat`, `/worker/{job}/progress`, `/worker/{job}/complete`, `/worker/{job}/fail`). It uses a per-launch worker token (random 256-bit, stored hashed, scoped to one job, revoked when the instance is destroyed). The SQL below runs inside those endpoints. This keeps job creation in the same transaction as the decision with no separate broker. A broker or orchestrator (Celery, Temporal, Prefect, and so on) may be added later. If one is added, the system must use the transactional outbox pattern: insert the job row in the decision transaction, then let a relay publish it to the broker.

---

## 3. Configuration

### 3.1 Project configuration (`projects/<project_id>/config.yaml`)

```yaml
project:
  id: "project_001"                 # slug; internal only, never placed in URLs
  name: "Midsummer — EN Dub"
  source_video: "input/source.mp4"
  target_language: "en"

storage:
  provider: minio                   # minio (default, self-hosted) | s3 (any S3-compatible) | local (dev/tests)
  endpoint: "${STORAGE_ENDPOINT}"     # e.g. https://media.<your-vps-domain>
  bucket: "${REVIEW_BUCKET}"
  prefix: "projects/"
  presigned_url_ttl_minutes: 60

notifications:
  enabled: true
  approval_email: "user@example.com"
  provider: smtp                    # smtp (default: personal mailbox app password or self-hosted) | console (dev) | file (test) | ses / sendgrid (optional plugins)
  smtp:
    host: "${SMTP_HOST}"
    port: 587
    username: "${SMTP_USERNAME}"
    password: "${SMTP_PASSWORD}"
    use_tls: true
  ses:
    region: "${AWS_REGION}"         # credentials from the standard AWS chain, never from config
  sendgrid:
    api_key: "${SENDGRID_API_KEY}"
  sender:
    email: "video-platform@example.com"
    name: "AI Video Production Platform"

  events:                           # per-project notification toggles
    sample_ready: true              # cannot be disabled while approval.enabled is true
    approval_received: true
    production_queued: true
    production_started: true
    production_progress: false
    progress_milestones_percent: [25, 50, 75]
    production_completed: true
    production_failed: true
    approval_expired: true
    project_stopped: true

  approval:
    enabled: true
    expiration_hours: 72            # allowed range 1–720
    allow_resend: true
    require_authentication: false   # true → email one-time code required before a decision POST
    reminder_hours_before_expiry: 12   # 0 disables the reminder

approval_service:
  public_base_url: "${APPROVAL_PUBLIC_BASE_URL}"   # must be https:// unless env=dev
  admin_api_token: "${APPROVAL_ADMIN_TOKEN}"
```

### 3.2 Secret handling rules

1. **Interpolation.** The config loader resolves `${VAR}` from the process environment. It also accepts secret references of the form `secret://<backend>/<name>`, with backends `aws-sm`, `gcp-sm`, `vault` and `env`. A missing variable is a startup error, never an empty string.
2. **Literal-secret rejection.** If any of these keys holds a literal value instead of `${…}` or `secret://…`, the loader refuses to start: `password`, `api_key`, `admin_api_token`, `secret`, `token`, and any key ending in `_secret` or `_password`. The error message names the key and never prints the value.
3. **Redaction.** Logs, audit records, the status output and exception messages never contain resolved secret values. A `Secret` wrapper type has a `__repr__` that returns `"***"`.
4. **Source control.** Ship a `.env.example` and add `.env`, `*.env`, and `projects/*/secrets*` to `.gitignore`.
5. **Service-side secrets** (`TOKEN_HASH_PEPPER`, `CSRF_SIGNING_KEY`, `DATABASE_URL`) exist only in the approval service's environment. Project configs never contain them.

### 3.3 CLI override

```
python run.py --project projects/project_001/config.yaml --mode analyze-and-sample --email user@example.com
```

- `--email` overrides `notifications.approval_email` for this run. The CLI checks the address against RFC 5322 basic syntax and a single-address rule: no lists and no display names.
- The effective recipient is stored on the project record (`projects.approval_email`). It is also written into the immutable approval request, so later config edits do not redirect an outstanding request.
- If notifications are enabled and the CLI gets no recipient from either the config or `--email`, it fails before analysis starts. That way the user does not wait hours to find out.

### 3.4 Email provider interface

```python
class EmailMessage(TypedDict):
    to: str
    subject: str
    html_body: str
    text_body: str
    inline_images: list[InlineImage]   # cid, bytes, mime; total ≤ 300 KB
    headers: dict[str, str]
    idempotency_key: str               # outbox row id

class EmailProvider(Protocol):
    name: str
    def send(self, msg: EmailMessage) -> SendResult: ...   # returns provider_message_id
    def classify_error(self, exc: Exception) -> Literal["retryable", "permanent"]: ...
```

Providers: `SmtpProvider` (STARTTLS required when `use_tls`), `SesProvider` (boto3 `send_raw_email`), `SendGridProvider` (HTTP API), `ConsoleProvider` (dev; prints and writes `.eml`), and `FileProvider` (tests; writes to a directory). New providers register through an entry-point group, `video_platform.email_providers`.

---

## 4. Project State Machine

### 4.1 States

`INITIALIZED`, `ANALYZING`, `ANALYSIS_COMPLETED`, `SAMPLE_GENERATING`, `SAMPLE_COMPLETED`, `APPROVAL_PENDING`, `APPROVED`, `PRODUCTION_QUEUED`, `PRODUCTION_RUNNING`, `PRODUCTION_COMPLETED`, `STOPPED`, `APPROVAL_EXPIRED`, `FAILED`.

### 4.2 Allowed transitions

| From | To | Trigger | Actor |
|---|---|---|---|
| `INITIALIZED` | `ANALYZING` | pipeline start | run.py |
| `ANALYZING` | `ANALYSIS_COMPLETED` | all analysis stages succeed | run.py |
| `ANALYSIS_COMPLETED` | `SAMPLE_GENERATING` | sample stage start | run.py |
| `SAMPLE_GENERATING` | `SAMPLE_COMPLETED` | trailer, thumbnail, reports, estimate generated **and** uploaded and registered | run.py, via admin API |
| `SAMPLE_COMPLETED` | `APPROVAL_PENDING` | approval request created and sample-ready email queued | approval service |
| `APPROVAL_PENDING` | `APPROVED` | valid PROCEED `POST` | approval service |
| `APPROVED` | `PRODUCTION_QUEUED` | production job inserted, **same transaction** as above | approval service |
| `PRODUCTION_QUEUED` | `PRODUCTION_RUNNING` | worker leases job | GPU worker |
| `PRODUCTION_RUNNING` | `PRODUCTION_QUEUED` | lease expired (worker died or was preempted) | sweeper |
| `PRODUCTION_RUNNING` | `PRODUCTION_COMPLETED` | final render uploaded | GPU worker |
| `APPROVAL_PENDING` | `STOPPED` | valid STOP `POST` | approval service |
| `APPROVAL_PENDING` | `APPROVAL_EXPIRED` | `now > expires_at` | sweeper, or lazily on request |
| `APPROVAL_PENDING` | `APPROVAL_PENDING` | resend (old request superseded, new one created) | admin CLI |
| `APPROVAL_EXPIRED` | `APPROVAL_PENDING` | resend, with `allow_resend: true` | admin CLI |
| `STOPPED` | `APPROVAL_PENDING` | new approval request for a revised sample (§9.3) | admin CLI |
| `STOPPED` | `SAMPLE_GENERATING` | sample revision run | run.py |
| `ANALYZING`, `SAMPLE_GENERATING`, `PRODUCTION_RUNNING` | `FAILED` | unrecoverable stage error, or retries exhausted | run.py / worker |
| `FAILED` | the state before the failure (`ANALYZING`, `SAMPLE_GENERATING`, or `PRODUCTION_QUEUED`) | admin retry | admin CLI |

`APPROVED` is a transient state. It is written and left within the same transaction, and it remains visible in `project_state_history`.

### 4.3 Enforcement

- One module, `platform/state_machine.py`, owns `ALLOWED: dict[State, set[State]]` and `transition(conn, project_id, to_state, actor, reason)`.
- `transition` locks the project row (`SELECT … FOR UPDATE`) and checks that `current → to_state` appears in `ALLOWED`. It then updates `projects.state` and `projects.state_version` (optimistic counter) and appends rows to `project_state_history` and `audit_log`. All of this happens inside the caller's transaction.
- An invalid transition raises `IllegalTransition`. Nothing is written, and the attempt itself goes to the audit log as `transition_rejected`.
- **Hard rule:** production job rows can be inserted only through `create_production_job()`. That function asserts the project is moving `APPROVED → PRODUCTION_QUEUED` in the same transaction and that a `decision='PROCEED'` row exists for the active approval request. A DB check (§6, trigger `trg_job_requires_approval`) backs this up. If the project is in `APPROVAL_PENDING`, `STOPPED` or `APPROVAL_EXPIRED`, production cannot start.
- The worker double-checks before running any production stage. It re-reads the job and its approval and aborts if `approval_requests.status != 'APPROVED'`.

---

## 5. Sample Completion → Approval Email Flow

These run automatically at the end of `--mode analyze-and-sample`:

| # | Step | Output | Notes |
|---|---|---|---|
| 1 | Generate trailer | `sample/trailer.mp4` | H.264/AAC, faststart, ≤1080p |
| 2 | Generate thumbnail | `sample/thumbnail.jpg` | 1280×720 for the review page, plus a ≤150 KB 640×360 email variant |
| 3 | Character coverage report | `sample/character_coverage.json` | Built from `characters/character_database.json` and `speaker_character_refined.json`: important characters included, missing ones, and uncertain ones (low match confidence) |
| 4 | Quality report | `sample/quality_report.json` | `summary` (one line), `score` (0–100), `issues[]` with severity. Sources include `timeline_quality_report.json` and audit outputs. |
| 5 | Cost estimate | `sample/cost_estimate.json` | `gpu_hours`, `cost`, `currency`, `assumptions`. Extrapolated from measured sample-stage GPU seconds per output minute multiplied by full duration, plus stage overheads. Reports a low–high range. |
| 6 | Upload to object storage | `s3://…/<project_uuid>/samples/<sample_version>/…` | Content-addressed. Each upload is verified by checksum. |
| 7 | Freeze sample snapshot | `sample_manifest.json` | Lists every asset that production will reuse (§8), with SHA-256 hashes. The manifest's own hash is `sample_manifest_hash`. |
| 8 | Register with service | `POST /admin/projects/{id}/samples` | Records the assets and moves `SAMPLE_GENERATING → SAMPLE_COMPLETED`. |
| 9 | Create approval request | `POST /admin/projects/{id}/approval-requests` | Generates the review and approval tokens (§7) and moves to `APPROVAL_PENDING`. Enqueues the sample-ready email in the **same transaction**. |
| 10 | Dispatch | Notification dispatcher | Sends asynchronously and retries independently of run.py. |
| 11 | Release GPU | run.py exits 0 | With `--shutdown-on-complete`, or when run by the launcher, the instance stops. |

If step 6, 8 or 9 fails, run.py retries three times with backoff. If all retries fail, it exits non-zero and leaves the project in `SAMPLE_GENERATING`. Re-running the same command is idempotent: content-addressed uploads are skipped, and registration is keyed by `(project_id, sample_manifest_hash)`.

If notifications are disabled, the flow stops at `SAMPLE_COMPLETED`. Approval can then be requested later with `--send-approval-email`.

---

## 6. Data Model (PostgreSQL)

```sql
CREATE TABLE projects (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug            text UNIQUE NOT NULL,           -- "project_001"; never exposed in URLs
  name            text NOT NULL,
  source_video    text NOT NULL,                  -- display name only (basename)
  target_language text NOT NULL,
  approval_email  text,
  notification_settings jsonb NOT NULL,
  state           text NOT NULL,
  state_version   bigint NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE project_state_history (
  id bigserial PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects,
  from_state text, to_state text NOT NULL, actor text NOT NULL, reason text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE samples (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id           uuid NOT NULL REFERENCES projects,
  version              int NOT NULL,
  sample_manifest_hash text NOT NULL,
  duration_seconds     numeric NOT NULL,
  trailer_object_key   text NOT NULL,
  thumbnail_object_key text NOT NULL,
  email_thumbnail_key  text NOT NULL,
  coverage             jsonb NOT NULL,
  quality              jsonb NOT NULL,
  cost_estimate        jsonb NOT NULL,
  manifest             jsonb NOT NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, version),
  UNIQUE (project_id, sample_manifest_hash)
);

CREATE TABLE approval_requests (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id           uuid NOT NULL REFERENCES projects,
  sample_id            uuid NOT NULL REFERENCES samples,
  recipient_email      text NOT NULL,
  approval_token_hash  bytea NOT NULL UNIQUE,     -- HMAC-SHA256(pepper, token)
  review_token_hash    bytea NOT NULL UNIQUE,
  status               text NOT NULL CHECK (status IN
                         ('PENDING','APPROVED','STOPPED','EXPIRED','SUPERSEDED')),
  expires_at           timestamptz NOT NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  decided_at           timestamptz,
  superseded_by        uuid REFERENCES approval_requests,
  otp_hash             bytea, otp_expires_at timestamptz, otp_attempts int DEFAULT 0
);
-- at most one live request per project
CREATE UNIQUE INDEX one_pending_request_per_project
  ON approval_requests(project_id) WHERE status = 'PENDING';

CREATE TABLE approval_decisions (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  approval_request_id uuid NOT NULL UNIQUE REFERENCES approval_requests,  -- one decision per request
  project_id          uuid NOT NULL REFERENCES projects,
  decision            text NOT NULL CHECK (decision IN ('PROCEED','STOP')),
  decided_at          timestamptz NOT NULL DEFAULT now(),
  client_ip           inet,
  user_agent          text,
  authenticated_via   text NOT NULL             -- 'token' | 'token+otp'
);

CREATE TABLE production_jobs (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id          uuid NOT NULL REFERENCES projects,
  approval_request_id uuid NOT NULL UNIQUE REFERENCES approval_requests, -- one job per approval
  sample_id           uuid NOT NULL REFERENCES samples,
  sample_manifest_hash text NOT NULL,
  status              text NOT NULL CHECK (status IN
                        ('QUEUED','RUNNING','COMPLETED','FAILED','CANCELLED')),
  attempt             int NOT NULL DEFAULT 0,
  max_attempts        int NOT NULL DEFAULT 3,
  lease_owner         text,
  lease_expires_at    timestamptz,
  progress_percent    int NOT NULL DEFAULT 0,
  current_stage       text,
  output_object_key   text,
  error               text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz, finished_at timestamptz
);
-- at most one active production job per project
CREATE UNIQUE INDEX one_active_job_per_project
  ON production_jobs(project_id) WHERE status IN ('QUEUED','RUNNING');

CREATE TABLE email_outbox (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id       uuid NOT NULL REFERENCES projects,
  event_type       text NOT NULL,
  dedupe_key       text NOT NULL UNIQUE,        -- e.g. "sample_ready:<approval_request_id>"
  recipient        text NOT NULL,
  template         text NOT NULL,
  context          jsonb NOT NULL,              -- no raw tokens; see §7.4
  status           text NOT NULL CHECK (status IN ('PENDING','SENDING','SENT','FAILED')),
  attempts         int NOT NULL DEFAULT 0,
  next_attempt_at  timestamptz NOT NULL DEFAULT now(),
  provider_message_id text,
  last_error       text,
  created_at timestamptz NOT NULL DEFAULT now(), sent_at timestamptz
);

CREATE TABLE audit_log (                         -- append-only
  id          bigserial PRIMARY KEY,
  project_id  uuid,
  event       text NOT NULL,
  actor       text NOT NULL,                     -- 'cli', 'recipient', 'worker:<id>', 'sweeper', 'system'
  details     jsonb NOT NULL,
  client_ip   inet,
  created_at  timestamptz NOT NULL DEFAULT now()
);
-- REVOKE UPDATE, DELETE ON audit_log FROM app_role;
```

`trg_job_requires_approval` is a `BEFORE INSERT` trigger on `production_jobs`. It raises an error unless all of these hold: an `approval_decisions` row with `decision='PROCEED'` exists for `NEW.approval_request_id`, its project matches `NEW.project_id`, and its sample hash matches `NEW.sample_manifest_hash`.

Schema migrations are managed with Alembic.

---

## 7. Tokens and Secure Links

### 7.1 Token types

| Token | Purpose | Lifetime | Grants |
|---|---|---|---|
| Review token | `GET /review/{review_token}`: watch the trailer and read the reports | `expires_at` + 30 days (read-only), revoked on supersede | Read-only view |
| Approval token | `GET /approval/{approval_token}/{proceed\|stop}` (confirmation page) and `POST /approval/{approval_token}/decision` | `expires_at` | One decision |
| CSRF form token | Hidden field on the confirmation page | 30 minutes | Binds a `POST` to a freshly rendered confirmation page |

The review and approval tokens are separate. A forwarded "watch" link, a shared trailer URL or a pasted review page link therefore never carries decision authority.

### 7.2 Generation and storage

- Generation: `secrets.token_urlsafe(32)`, which gives 256 bits of entropy (43 URL-safe characters). Guessing a token is not feasible.
- Storage: only `HMAC-SHA256(TOKEN_HASH_PEPPER, token)` is stored. A DB leak alone does not reveal usable tokens, and without the pepper an attacker cannot run an offline guess.
- Lookup is by hash equality, through the unique index. An unknown, malformed, expired or superseded token gets the same generic "This link is no longer valid" page with HTTP 404. The response never reveals whether a project or request exists.
- Binding: the token row links to exactly one `approval_request`, which links to one `project` and one `sample`. The service derives every identifier from the token. No project ID, slug or path is accepted from the URL or form, which rules out cross-project access.
- The plaintext token exists only in memory, while the email is rendered (§7.4), and in the recipient's inbox.

### 7.3 Routes

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/review/{review_token}` | Review page: player using a presigned trailer URL generated per page load (TTL `presigned_url_ttl_minutes`), thumbnail, coverage, quality, cost estimate, expiry time, and current decision status. No decision buttons; it links to the action pages only through the email. |
| `GET` | `/approval/{approval_token}/proceed` | **Confirmation page only. No state change.** Shows a project summary, "You are about to start full production (est. $X, Y GPU-hours)", a **Confirm PROCEED** button and a link to the STOP page. |
| `GET` | `/approval/{approval_token}/stop` | Confirmation page for STOP. No state change. |
| `POST` | `/approval/{approval_token}/otp` | Only when `require_authentication: true`. Emails a 6-digit code to `recipient_email` (10-minute TTL, 5 attempts, 3 sends per hour). |
| `POST` | `/approval/{approval_token}/decision` | Form fields: `action` (`proceed` or `stop`), `csrf`, and `otp` if required. Performs the decision (§8.1). Responds with `303` to `GET /approval/{approval_token}/result`. |
| `GET` | `/approval/{approval_token}/result` | Shows the recorded outcome. Idempotent and read-only. |
| `GET` | `/healthz`, `/readyz` | Liveness, and readiness with a DB check |
| | `/admin/*` | Bearer `APPROVAL_ADMIN_TOKEN` auth. Used by the CLI only (§10). |

### 7.4 Link-scanner and security protections

- **No `GET` request changes state.** This is enforced by a middleware assertion in tests: every non-`POST` route runs inside a read-only transaction (`SET TRANSACTION READ ONLY`).
- The decision must be a `POST` carrying a valid CSRF token. The CSRF token is `HMAC(CSRF_SIGNING_KEY, approval_token_hash ‖ action ‖ issued_at)`, issued at most 30 minutes before the `POST`. Scanners that only `GET` cannot produce one. Scanners that render JavaScript still do not submit forms.
- The confirmation form has no JavaScript auto-submit. The button is a plain `<button type="submit">` that requires a real click.
- Response headers on all token pages:
  - `Cache-Control: no-store`
  - `Referrer-Policy: no-referrer`, so the token does not leak to the CDN or video host
  - `X-Robots-Tag: noindex, nofollow`
  - `Content-Security-Policy: default-src 'self'; media-src <storage-origin>; img-src 'self' <storage-origin>; frame-ancestors 'none'`
  - `Strict-Transport-Security: max-age=31536000`
  - `X-Content-Type-Options: nosniff`
- HTTPS only. In non-dev environments the service refuses to start unless `public_base_url` starts with `https://`. TLS terminates at a reverse proxy or load balancer, the app trusts `X-Forwarded-*` only from configured proxy CIDRs, and plain-HTTP requests are redirected.
- URLs contain only the opaque token and the action. They never include filesystem paths, bucket names, internal UUIDs, emails or credentials. Presigned media URLs point at storage keys built from a random `project_uuid` prefix, never local paths.
- Outbox `context` holds the approval request ID, not the plaintext token. The dispatcher cannot rebuild a token from it. So the token is generated and the email rendered in the same transaction that creates the request, and only the **rendered HTML/text body** is stored in the outbox, encrypted at rest with `pgcrypto` or an application key. Once a request is no longer `PENDING`, that stored body is purged.
- **Rate limiting** uses a sliding window in Postgres or Redis:
  - Per IP: 30 requests/min across `/approval/*` and `/review/*`, and 10 decision `POST`s per hour.
  - Per token hash: 20 requests/min.
  - Globally: invalid-token responses beyond 100 per minute raise an alert.
  - A limited request gets `429` with `Retry-After`.

---

## 8. Decision Handling

### 8.1 PROCEED transaction (single DB transaction, `SERIALIZABLE` or `READ COMMITTED` + row locks)

```
BEGIN;
  ar  := SELECT * FROM approval_requests WHERE approval_token_hash = $h FOR UPDATE;
  IF ar IS NULL                          → 404 generic
  verify CSRF (and OTP if required)      → 400 / 401 on failure (no state change)
  IF ar.status = 'APPROVED' AND existing decision = PROCEED
                                         → COMMIT; 303 → result ("already approved")   -- idempotent
  IF ar.status IN ('STOPPED','SUPERSEDED','EXPIRED') → 409 page explaining current status
  IF now() > ar.expires_at               → mark EXPIRED + project → APPROVAL_EXPIRED; COMMIT; 410 page
  p   := SELECT * FROM projects WHERE id = ar.project_id FOR UPDATE;
  IF p.state != 'APPROVAL_PENDING'       → 409
  INSERT approval_decisions (ar.id, p.id, 'PROCEED', now(), ip, ua, via);   -- UNIQUE(ar.id)
  UPDATE approval_requests SET status='APPROVED', decided_at=now();          -- token now unusable
  transition(p, 'APPROVED'); 
  INSERT production_jobs (p.id, ar.id, ar.sample_id, sample_hash, 'QUEUED'); -- UNIQUE(ar.id) + partial unique
  transition(p, 'PRODUCTION_QUEUED');
  INSERT email_outbox ('approval_received:<ar.id>'), ('production_queued:<job.id>');
  INSERT audit_log ('decision_recorded', ...), ('production_job_created', ...);
COMMIT;
```

- **Duplicate-click protection.** Concurrent `POST`s are serialized by the row lock on `approval_requests`. The second request sees `APPROVED` and returns the "already approved" result. If the lock is bypassed somehow, the `UNIQUE` constraints on `approval_decisions.approval_request_id` and `production_jobs.approval_request_id`, plus the `one_active_job_per_project` index, make the second insert fail. That failure is caught, the transaction rolls back, the current state is re-read, and the result page is shown.
- **Token invalidation** is implicit: once a request has `status != 'PENDING'`, its token can no longer make decisions. The hash stays stored so the result page and audit trail keep working, and the result page shows status only.
- The client's submit button is disabled after the first click. This is a UX nicety only; correctness does not depend on it.

### 8.2 STOP transaction

The same locking and validation as PROCEED. Then:

- Insert a `STOP` decision.
- Set `approval_requests.status='STOPPED'`.
- `transition(p, 'STOPPED')`.
- Enqueue `project_stopped:<ar.id>`.
- Write the audit log.

No job is created. Sample, analysis artifacts and storage objects are kept. The system never deletes project data automatically.

### 8.3 Decision matrix for repeated or conflicting requests

| Request status | PROCEED POST | STOP POST |
|---|---|---|
| `PENDING` (not expired) | Approve + queue job | Stop |
| `APPROVED` | "Already approved — production is <status>" (200) | 409 "Already approved; production cannot be stopped from this link" |
| `STOPPED` | 409 "Project was stopped on <date>" | "Already stopped" (200) |
| `EXPIRED` | 410 "Request expired; contact the project owner for a new link" | 410 |
| `SUPERSEDED` | 404 generic ("A newer link was sent") | 404 |

Cancelling a production run that is already in progress is out of scope for email links. It is an admin CLI operation (§10).

### 8.4 Expiry

- The sweeper runs every 5 minutes. For each request in `PENDING` with `expires_at < now()`, it sets the request to `EXPIRED`, moves the project to `APPROVAL_EXPIRED`, enqueues `approval_expired:<ar.id>`, and writes the audit log.
- The decision handler also checks `expires_at` itself, so expiry is enforced exactly even between sweeps.
- If `reminder_hours_before_expiry > 0`, the sweeper sends one reminder per request, with dedupe key `approval_reminder:<ar.id>`.

---

## 9. Production Resume and Asset Reuse

### 9.1 Sample snapshot

Every stage writes `stage_manifest.json` with:

- `stage`, `version` (code version or git SHA)
- `params_hash`
- `inputs[]` as `{path, sha256}`
- `outputs[]` as `{path, sha256, object_key}`
- `completed_at`

The sample snapshot (`sample_manifest.json`) aggregates the analysis stage manifests together with the approved creative assets:

- character references and the character database
- voice assignments and voice model or embedding IDs
- environment and style references
- prompts and seeds
- speaker↔character mapping
- the translation glossary

Everything is uploaded to object storage under the sample version, so a fresh worker with an empty disk can resume.

### 9.2 Worker behavior

1. Lease a job:
   ```sql
   UPDATE production_jobs
   SET status = 'RUNNING', lease_owner = $w, lease_expires_at = now() + interval '10 min',
       attempt = attempt + 1, started_at = coalesce(started_at, now())
   WHERE id = (SELECT id FROM production_jobs
               WHERE status = 'QUEUED'
               ORDER BY created_at
               FOR UPDATE SKIP LOCKED
               LIMIT 1)
   RETURNING *;
   ```
   On the first lease, move the project `PRODUCTION_QUEUED → PRODUCTION_RUNNING` and enqueue `production_started`.
2. Check the authorization again: `approval_requests.status='APPROVED'`, plus a `PROCEED` decision row. If either is missing, abort.
3. Download the snapshot and verify every SHA-256 against `sample_manifest_hash`. On any mismatch, fail the job. The worker never silently regenerates approved assets.
4. For each production stage, the stage is skipped when:
   - a stage manifest exists locally or in storage,
   - its `inputs` hashes and `params_hash` match, and
   - its code version is compatible.

   Analysis stages are always taken from the snapshot and never re-run during production: scene detection, transcription, diarization, face and character analysis, and voice generation for sample-covered lines. If an analysis input differs from the snapshot, which only happens when the source video or params changed, the job fails with `ANALYSIS_INPUT_CHANGED`. In that case the user needs a new sample and a new approval. Production does not re-analyze quietly.
5. Send a heartbeat every 60 seconds to extend the lease, and update `progress_percent` and `current_stage`. When progress crosses a configured milestone, enqueue `production_progress:<job>:<pct>`.
6. Checkpoint per stage or per segment to object storage, so a restarted attempt resumes mid-production.
7. On success:
   - Upload the final video.
   - Set `status='COMPLETED'` and `output_object_key`.
   - Move the project to `PRODUCTION_COMPLETED`.
   - Enqueue `production_completed`. The email carries a link to `/review/{review_token}`, which now shows the final video, rather than an expiring raw presigned URL.
8. On failure:
   - A retryable failure requeues the job until `max_attempts` is reached, with backoff.
   - If attempts run out, set `FAILED`, move the project to `FAILED`, and enqueue `production_failed` with a sanitized error summary (no stack traces, paths or secrets).
9. If a lease expires, the sweeper requeues the job (`RUNNING → QUEUED`, project `PRODUCTION_RUNNING → PRODUCTION_QUEUED`). The job's ID and its approval stay the same, so no new job is created.

### 9.3 GPU lifecycle

- The analysis and sample run ends when run.py exits. With `--shutdown-on-complete`, or when the launcher manages the instance, the GPU instance is stopped. Waiting for an email reply never keeps a GPU running.
- The **GPU launcher** runs every minute. If `QUEUED` jobs exist and no worker heartbeat is newer than 2 minutes, it starts a GPU instance through the **Vast.ai API** using your **personal** `VAST_API_KEY` (behind a `ComputeProvider` interface, so other providers can be added later), with `python run.py --mode worker`. Before launching, it checks `max_instances` (default 1), and any spend caps you have set. Caps are `null` (open) by default; if you set `max_price_per_hour_usd` or `max_total_spend_usd`, it refuses to launch and emails you when a cap would be exceeded. With `compute.launcher: manual`, you start the instance yourself from the Vast console using the published image, and the worker drains the queue.
- A worker exits, and its instance stops, after `worker_idle_shutdown_minutes` (default 10) with no leasable job.

### 9.4 Sample revision after STOP

An admin can run the sample stage again (`--mode sample --revise`). That run produces a new `samples` row with `version+1` and a new manifest hash, moving the project `STOPPED → SAMPLE_GENERATING → SAMPLE_COMPLETED → APPROVAL_PENDING`, and sends a new email. Earlier requests stay in the audit trail.

---

## 10. CLI

| Command | Behavior |
|---|---|
| `run.py --project <cfg> --mode analyze-and-sample [--email X] [--shutdown-on-complete]` | The normal entry point. Runs §5 end to end, sends the email automatically, and exits. |
| `run.py --project <cfg> --mode worker` | Production worker loop (§9.2). |
| `run.py --project <cfg> --send-approval-email` | If the project is in `SAMPLE_COMPLETED` with no pending request, creates the request and sends the email. If a request is already `PENDING`, re-sends the **same** request without issuing a new token. Because only hashes are stored, this uses the encrypted rendered body held in the outbox (§7.4). The expiry does not change. |
| `run.py --project <cfg> --resend-approval [--email X]` | Requires `allow_resend: true`. Allowed only from `APPROVAL_PENDING` or `APPROVAL_EXPIRED`. Marks the current request `SUPERSEDED`, so its links stop working. Then creates a new request with fresh tokens and a new expiry and sends a new email. `--email` changes the recipient. |
| `run.py --project <cfg> --approval-status [--json]` | Prints project state, the active request (status, recipient masked as `u***@example.com`, created/expires/decided timestamps), the last delivery status of each email, the job status and progress, and the last 20 audit events. |
| `run.py --project <cfg> --request-new-approval` | From `STOPPED`, for a revised sample (§9.4). |
| `run.py --project <cfg> --cancel-production` | Admin only. Cancels a job in `QUEUED` or `RUNNING`; the worker stops at the next checkpoint. The action is audited. |
| `run.py --project <cfg> --retry` | From `FAILED`, retries the failed phase. |

All admin commands go through the approval service's `/admin` API using `APPROVAL_ADMIN_TOKEN`. None of them is needed in the normal workflow.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Configuration error |
| 3 | Illegal state for the command |
| 4 | Service unreachable |
| 5 | Pipeline stage failure |

---

## 11. Notifications

### 11.1 Events

| Event | Trigger | Dedupe key | Content highlights |
|---|---|---|---|
| `sample_ready` | Approval request created | `sample_ready:<ar>` | Full template (§11.2) |
| `approval_reminder` | `expires_at - reminder_hours` | `approval_reminder:<ar>` | Short message with the same links |
| `approval_received` | PROCEED recorded | `approval_received:<ar>` | Decision time; production will start when GPU capacity is available |
| `production_queued` | Job created | `production_queued:<job>` | Queue position (best effort) |
| `production_started` | First lease | `production_started:<job>` | Start time, ETA from the estimate |
| `production_progress` | Milestone crossed (when enabled) | `production_progress:<job>:<pct>` | Percent, current stage |
| `production_completed` | Job completed | `production_completed:<job>` | Final video link (review page), duration, actual GPU hours and cost against the estimate |
| `production_failed` | Job failed permanently | `production_failed:<job>` | Sanitized reason; next steps |
| `approval_expired` | Expiry | `approval_expired:<ar>` | Production not started; how to request a new link |
| `project_stopped` | STOP recorded | `project_stopped:<ar>` | Confirmation; artifacts retained |

`approval_received` and `production_queued` can be merged into a single email, controlled by the `merge_ack_emails: true` setting (default `true`) to reduce noise.

### 11.2 Sample-ready template

The template is Jinja2 with autoescape, rendered in both HTML and plain text. It lives at `templates/email/sample_ready.{html,txt}.j2`.

**Subject:** `AI Video Production — Sample Trailer Ready for Review — {{project_name}}`

**Body:** the text from the requirement, verbatim, plus these sections:

- Project, source video (basename only), sample duration (`mm:ss`), target language.
- Characters included (`character_count`) and speaking characters covered (`speaking_character_count`).
- **Important characters included:** a list of names or labels.
- **Missing or uncertain characters:** a list with the reason for each (`not in sample`, `low match confidence 0.42`).
- Quality assessment (`quality_summary`), plus **Known issues**: up to 5, ordered by severity, with a "+N more on the review page" overflow.
- Estimated cost (`estimated_cost`, formatted with currency and range) and GPU hours (`estimated_gpu_hours`).
- Expiry, shown as an absolute timestamp in UTC and the recipient's configured timezone, plus `expiration_hours`.
- Thumbnail: an inline CID image (`email_thumbnail_key`, ≤150 KB) that links to the review page. No video is attached.
- Buttons, built as bulletproof table-based HTML:
  - **WATCH SAMPLE TRAILER** → `/review/{review_token}`
  - **PROCEED WITH FULL PRODUCTION** → `/approval/{approval_token}/proceed`
  - **STOP THIS PROJECT** → `/approval/{approval_token}/stop`
- Footer: "Clicking a button opens a confirmation page; nothing happens until you confirm." Also: "If you did not expect this email, ignore it; the request expires automatically."

The plain-text version contains the same information with full URLs.

### 11.3 Delivery reliability

- **Outbox pattern.** Every email is inserted in the same transaction as the state change that causes it. Nothing is sent inline from a request handler.
- **Dispatcher loop:**
  1. Claim a batch with `FOR UPDATE SKIP LOCKED` where `status='PENDING' AND next_attempt_at <= now()`, and set it to `SENDING`.
  2. Send.
  3. On success, mark the row `SENT` and store `provider_message_id`.
  4. On a retryable error, use exponential backoff: 1 min, 5, 15, 60, 180, 360. After 8 attempts, mark the row `FAILED` and raise an alert.
  5. On a permanent error (bad address, 5xx SMTP reject), mark the row `FAILED` immediately and write the audit log.
- A `SENDING` row older than 10 minutes is reset to `PENDING`, which covers a dispatcher crash. The provider idempotency key is the outbox ID, passed as `Message-ID` for SMTP and SES and as a custom arg for SendGrid. That keeps duplicates minimal.
- SPF, DKIM and DMARC for the sender domain are a deployment prerequisite (§14).
- `--approval-status` shows the delivery state. If `sample_ready` ends up `FAILED`, the project stays in `APPROVAL_PENDING`. The operator can fix the address and run `--resend-approval --email`.

---

## 12. Audit Logging

Every event below is written to `audit_log` in the same transaction as the change it describes:

- `project_created`
- `state_transition` (from, to, actor)
- `transition_rejected`
- `sample_registered` (manifest hash)
- `approval_request_created` (request ID, recipient masked, expires_at)
- `approval_request_superseded`
- `approval_page_viewed` (action, IP, UA; sampled at one per minute per token)
- `otp_sent`, `otp_failed`
- `decision_recorded` (decision, request ID, IP, UA, auth method)
- `decision_rejected` (reason: expired, already_decided, csrf_invalid, state_invalid)
- `production_job_created`, `job_leased`, `job_lease_expired`, `job_completed`, `job_failed`, `job_cancelled`
- `email_queued`, `email_sent`, `email_failed`
- `admin_command` (command, caller)

Rules:

- The application DB role can only `INSERT` into the audit log, never `UPDATE` or `DELETE` it.
- Audit records never contain tokens, token hashes, secrets or full rendered email bodies.
- Retention is at least 1 year (configurable).

---

## 13. Prerequisite Refactor (minimal, for this feature)

1. Add `run.py` with modes `analyze-and-sample`, `sample`, and `worker`, plus the admin flags.
2. Move to a per-project workspace at `projects/<id>/workspace/{audio,transcripts,diarization,scenes,faces,characters,timeline,verification,sample,production}`. Today's scripts use hardcoded CWD-relative paths. Each script gets a `run(ctx: StageContext)` entry point in which paths come from `ctx`. The existing `__main__` blocks stay as thin wrappers for backward compatibility.
3. Add `StageRunner`, which computes input and params hashes, skips a stage when its manifest matches, writes `stage_manifest.json`, and records GPU seconds per stage for the cost estimate.
4. Add a `platform/` package containing `config.py` (loader with secret rules), `state_machine.py`, `storage.py`, `notifications/`, `approval_service/` (FastAPI app), `worker.py`, and `db/` (SQLAlchemy models and Alembic).

---

## 14. Deployment

All components are open source and come from **public** sources only: pypi.org, docker.io, ghcr.io, nvcr.io, huggingface.co. On a development machine with a different global package index, the project's `pip.conf` / `uv.toml` override it for this project only.

**Production: a personal Linux VPS** (about 2 vCPU / 4 GB / 100–200 GB disk), with one `docker compose` stack:

| Service | Image (public) | Exposure |
|---|---|---|
| Caddy | `docker.io/library/caddy` | :443 public. Automatic Let's Encrypt TLS for `approve.<domain>` (service) and `media.<domain>` (MinIO S3 API). A free dynamic-DNS hostname works if you don't own a domain. |
| Approval service + dispatcher + sweeper + launcher | built from this repo, pushed to your **personal** GHCR / Docker Hub | behind Caddy |
| PostgreSQL 16 | `docker.io/library/postgres` | internal network only, **never public** |
| MinIO | `quay.io/minio/minio` or `docker.io/minio/minio` | S3 API behind Caddy. Console bound to localhost (reach it via SSH tunnel). |

- **Backups:** nightly `pg_dump`, plus `mc mirror` of approved and final artifacts to a second disk or location you control.
- **Secrets:** a `.env` file on the VPS (mode 600). It is never committed and never copied to the Mac.
- **Environment variables:**
  - `DATABASE_URL`
  - `TOKEN_HASH_PEPPER` (32 random bytes)
  - `CSRF_SIGNING_KEY`
  - `APPROVAL_ADMIN_TOKEN`
  - `APPROVAL_PUBLIC_BASE_URL`
  - `STORAGE_ENDPOINT` and MinIO keys
  - `SMTP_*`
  - `VAST_API_KEY`
- **Email:**
  - SMTP with a **personal** mailbox app password (not a work account), or a self-hosted MTA if you own a domain. With a personal mailbox, the sender should be that mailbox address, so the message is not marked as spam.
  - SPF, DKIM and DMARC apply only if you send from your own domain.
- **Media links** are presigned MinIO URLs on `media.<domain>`, with a short TTL, generated per page view.

**Development: your own machine:**

- The same compose file with `ENV=dev`, plus **Mailpit** (`docker.io/axllent/mailpit`) to capture emails. All ports bind to **`127.0.0.1` only**, and nothing is exposed to the network.
- `http://localhost` is allowed only when `ENV=dev`.
- Docker images are pulled only from the public registries listed above.
- If you don't want Docker on the Mac, tests can use `provider: file` (email) and `provider: local` (storage) with a throwaway Postgres on the VPS.

---

## 15. Testing Strategy

| Area | Tests |
|---|---|
| State machine | Table-driven tests covering every allowed transition and a representative set of illegal ones. Production cannot start from `APPROVAL_PENDING`, `STOPPED` or `APPROVAL_EXPIRED`, checked through both the code path and the DB trigger. |
| Link scanners | `GET` each of `/review`, `/approval/*/proceed` and `/approval/*/stop` 100 times (also with HEAD, OPTIONS, and the user agents Outlook SafeLinks, Mimecast and Slackbot). Then assert zero decisions, zero jobs and unchanged state. A `POST` without CSRF, with an expired CSRF token, or with CSRF minted for another token is rejected. |
| Concurrency | 50 parallel PROCEED `POST`s with valid CSRF produce exactly one decision and one job, and every response is either a 303 to the result page or a "already approved" page. A PROCEED racing a STOP produces exactly one decision. |
| Tokens | Unknown, expired, superseded and cross-project tokens all return the same generic 404 or 410. Using project A's CSRF with project B's token fails. Token plaintext never appears in the DB, logs or audit log (grep the test DB dump). |
| Expiry | Freeze time past `expires_at` and confirm both the lazy rejection and the sweeper transition. A resend from `APPROVAL_EXPIRED` works, and the old links stay dead. |
| Restarts | Kill the service between COMMIT and the HTTP response, and confirm the state persisted. Kill the worker mid-job, and confirm the lease expires, the job is requeued with the same job ID, it resumes from a checkpoint, and no second job exists. Restart the dispatcher with rows in `SENDING`, and confirm they are recovered. |
| No-GPU approval | With no worker running, PROCEED succeeds, the job is `QUEUED`, and the launcher is invoked (with a mocked ComputeProvider). |
| Asset reuse | Production run on a fixture project. Assert that no analysis stage executed, by checking the stage-runner skip counters and that analysis function call counts are 0. Assert that voice and character assets match the snapshot hashes. Tamper with one asset, and the job fails with `SNAPSHOT_MISMATCH`. |
| Email | Use `FileProvider` to assert the subject, all required fields, all three links, the inline thumbnail, no video attachment, and that the HTML and text versions match. Provider retry classification. Dedupe: the same event enqueued twice produces one email. |
| Config | Literal `password:` → startup error. Missing `${SMTP_HOST}` → startup error. `--email` override is used. An invalid email is rejected before analysis starts. |
| Security headers / HTTPS | Header assertions. A non-https `public_base_url` outside dev fails to start. |
| Rate limit | Over the threshold, the response is 429, and valid users from other IPs are unaffected. |
| E2E | The docker-compose stack with Mailpit and MinIO. Run `analyze-and-sample` on a 2-minute fixture, read the email from the Mailpit API, follow the PROCEED link, submit the form, run the worker, and assert a completion email whose final video link returns 200 over the review page. No terminal input happens after the first command. |

---

## 16. Acceptance Criteria Traceability

| Requirement | Satisfied by | Verified by |
|---|---|---|
| Email recipient supplied at project initialization | §3.1, §3.3 | Config tests |
| Sample trailer generated automatically | §5 steps 1–5 | E2E |
| Working review link emailed | §5, §7.3, §11.2 | E2E (Mailpit → GET review returns 200 with a playable presigned URL) |
| Proceed and Stop actions included | §11.2 | Email tests |
| Email link scanners cannot authorize production | §7.4, read-only GETs, CSRF POST | Link-scanner tests |
| Proceed starts production with no terminal interaction | §8.1, §9.2–9.3 | E2E, No-GPU test |
| Stop prevents production | §8.2, §4.3 trigger | State-machine tests, STOP E2E variant |
| Repeated clicks cannot create duplicate jobs | §8.1 locks + unique indexes | Concurrency tests |
| Approval survives service and GPU worker restarts | Postgres source of truth, outbox, leases | Restart tests |
| No GPU kept running while waiting | §5 step 11, §9.3 | No-GPU test, launcher idle-shutdown test |
| Final production uses approved sample assets | §9.1–9.2 snapshot + hash verification | Asset-reuse tests |
| All decisions auditable | §12 | Audit assertions across every test above |

---

## 17. Open Questions

1. **Hosting** (deferred to M5): no VPS or domain yet. Plan: a personal VPS plus a free dynamic-DNS hostname, chosen at M5. GPU compute is decided: personal Vast.ai.
2. **Recipients.** Is one approval recipient per project enough, or are multiple recipients needed (any one can decide, or all must approve)? The current design assumes one.
3. **Authentication default.** Should `require_authentication` default to `true` when the estimated cost exceeds a threshold, such as $100? The OTP adds friction but protects against forwarded emails.
4. **Final-video retention.** How long should final-video links stay valid? The proposal is the life of the review token, renewable through `--resend-approval`-style admin commands.
5. **Cancellation from email.** Is a STOP for production that is already running needed in emails, or is CLI-only cancellation acceptable?

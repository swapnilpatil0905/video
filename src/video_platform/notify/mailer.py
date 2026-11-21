"""Email notifications (stdlib only).

Two transports, chosen automatically:
  1. SMTP with your own account (best: attachments, reliable delivery) when SMTP_* is set.
  2. ntfy.sh e-mail forwarding (no credentials) when only NOTIFY_EMAIL is set. Text only,
     sender is ntfy.sh, the public server rate-limits e-mails (fine for a few notifications).

Configured entirely from the environment / .env (never from config files):
    NOTIFY_EMAIL    recipient
    NTFY_TOPIC      optional random topic name for the ntfy fallback (default: derived, hard to guess)
    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       587 (STARTTLS) or 465 (SSL)
    SMTP_USERNAME   e.g. your Gmail address
    SMTP_PASSWORD   an app password (Gmail: myaccount.google.com/apppasswords)
    SMTP_FROM       optional sender (defaults to SMTP_USERNAME)
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import smtplib
import socket
import ssl
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path

from ..core.log import get_logger

log = get_logger(__name__)

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # Gmail limit is 25 MB including encoding overhead


class MailNotConfigured(RuntimeError):
    pass


def smtp_settings(env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    missing = [k for k in ("NOTIFY_EMAIL", "SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD") if not env.get(k)]
    if missing:
        raise MailNotConfigured(f"email not configured (missing {', '.join(missing)} in .env)")
    return {
        "to": env["NOTIFY_EMAIL"], "host": env["SMTP_HOST"], "port": env.get("SMTP_PORT", "587"),
        "user": env["SMTP_USERNAME"], "password": env["SMTP_PASSWORD"], "from": env.get("SMTP_FROM") or env["SMTP_USERNAME"],
    }


def build_message(settings: dict[str, str], subject: str, body: str, attachments: list[Path]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["from"]
    msg["To"] = settings["to"]
    msg.set_content(body)
    total = 0
    for path in attachments:
        path = Path(path)
        if not path.exists():
            continue
        size = path.stat().st_size
        if total + size > MAX_ATTACHMENT_BYTES:
            log.warning("skipping attachment %s (%.1f MB): email size limit", path.name, size / 1e6)
            continue
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)
        total += size
    return msg


def ntfy_topic(env: dict[str, str] | None = None) -> str:
    env = dict(os.environ if env is None else env)
    if env.get("NTFY_TOPIC"):
        return env["NTFY_TOPIC"]
    seed = f"{env.get('NOTIFY_EMAIL', '')}|{socket.gethostname()}|video-platform"
    return "vp-" + hashlib.sha256(seed.encode()).hexdigest()[:24]


def send_via_ntfy(subject: str, body: str, *, retries: int = 3) -> None:
    """Credential-free e-mail via ntfy.sh forwarding (text only)."""
    to = os.environ.get("NOTIFY_EMAIL")
    if not to:
        raise MailNotConfigured("email not configured (set NOTIFY_EMAIL in .env)")
    text = body if len(body.encode()) < 3800 else body.encode()[:3800].decode("utf-8", "ignore") + "\n…(truncated)"
    request = urllib.request.Request(
        f"https://ntfy.sh/{ntfy_topic()}", data=text.encode("utf-8"), method="POST",
        headers={"Title": subject.encode("ascii", "ignore").decode() or "AI video", "Email": to, "Tags": "movie_camera"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https host
                json.loads(response.read() or b"{}")
            log.info("notification sent via ntfy.sh e-mail forwarding to %s", to)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ntfy.sh forwarding failed: {last}")


def send(subject: str, body: str, attachments: list[Path] | None = None, *, retries: int = 3) -> None:
    try:
        settings = smtp_settings()
    except MailNotConfigured:
        if os.environ.get("NOTIFY_EMAIL"):
            note = "\n\n(Sent without attachments via ntfy.sh. Add a Gmail app password to .env for previews.)"
            send_via_ntfy(subject, body + (note if attachments else ""), retries=retries)
            return
        raise
    msg = build_message(settings, subject, body, attachments or [])
    last: Exception | None = None
    for attempt in range(retries):
        try:
            port = int(settings["port"])
            context = ssl.create_default_context()
            if port == 465:
                with smtplib.SMTP_SSL(settings["host"], port, context=context, timeout=60) as smtp:
                    smtp.login(settings["user"], settings["password"])
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(settings["host"], port, timeout=60) as smtp:
                    smtp.starttls(context=context)
                    smtp.login(settings["user"], settings["password"])
                    smtp.send_message(msg)
            log.info("email sent to %s: %s", settings["to"], subject)
            return
        except (smtplib.SMTPException, OSError) as exc:
            last = exc
            log.warning("email attempt %d failed: %s", attempt + 1, exc)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not send email after {retries} attempts: {last}")

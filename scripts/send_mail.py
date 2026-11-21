#!/usr/bin/env python3
"""Send a plain notification email using only the standard library (works before any setup).

    python3 scripts/send_mail.py --subject "..." --body "..." [--log /workspace/run.log] [--attach file ...]
SMTP settings come from the environment / .env (NOTIFY_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_platform.notify.mailer import MailNotConfigured, send  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--body", default="")
    p.add_argument("--log", help="append the last 40 lines of this file")
    p.add_argument("--attach", nargs="*", default=[])
    a = p.parse_args()
    body = a.body
    if a.log and Path(a.log).exists():
        tail = Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        body += "\n\nLast log lines:\n" + "\n".join(tail)
    try:
        send(a.subject, body, [Path(x) for x in a.attach])
    except MailNotConfigured as exc:
        print(f"[mail] skipped: {exc}")
        return 0
    except Exception as exc:  # report, never break the pipeline
        print(f"[mail] failed: {exc}")
        return 1
    print("[mail] sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import pytest

from video_platform.notify.mailer import MAX_ATTACHMENT_BYTES, MailNotConfigured, build_message, smtp_settings

ENV = {"NOTIFY_EMAIL": "me@example.com", "SMTP_HOST": "smtp.example.com", "SMTP_USERNAME": "me@example.com",
       "SMTP_PASSWORD": "app-password"}


def test_missing_settings_are_reported():
    with pytest.raises(MailNotConfigured, match="SMTP_PASSWORD"):
        smtp_settings({k: v for k, v in ENV.items() if k != "SMTP_PASSWORD"})


def test_message_with_attachments_respects_size_limit(tmp_path):
    small = tmp_path / "thumb.jpg"
    small.write_bytes(b"x" * 1000)
    big = tmp_path / "huge.mp4"
    big.write_bytes(b"0" * (MAX_ATTACHMENT_BYTES + 1))
    msg = build_message(smtp_settings(ENV), "subject", "body", [small, big, tmp_path / "missing.png"])
    names = [part.get_filename() for part in msg.iter_attachments()]
    assert names == ["thumb.jpg"]
    assert msg["To"] == "me@example.com" and msg["From"] == "me@example.com"

from pathlib import Path

from video_platform.runtime.launcher import _fmt_elapsed, last_log_line
from video_platform.workers.common import Progress


def test_last_log_line_handles_tqdm(tmp_path: Path):
    log = tmp_path / "w.log"
    log.write_text("$ (cwd=None) python -m x\nloading\rProcessed prompts:  10%\rProcessed prompts:  45%\x1b[0m\n\n")
    assert last_log_line(log) == "Processed prompts:  45%"
    assert last_log_line(tmp_path / "missing.log") == ""


def test_fmt_elapsed():
    assert _fmt_elapsed(75) == "1m15s"
    assert _fmt_elapsed(3725) == "1h02m"


def test_progress_prints_first_and_last(capsys):
    p = Progress("tts", 3, every=3600)
    for _ in range(3):
        p.step()
    err = capsys.readouterr().err.splitlines()
    assert err[0] == "[tts] starting 3 items"
    assert err[1].startswith("[tts] item 1/3 (0% done)")
    assert err[2].startswith("[tts] item 3/3 (67% done)") and "eta" in err[2]
    assert len(err) == 3

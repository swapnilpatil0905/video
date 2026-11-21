from video_platform.core.network_policy import assert_allowed, host_of, is_forbidden_host, preflight
from video_platform.core.errors import PolicyViolation

import pytest

PAT = ("corp.example", "*.corp.example")


@pytest.mark.parametrize("value,host", [
    ("https://mirror.corp.example/pypi/simple", "mirror.corp.example"),
    ("registry.corp.example", "registry.corp.example"),
    ("ghcr.io", "ghcr.io"),
    ("https://pypi.org/simple", "pypi.org"),
])
def test_host_of(value, host):
    assert host_of(value) == host


def test_forbidden_patterns():
    assert is_forbidden_host("mirror.corp.example", PAT)
    assert is_forbidden_host("corp.example", PAT)
    assert not is_forbidden_host("pypi.org", PAT)
    assert not is_forbidden_host("notcorp.example", PAT)
    assert not is_forbidden_host("mirror.corp.example")  # nothing is forbidden by default
    with pytest.raises(PolicyViolation):
        assert_allowed("https://registry.corp.example/v2/", PAT)


def test_preflight_flags_forbidden_env_index():
    env = {"PIP_INDEX_URL": "https://mirror.corp.example/simple", "UV_INDEX_URL": "https://pypi.org/simple"}
    findings = {f.check: f.status for f in preflight(patterns=PAT, env=env, check_pip=False)}
    assert findings["env:PIP_INDEX_URL"] == "fail"
    assert findings["env:UV_INDEX_URL"] == "ok"


def test_preflight_configured_urls():
    findings = preflight(env={}, check_pip=False, configured_urls=[("storage", "https://media.example.org")])
    assert [f.status for f in findings] == ["ok"]

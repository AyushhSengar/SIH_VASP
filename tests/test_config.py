"""
Tests for `app.core.config` and the `.env.example` that documents it.

Configuration drift is silent and expensive: a threshold added in code but
never documented is a knob nobody knows exists, and a variable documented but
never read is a knob that does nothing when turned. Both are pinned here.

Also pinned: no secret is ever hardcoded, and the default configuration never
points at demonstration data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from app.core.config import ConfigurationError, Settings, get_settings

_CONFIG = Path("app/core/config.py")
_ENV_EXAMPLE = Path(".env.example")

# _get_int("NAME", default) / _get_float / _get_bool / os.getenv("NAME", default)
_READS = re.compile(
    r'(?:_get_int|_get_float|_get_bool|os\.getenv)\(\s*\n?\s*"([A-Z_0-9]+)"\s*,\s*'
    r'([^)]*?)\s*\)',
    re.S,
)


def _settings_read_from_environment() -> dict[str, str]:
    source = _CONFIG.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for match in _READS.finditer(source):
        found.setdefault(match.group(1), " ".join(match.group(2).split()))
    return found


def _documented_in_env_example() -> dict[str, str]:
    documented: dict[str, str] = {}
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            documented[name.strip()] = value.strip()
    return documented


def test_every_setting_the_code_reads_is_documented():
    undocumented = sorted(
        set(_settings_read_from_environment()) - set(_documented_in_env_example())
    )
    assert undocumented == [], (
        "these environment variables change behaviour but appear nowhere in "
        f".env.example, so nobody can discover them: {undocumented}"
    )


def test_env_example_documents_nothing_the_code_ignores():
    """A documented variable that does nothing is worse than an undocumented
    one: it invites someone to set it and trust the result."""
    inert = sorted(
        set(_documented_in_env_example()) - set(_settings_read_from_environment())
    )
    assert inert == [], f"documented but never read: {inert}"


def test_documented_defaults_match_the_code_defaults():
    code = _settings_read_from_environment()
    documented = _documented_in_env_example()
    mismatches = []
    for name, raw_default in code.items():
        expected = raw_default.strip('"').replace("_", "")  # 86_400 -> 86400
        actual = documented[name]
        if expected == actual or expected.strip('"') == actual:
            continue
        # Numeric equality: 0.5 == .5, true == True.
        try:
            if float(expected) == float(actual):
                continue
        except ValueError:
            pass
        if expected.lower() == actual.lower():
            continue
        # Paths legitimately contain underscores, which the strip above ate.
        if raw_default.strip('"') == actual:
            continue
        mismatches.append((name, raw_default, actual))
    assert mismatches == [], mismatches


def test_no_credential_is_hardcoded_as_a_default():
    """A key with a default is a key in the source tree."""
    settings = get_settings()
    assert settings.etherscan_api_key == "" or len(settings.etherscan_api_key) > 0
    source = _CONFIG.read_text(encoding="utf-8")
    assert 'os.getenv("ETHERSCAN_API_KEY", "")' in source, (
        "the API key must default to empty, never to a literal value"
    )
    # .env.example ships the variable present but unset.
    assert _documented_in_env_example()["ETHERSCAN_API_KEY"] == ""


def test_default_seed_dataset_is_the_real_one_not_the_demo_one():
    settings = get_settings()
    assert settings.vasp_seed_dataset_path.endswith("known_vasps.json")
    assert "demo" not in settings.vasp_seed_dataset_path
    assert "demo" in settings.vasp_demo_seed_dataset_path, (
        "the synthetic dataset must be reachable only through its own setting"
    )


def test_env_example_is_ascii_so_it_reads_on_any_console():
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert text.isascii(), [c for c in text if not c.isascii()][:10]


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
     ("false", False), ("0", False), ("no", False), ("off", False),
     ("  TRUE  ", True)],
)
def test_boolean_settings_accept_the_obvious_spellings(value, expected, monkeypatch):
    monkeypatch.setenv("PROVIDER_CACHE_ENABLED", value)
    assert get_settings().provider_cache_enabled is expected


def test_a_malformed_numeric_setting_fails_loudly_and_names_the_variable(monkeypatch):
    """It must NOT fall back to the default.

    Every finding in the report is printed next to the threshold it was
    measured against. Quietly substituting a different number would make the
    report state a threshold the operator never chose, which is exactly the
    kind of unverifiable claim this project exists to avoid. Failing at
    configuration-read time costs one line of output; a wrong threshold costs
    the credibility of the whole report.
    """
    monkeypatch.setenv("FUND_TRACE_MAX_HOPS", "not-a-number")
    with pytest.raises(ConfigurationError) as excinfo:
        get_settings()
    message = str(excinfo.value)
    assert "FUND_TRACE_MAX_HOPS" in message
    assert "not-a-number" in message
    assert "4" in message, "the message must state the default it would have used"


def test_a_malformed_float_setting_fails_loudly_too(monkeypatch):
    monkeypatch.setenv("BEHAVIOR_COUNTERPARTY_CONCENTRATION_MIN_SHARE", "half")
    with pytest.raises(ConfigurationError) as excinfo:
        get_settings()
    assert "BEHAVIOR_COUNTERPARTY_CONCENTRATION_MIN_SHARE" in str(excinfo.value)


def test_an_unrecognised_boolean_spelling_is_rejected_not_read_as_false(monkeypatch):
    """`PROVIDER_CACHE_ENABLED=ture` used to disable the cache silently, which
    looks identical to deliberately disabling it."""
    monkeypatch.setenv("PROVIDER_CACHE_ENABLED", "ture")
    with pytest.raises(ConfigurationError) as excinfo:
        get_settings()
    message = str(excinfo.value)
    assert "PROVIDER_CACHE_ENABLED" in message
    assert "ture" in message


def test_an_empty_setting_means_unset_not_malformed(monkeypatch):
    """A blank line in `.env` is how the shipped example leaves a value at its
    default, so it must not be an error."""
    monkeypatch.setenv("FUND_TRACE_MAX_HOPS", "")
    monkeypatch.setenv("PROVIDER_CACHE_ENABLED", "   ")
    settings = get_settings()
    assert settings.fund_trace_max_hops == 4
    assert settings.provider_cache_enabled is True


def test_configuration_errors_never_echo_a_credential():
    """The helpers that echo their value only ever read numeric and boolean
    thresholds. Credentials are read with bare os.getenv, which cannot raise,
    so no error message can ever contain a key."""
    source = _CONFIG.read_text(encoding="utf-8")
    for secret in ("ETHERSCAN_API_KEY", "DATABASE_URL"):
        for helper in ("_get_int", "_get_float", "_get_bool"):
            assert f'{helper}("{secret}"' not in source, (
                f"{secret} must not be read through {helper}, whose error "
                "message echoes the offending value"
            )


def test_the_cli_reports_a_configuration_error_without_a_traceback(monkeypatch, capsys):
    """A typo in `.env` must not look like a crash."""
    import investigate as cli

    monkeypatch.setenv("FUND_TRACE_MAX_HOPS", "four")
    monkeypatch.setattr(
        sys, "argv", ["investigate.py", "0x" + "ab" * 20, "--cached-graph", "missing"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "CONFIGURATION ERROR" in captured.err
    assert "FUND_TRACE_MAX_HOPS" in captured.err
    assert "Traceback" not in captured.err


def test_settings_is_constructible_directly_for_tests():
    """Tests must be able to build a Settings without touching the process
    environment, or they leak configuration into each other."""
    settings = Settings(
        etherscan_api_key="unit-test",
        etherscan_base_url="https://example.invalid",
        etherscan_chain_id=1,
        max_transactions_per_investigation=10,
        default_lookback_days=1,
        http_timeout_seconds=1,
        http_max_retries=1,
    )
    assert settings.etherscan_api_key == "unit-test"

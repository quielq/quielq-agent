from pathlib import Path

import pytest

from quielq_agent.config import ConfigError, load_agent_config

ROOT = Path(__file__).resolve().parents[1]


def test_echo_config_loads():
    config = load_agent_config(ROOT / "fleet/agents/echo.yaml")
    assert config.name == "echo"
    assert config.tools.allow == []


def test_research_config_loads():
    config = load_agent_config(ROOT / "fleet/agents/research.yaml")
    assert config.name == "research"
    assert config.tools.allow == ["web_search", "local_search"]
    assert config.memory_dir == "fleet/memory/research"


def test_ops_watch_config_loads_schedule():
    config = load_agent_config(ROOT / "fleet/agents/ops-watch.yaml")
    assert config.name == "ops-watch"
    assert config.tools.allow == ["web_search", "web_fetch", "github_repo"]
    assert len(config.schedule) == 1
    entry = config.schedule[0]
    assert entry.cron == "0 */6 * * *"
    assert entry.deliver_to == "memory"


def test_channels_and_limits_parse(tmp_path):
    full = tmp_path / "full.yaml"
    full.write_text(
        """
name: full
display_name: Full
model: m
system_prompt_file: fleet/prompts/echo.md
channels:
  web: {path: /full}
  telegram: {bot_token_env: TG_FULL, allowed_user_ids: [111, 222]}
limits:
  max_tokens_per_day: 1000
  max_usd_per_day: 0.5
schedule:
  - cron: "0 6 * * *"
    message: "go"
    deliver_to: telegram
"""
    )
    config = load_agent_config(full)
    assert config.channels.web.path == "/full"
    assert config.channels.telegram.bot_token_env == "TG_FULL"
    assert config.channels.telegram.allowed_user_ids == [111, 222]
    assert config.limits.max_tokens_per_day == 1000
    assert config.limits.max_usd_per_day == 0.5
    assert config.schedule[0].deliver_to == "telegram"


def test_channels_limits_schedule_default_empty():
    config = load_agent_config(ROOT / "fleet/agents/echo.yaml")
    assert config.channels.web is None
    assert config.channels.telegram is None
    assert config.limits.max_tokens_per_day is None
    assert config.schedule == []


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_agent_config(ROOT / "fleet/agents/does_not_exist.yaml")


def test_name_mismatch_raises(tmp_path):
    bad = tmp_path / "mismatch.yaml"
    bad.write_text(
        "name: other\ndisplay_name: Other\nmodel: x\nsystem_prompt_file: fleet/prompts/echo.md\n"
    )
    with pytest.raises(ConfigError):
        load_agent_config(bad)


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "incomplete.yaml"
    bad.write_text("name: incomplete\n")
    with pytest.raises(ConfigError):
        load_agent_config(bad)

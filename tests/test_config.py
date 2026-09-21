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
    assert config.tools.allow == ["web_search"]
    assert config.memory_dir == "fleet/memory/research"


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

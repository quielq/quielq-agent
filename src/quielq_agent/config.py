"""Loading and validating per-agent YAML configs (fleet/agents/<name>.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Raised when an agent config file is missing or invalid."""


class ToolsConfig(BaseModel):
    allow: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    name: str
    display_name: str
    model: str
    system_prompt_file: str
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory_dir: str | None = None

    @property
    def system_prompt_path(self) -> Path:
        return (PROJECT_ROOT / self.system_prompt_file).resolve()

    @property
    def memory_path(self) -> Path | None:
        if self.memory_dir is None:
            return None
        return (PROJECT_ROOT / self.memory_dir).resolve()

    def load_system_prompt(self) -> str:
        path = self.system_prompt_path
        if not path.is_file():
            raise ConfigError(f"system_prompt_file not found: {path}")
        return path.read_text().strip()


def load_agent_config(path: str | Path) -> AgentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"agent config not found: {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    try:
        config = AgentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid agent config {config_path}: {exc}") from exc

    stem = config_path.stem
    if config.name != stem:
        raise ConfigError(
            f"agent config {config_path}: 'name' ({config.name!r}) must match filename ({stem!r})"
        )

    return config

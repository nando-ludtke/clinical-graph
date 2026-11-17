"""Versioned YAML prompt template loader."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

_PROMPTS_DIR = Path(__file__).resolve().parent


class PromptTemplate(BaseModel):
    """Structured prompt template loaded from a versioned YAML file."""

    name: str
    version: str
    description: str = ""
    system_prompt: str


@lru_cache
def load_prompt(name: str, version: str | None = None) -> PromptTemplate:
    """Load and cache a ``PromptTemplate`` from ``{name}.yml``."""
    prompt_path = _PROMPTS_DIR / f"{name}.yml"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    target_version = version or data.get("default_version", "v1")
    versions = data.get("versions") or {}
    if target_version not in versions:
        raise ValueError(f"Version {target_version!r} not found in {prompt_path}")

    entry = versions[target_version]
    return PromptTemplate(
        name=data.get("name", name),
        version=target_version,
        description=entry.get("description", ""),
        system_prompt=entry["system_prompt"],
    )

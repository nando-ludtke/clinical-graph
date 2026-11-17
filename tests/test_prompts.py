"""Unit tests for versioned YAML prompt templates and cached loader."""

import pytest

from clinical_graph.prompts import PromptTemplate, load_prompt

EXPECTED_DEFAULT_VERSIONS = (
    ("node_extraction", "v3"),
    ("edge_extraction", "v3"),
    ("graph_extraction", "v1"),
    ("table_summary", "v1"),
    ("query_rewrite", "v3"),
    ("answer_synthesis", "v4"),
    ("document_classifier", "v1"),
)


@pytest.mark.parametrize(("prompt_name", "expected_default"), EXPECTED_DEFAULT_VERSIONS)
def test_load_prompt_default_and_v1_templates(
    prompt_name: str, expected_default: str
) -> None:
    template = load_prompt(prompt_name)

    assert isinstance(template, PromptTemplate)
    assert template.name == prompt_name
    assert template.version == expected_default
    assert template.description
    assert template.system_prompt.strip()
    # Subsequent call with same arguments hits lru_cache and returns the identical cached instance.
    assert load_prompt(prompt_name) is template
    assert load_prompt(prompt_name, version=expected_default) == template

    v1_template = load_prompt(prompt_name, version="v1")
    assert v1_template.version == "v1"
    assert v1_template.system_prompt.strip()


@pytest.mark.parametrize("prompt_name", ("node_extraction", "edge_extraction", "answer_synthesis"))
def test_load_prompt_v2_templates(prompt_name: str) -> None:
    v2_template = load_prompt(prompt_name, version="v2")
    assert v2_template.name == prompt_name
    assert v2_template.version == "v2"
    assert v2_template.description
    assert v2_template.system_prompt.strip()


def test_load_prompt_answer_synthesis_v3() -> None:
    v3_template = load_prompt("answer_synthesis", version="v3")
    assert v3_template.name == "answer_synthesis"
    assert v3_template.version == "v3"
    assert "SynthesizedAnswer" in v3_template.system_prompt
    assert "ref_id" in v3_template.system_prompt


def test_load_prompt_answer_synthesis_v4() -> None:
    v4_template = load_prompt("answer_synthesis", version="v4")
    assert v4_template.name == "answer_synthesis"
    assert v4_template.version == "v4"
    assert "STRICT NO-PRETRAINING RULE" in v4_template.system_prompt
    assert "STRICTLY VERBATIM" in v4_template.system_prompt
    assert "NEVER cite graph edge tags" in v4_template.system_prompt
    assert '"citations": []' in v4_template.system_prompt


def test_load_prompt_missing_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_prompt", version="v1")


def test_load_prompt_unknown_version_raises_value_error() -> None:
    with pytest.raises(ValueError):
        load_prompt("node_extraction", version="v999")

"""Unit tests for DocumentParser (.txt, .md, .pdf) and LiteLLMEmbedder."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clinical_graph.graph.vector_store import LiteLLMEmbedder
from clinical_graph.parser import (
    DocumentParser,
    ParsedDocument,
    ParsedTable,
    group_chunks_for_extraction,
    strip_references_and_boilerplate,
)
from clinical_graph.schemas.document import SourceType


def test_parse_txt_bytes_and_file(tmp_path: Path) -> None:
    parser = DocumentParser()
    content = "Metformin is first-line therapy for type 2 diabetes."

    parsed_bytes = parser.parse_bytes(content.encode("utf-8"), "guideline.txt")
    assert isinstance(parsed_bytes, ParsedDocument)
    assert parsed_bytes.text == content
    assert parsed_bytes.tables == []
    assert parsed_bytes.page_count is None
    assert parsed_bytes.source_type == SourceType.RAW_TEXT

    txt_file = tmp_path / "note.TXT"
    txt_file.write_text(content, encoding="utf-8")
    parsed_file = parser.parse_file(txt_file)
    assert parsed_file == parsed_bytes


def test_parse_markdown_bytes_and_file(tmp_path: Path) -> None:
    parser = DocumentParser()
    content = "# Sepsis Bundle\n\n- Administer broad-spectrum antibiotics within 1 hour.\n"

    parsed_bytes = parser.parse_bytes(content.encode("utf-8"), "sepsis.md")
    assert parsed_bytes.text == content
    assert parsed_bytes.tables == []
    assert parsed_bytes.page_count is None
    assert parsed_bytes.source_type == SourceType.MARKDOWN

    md_file = tmp_path / "sepsis.MD"
    md_file.write_text(content, encoding="utf-8")
    parsed_file = parser.parse_file(md_file)
    assert parsed_file == parsed_bytes


@pytest.mark.parametrize("filename", ["protocol.docx", "data.csv", "README", "image.png"])
def test_parse_unsupported_extension_raises_value_error(filename: str) -> None:
    parser = DocumentParser()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parser.parse_bytes(b"sample", filename)


def test_parse_pdf_with_mocked_pymupdf(tmp_path: Path) -> None:
    mock_table = MagicMock()
    mock_table.extract.return_value = [
        ["Drug", "Dose"],
        ["Metformin", "500 mg BID"],
        ["Lisinopril", "10 mg daily"],
    ]

    page_1 = MagicMock()
    page_1.get_text.return_value = "Page 1: Clinical Overview"
    page_1.find_tables.return_value = SimpleNamespace(tables=[mock_table])

    page_2 = MagicMock()
    page_2.get_text.return_value = "Page 2: Contraindications"
    page_2.find_tables.return_value = SimpleNamespace(tables=[])

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__iter__.side_effect = lambda: iter([page_1, page_2])
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = False

    mock_pymupdf = MagicMock()
    mock_pymupdf.open.return_value = mock_doc

    parser = DocumentParser()
    pdf_bytes = b"%PDF-1.4 fake pdf bytes"

    with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
        parsed = parser.parse_bytes(pdf_bytes, "guideline.pdf")

        pdf_file = tmp_path / "guideline.pdf"
        pdf_file.write_bytes(pdf_bytes)
        parsed_from_file = parser.parse_file(pdf_file)

    assert parsed.source_type == SourceType.UPLOADED_PDF
    assert parsed.page_count == 2
    expected_table = (
        "| Drug | Dose |\n"
        "| --- | --- |\n"
        "| Metformin | 500 mg BID |\n"
        "| Lisinopril | 10 mg daily |"
    )
    assert parsed.text == (
        f"Page 1: Clinical Overview\n\n{expected_table}\n\f\nPage 2: Contraindications"
    )
    assert len(parsed.tables) == 1
    assert parsed.tables[0] == expected_table
    assert parsed_from_file == parsed

    mock_pymupdf.open.assert_called_with(stream=pdf_bytes, filetype="pdf")
    assert page_1.get_text.call_count == 4
    page_1.get_text.assert_any_call("dict")
    page_1.get_text.assert_called_with("text")
    assert page_2.get_text.call_count == 4
    page_2.get_text.assert_any_call("dict")
    page_2.get_text.assert_called_with("text")


def test_litellm_embedder_returns_l2_normalized_float32_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    raw_vec = [3.0, 4.0, 0.0]
    mock_response = SimpleNamespace(data=[{"embedding": raw_vec}])
    mock_embedding = MagicMock(return_value=mock_response)
    monkeypatch.setattr(litellm, "embedding", mock_embedding)

    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    vec = embedder.embed("type 2 diabetes")

    mock_embedding.assert_called_once_with(
        model="text-embedding-3-small",
        input=["type 2 diabetes"],
        timeout=30.0,
    )
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (3,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)
    assert np.allclose(vec, np.array([0.6, 0.8, 0.0], dtype=np.float32), atol=1e-6)


def test_strip_references_preserves_appendix() -> None:
    text = (
        "# Clinical Guideline\n\n"
        "Metformin is recommended as first-line pharmacotherapy for type 2 diabetes.\n\n"
        "## Acknowledgements\n\n"
        "We thank the guideline committee for their contributions.\n\n"
        "## References\n\n"
        "[1] Smith J et al. Diabetes Care. 2023;46(1):1-10. doi:10.2337/dc23-0001\n"
        "[2] Doe A et al. Lancet. 2022;400:100-112. PMID: 12345678\n\n"
        "## Appendix A: Dosing Table\n\n"
        "| Drug | Initial Dose |\n"
        "| --- | --- |\n"
        "| Metformin | 500 mg daily |"
    )

    cleaned, omissions = strip_references_and_boilerplate(text)

    assert "Metformin is recommended" in cleaned
    assert "## Appendix A: Dosing Table" in cleaned
    assert "| Metformin | 500 mg daily |" in cleaned
    assert "Smith J et al." not in cleaned
    assert "guideline committee" not in cleaned

    assert len(omissions) == 2
    assert omissions[0].reason == "boilerplate_section"
    assert omissions[0].heading == "## Acknowledgements"
    assert omissions[1].reason == "references"
    assert omissions[1].heading == "## References"
    assert omissions[1].omission_id.startswith("omit_001_")
    assert omissions[1].char_count > 0
    assert len(omissions[1].preview) <= 120


def test_strip_citation_density_fallback_and_repeated_headers() -> None:
    watermark = "Journal of Clinical Practice | Page X of Y"
    body = "\n\n".join(
        [
            watermark,
            "# Overview\n\nAspirin reduces cardiovascular events in secondary prevention.",
            watermark,
            "Clopidogrel is an alternative P2Y12 inhibitor for aspirin-intolerant patients.",
            watermark,
            "Monitor for gastrointestinal bleeding during dual antiplatelet therapy.",
            "Proton pump inhibitors are recommended in patients with high bleeding risk.",
            "Discontinue dual antiplatelet therapy prior to elective major surgery when safe.",
            watermark,
            (
                "[1] Alpha A et al. NEJM. 2020;382:10-20. doi:10.1056/NEJMoa1\n"
                "[2] Beta B et al. JAMA. 2021;325:30-40. PMID: 23456789\n"
                "[3] Gamma C et al. Circulation. 2022;145:50-60. doi:10.1161/CIRC1\n"
                "Note: additional bibliography entry without bracket."
            ),
        ]
    )

    cleaned, omissions = strip_references_and_boilerplate(body)

    assert watermark not in cleaned
    assert "Aspirin reduces cardiovascular events" in cleaned
    assert "Alpha A et al." not in cleaned

    reasons = [item.reason for item in omissions]
    assert "repeated_header_footer" in reasons
    assert "citation_density" in reasons


def test_repeated_midpage_clinical_headings_not_stripped_pdf_and_text() -> None:
    # 1) PDF text with \f page boundaries: boundary header/footer stripped, mid-page headings kept.
    header = "NVL Typ-2-Diabetes | Kurzfassung"
    footer = "Bundesärztekammer"
    pages = [
        "\n".join(
            [
                header,
                "Kapitel 1",
                "Empfehlung",
                "Metformin",
                "Metformin ist Mittel der ersten Wahl bei Typ-2-Diabetes.",
                "Abschnitt 1 Ende",
                footer,
            ]
        ),
        "\n".join(
            [
                header,
                "Kapitel 2",
                "Empfehlung",
                "Metformin",
                "Bei Niereninsuffizienz ist die Dosis von Metformin anzupassen.",
                "Abschnitt 2 Ende",
                footer,
            ]
        ),
        "\n".join(
            [
                header,
                "Kapitel 3",
                "Empfehlung",
                "Metformin",
                "Kombinationstherapie mit SGLT2-Inhibitoren erwägen.",
                "Abschnitt 3 Ende",
                footer,
            ]
        ),
    ]
    pdf_text = "\n\f\n".join(pages)
    cleaned_pdf, pdf_omissions = strip_references_and_boilerplate(pdf_text)

    assert "\f" not in cleaned_pdf
    assert header not in cleaned_pdf
    assert footer not in cleaned_pdf
    assert cleaned_pdf.count("Empfehlung") == 3
    assert cleaned_pdf.count("Metformin\n") == 3
    assert [o.reason for o in pdf_omissions] == [
        "repeated_header_footer",
        "repeated_header_footer",
    ]

    # 2) Markdown/plain text without \f: repeated clinical words/headings kept, watermarks stripped.
    md_text = "\n\n".join(
        [
            "Seite 1 von 3",
            "## Empfehlung",
            "Metformin",
            "Seite 1 von 3",
            "## Empfehlung",
            "Metformin",
            "Seite 1 von 3",
            "## Empfehlung",
            "Metformin",
        ]
    )
    cleaned_md, md_omissions = strip_references_and_boilerplate(md_text)

    assert "Seite 1 von 3" not in cleaned_md
    assert cleaned_md.count("## Empfehlung") == 3
    assert cleaned_md.count("Metformin") == 3
    assert [o.reason for o in md_omissions] == ["repeated_header_footer"]


def test_strip_german_references_and_boilerplate_preserves_anhang() -> None:
    text = (
        "# Nationale VersorgungsLeitlinie\n\n"
        "Metformin wird als Basistherapie empfohlen.\n\n"
        "## 12.1 Interessenkonflikte\n\n"
        "Alle Autoren haben ihre Interessenkonflikte offengelegt.\n\n"
        "Section 13 Literaturverzeichnis\n\n"
        "[1] Müller A et al. Dtsch Arztebl. 2023;120:1-8.\n"
        "[2] Schmidt B et al. Diabetologie. 2022;17:100-110.\n\n"
        "## 14. Anhang: Dosierungstabelle\n\n"
        "| Wirkstoff | Startdosis |\n"
        "| --- | --- |\n"
        "| Metformin | 500 mg |"
    )

    cleaned, omissions = strip_references_and_boilerplate(text)

    assert "Metformin wird als Basistherapie empfohlen." in cleaned
    assert "## 14. Anhang: Dosierungstabelle" in cleaned
    assert "| Metformin | 500 mg |" in cleaned
    assert "Interessenkonflikte offengelegt" not in cleaned
    assert "Müller A et al." not in cleaned

    assert [(o.reason, o.heading) for o in omissions] == [
        ("boilerplate_section", "## 12.1 Interessenkonflikte"),
        ("references", "Section 13 Literaturverzeichnis"),
    ]


def test_group_chunks_for_extraction_groups_and_validates() -> None:
    assert group_chunks_for_extraction([]) == []
    with pytest.raises(ValueError, match="max_extract_chars"):
        group_chunks_for_extraction([("chunk", 0, 5)], max_extract_chars=0)

    chunks = [
        ("Alpha sentence.", 0, 15),
        ("Beta sentence.", 17, 31),
        ("Gamma very long sentence here.", 33, 63),
    ]
    windows = group_chunks_for_extraction(chunks, max_extract_chars=35)
    assert len(windows) == 2
    assert windows[0] == ("Alpha sentence.\n\nBeta sentence.", [0, 1])
    assert windows[1] == ("Gamma very long sentence here.", [2])


def test_pdf_parsed_table_entries_capture_context_and_page_number() -> None:
    fake_table = MagicMock()
    fake_table.extract.return_value = [
        ["Drug", "eGFR Threshold"],
        ["Metformin", "< 30 mL/min"],
    ]
    fake_table.bbox = (0, 100, 200, 150)

    fake_page = MagicMock()
    fake_page.get_text.side_effect = lambda mode="text": (
        "Dosing guidance before table.\nMetformin renal rules follow.\nFollow-up monitoring."
        if mode == "text"
        else []
    )
    fake_page.find_tables.return_value = SimpleNamespace(tables=[fake_table])

    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 1
    fake_doc.__iter__.return_value = iter([fake_page])
    fake_doc.__enter__.return_value = fake_doc
    fake_doc.__exit__.return_value = False

    fake_pymupdf = SimpleNamespace(open=MagicMock(return_value=fake_doc))
    with patch.dict(sys.modules, {"pymupdf": fake_pymupdf}):
        parsed = DocumentParser().parse_bytes(b"%PDF-1.4", "renal.pdf")

    assert len(parsed.tables) == 1
    assert len(parsed.table_entries) == 1
    entry = parsed.table_entries[0]
    assert isinstance(entry, ParsedTable)
    assert entry.page_number == 1
    assert "| Drug | eGFR Threshold |" in entry.markdown
    assert "| Metformin | < 30 mL/min |" in entry.markdown


def test_is_toc_table_detects_dotted_leaders_and_headings() -> None:
    from clinical_graph.parser.document_parser import _is_toc_table

    dotted_toc = (
        "| Section | Page |\n"
        "| --- | --- |\n"
        "| 1. Introduction ...................... | 4 |\n"
        "| 2. Diagnosis . . . . . . . . . . . . . | 9 |\n"
        "| 3. Treatment ......................... | 15 |"
    )
    assert _is_toc_table(dotted_toc) is True

    german_toc = (
        "| Inhaltsverzeichnis | Seite |\n"
        "| --- | --- |\n"
        "| 1 Einleitung | 3 |\n"
        "| 2 Diagnostik | 8 |\n"
        "| 3 Therapie | 14 |"
    )
    assert _is_toc_table(german_toc) is True

    english_toc = (
        "| Table of Contents | |\n"
        "| --- | --- |\n"
        "| Overview | 1 |\n"
        "| Dosing | 5 |"
    )
    assert _is_toc_table(english_toc) is True

    clinical_table = (
        "| Drug | Dose | eGFR |\n"
        "| --- | --- | --- |\n"
        "| Metformin | 500 mg BID | >= 30 mL/min |\n"
        "| Empagliflozin | 10 mg daily | >= 20 mL/min |"
    )
    assert _is_toc_table(clinical_table) is False


def test_parse_pdf_skips_toc_tables_and_keeps_clinical_tables() -> None:
    toc_table = MagicMock()
    toc_table.extract.return_value = [
        ["Inhaltsverzeichnis", "Seite"],
        ["1 Einleitung ..........", "3"],
        ["2 Therapie ............", "10"],
    ]

    clinical_table = MagicMock()
    clinical_table.extract.return_value = [
        ["Drug", "Dose"],
        ["Empagliflozin", "10 mg daily"],
    ]
    clinical_table.bbox = (0, 100, 200, 150)

    fake_page = MagicMock()
    fake_page.get_text.side_effect = lambda mode="text": (
        "Clinical dosing guidance." if mode == "text" else []
    )
    fake_page.find_tables.return_value = SimpleNamespace(tables=[toc_table, clinical_table])

    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 1
    fake_doc.__iter__.return_value = iter([fake_page])
    fake_doc.__enter__.return_value = fake_doc
    fake_doc.__exit__.return_value = False

    fake_pymupdf = SimpleNamespace(open=MagicMock(return_value=fake_doc))
    with patch.dict(sys.modules, {"pymupdf": fake_pymupdf}):
        parsed = DocumentParser().parse_bytes(b"%PDF-1.4", "guideline.pdf")

    assert len(parsed.tables) == 1
    assert len(parsed.table_entries) == 1
    assert "Empagliflozin" in parsed.tables[0]
    assert "Inhaltsverzeichnis" not in parsed.tables[0]


def test_parse_pdf_dict_font_aware_markdown_formatting() -> None:
    # Median body font size will be 10.0 across the majority of spans.
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"text": "Clinical Guideline", "size": 16.0, "flags": 16},
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"text": "Pharmacotherapy", "size": 13.0, "flags": 16},
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"text": "First-Line Therapy", "size": 11.0, "flags": 16},
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"text": "Start ", "size": 10.0, "flags": 0},
                            {"text": "Metformin", "size": 10.0, "flags": 16},
                            {"text": " for ", "size": 10.0, "flags": 0},
                            {"text": "type 2 diabetes", "size": 10.0, "flags": 2},
                            {"text": " in adults.", "size": 10.0, "flags": 0},
                        ]
                    },
                    {
                        "spans": [
                            {"text": "• Monitor eGFR regularly", "size": 10.0, "flags": 0},
                        ]
                    },
                    {
                        "spans": [
                            {"text": "▪ ", "size": 10.0, "flags": 16},
                            {"text": "Adjust dose below 45 mL/min", "size": 10.0, "flags": 0},
                        ]
                    },
                ],
            },
            {
                # Image block should be ignored.
                "type": 1,
                "lines": [],
            },
        ]
    }

    fake_table = MagicMock()
    fake_table.extract.return_value = [
        ["eGFR", "Action"],
        [">= 45", "Full dose"],
    ]
    fake_table.bbox = (0, 200, 300, 260)

    fake_page = MagicMock()
    fake_page.get_text.side_effect = lambda mode="text": (
        page_dict if mode == "dict" else []
    )
    fake_page.find_tables.return_value = SimpleNamespace(tables=[fake_table])

    fake_doc = MagicMock()
    fake_doc.__len__.return_value = 1
    fake_doc.__iter__.return_value = iter([fake_page])
    fake_doc.__enter__.return_value = fake_doc
    fake_doc.__exit__.return_value = False

    fake_pymupdf = SimpleNamespace(open=MagicMock(return_value=fake_doc))
    with patch.dict(sys.modules, {"pymupdf": fake_pymupdf}):
        parsed = DocumentParser().parse_bytes(b"%PDF-1.4", "rich.pdf")

    assert "# Clinical Guideline" in parsed.text
    assert "## Pharmacotherapy" in parsed.text
    assert "### First-Line Therapy" in parsed.text
    assert "Start **Metformin** for *type 2 diabetes* in adults." in parsed.text
    assert "- Monitor eGFR regularly" in parsed.text
    assert "- Adjust dose below 45 mL/min" in parsed.text
    assert "| eGFR | Action |" in parsed.text
    assert "| >= 45 | Full dose |" in parsed.text



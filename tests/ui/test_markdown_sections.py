"""Tests for parsing markdown headings into a nested section tree."""

from pathlib import Path

from ui.markdown_sections import parse_sections

ROOT = Path(__file__).resolve().parent.parent.parent
DETAILS_DOC = ROOT / "explanations" / "frontend" / "01_single_diode_implementation_details.md"


def test_headings_nest_by_level():
    preamble, sections = parse_sections(
        "## Parent\nintro\n### Child\nmiddle\n#### Grandchild\ndeep\n"
    )

    assert preamble == ""
    assert [s.title for s in sections] == ["Parent"]

    parent = sections[0]
    assert parent.level == 2
    assert parent.body == "intro"

    child = parent.children[0]
    assert (child.level, child.title, child.body) == (3, "Child", "middle")

    grandchild = child.children[0]
    assert (grandchild.level, grandchild.title, grandchild.body) == (4, "Grandchild", "deep")
    assert grandchild.children == []


def test_content_above_first_section_is_preamble():
    preamble, sections = parse_sections("# Title\n\nIntro paragraph.\n\n## First\nbody\n")

    assert preamble == "# Title\n\nIntro paragraph."
    assert [s.title for s in sections] == ["First"]
    assert sections[0].body == "body"


def test_hashes_inside_fenced_code_are_not_headings():
    """Python comments in fenced snippets must not be parsed as headings.

    The explanation docs embed code whose comments start with '#'; treating those
    as headings would split the code block into bogus expanders.
    """
    text = (
        "## Code Implementation\n"
        "```python\n"
        "def solve_current(voltage, params):\n"
        "    if r_s == 0:\n"
        "        # Degenerate case: no series resistance\n"
        "        return current\n"
        "\n"
        "    # Standard closed-form (Lambert W) solution\n"
        "    w = lambertw(a * np.exp(b)).real\n"
        "```\n"
    )

    _, sections = parse_sections(text)

    assert [s.title for s in sections] == ["Code Implementation"]
    assert sections[0].children == []
    body = sections[0].body
    assert "# Degenerate case: no series resistance" in body
    assert "# Standard closed-form (Lambert W) solution" in body
    assert body.count("```") == 2


def test_siblings_share_a_parent():
    _, sections = parse_sections("## Parent\n### One\na\n### Two\nb\n")

    parent = sections[0]
    assert [c.title for c in parent.children] == ["One", "Two"]
    assert [c.body for c in parent.children] == ["a", "b"]


def test_dedent_closes_back_to_root():
    _, sections = parse_sections("## First\n#### Deep\nx\n## Second\ny\n")

    assert [s.title for s in sections] == ["First", "Second"]
    assert [c.title for c in sections[0].children] == ["Deep"]
    assert sections[1].children == []
    assert sections[1].body == "y"


def test_min_level_controls_what_becomes_a_section():
    preamble, sections = parse_sections("# Title\nintro\n## Sub\nbody\n", min_level=1)

    assert preamble == ""
    assert [s.title for s in sections] == ["Title"]
    assert [c.title for c in sections[0].children] == ["Sub"]


def test_trailing_closing_hashes_are_stripped_from_title():
    _, sections = parse_sections("## Heading ##\nbody\n")

    assert sections[0].title == "Heading"


def test_parses_the_shipped_explanation_doc():
    preamble, sections = parse_sections(DETAILS_DOC.read_text(encoding="utf-8"))

    assert preamble.startswith("# Underlying Explanation to the Single Diode Model")
    assert [s.title for s in sections] == [
        "The Single Diode Model",
        "Reference Parameters",
        "Single Diode Fitting Overview and Limitations",
    ]

    # Third-level nesting the modal relies on: parameters, then their snippets.
    reference_params = sections[1]
    assert "Photo-current density" in [c.title for c in reference_params.children]

    temperature = next(
        c for c in reference_params.children if c.title == "Effect of Temperature"
    )
    assert [c.title for c in temperature.children] == [
        "`J_0` is  temperature-sensitive",
        "`J_ph`",
        "Reference",
    ]

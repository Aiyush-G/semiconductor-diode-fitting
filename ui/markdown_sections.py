"""Render a markdown document as a tree of collapsible Streamlit expanders.

Long explanatory documents are hard to navigate as one continuous scroll, so the
headings are parsed into a section tree and each becomes an expander nested at
its heading depth. Sections are derived from the markdown itself, so new
headings appear in the UI without any code change here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import streamlit as st

# ATX heading: 1-6 leading '#' followed by a space. Any trailing '#'s are
# closing marks rather than title text, so they are stripped.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Opening/closing code fence (``` or ~~~), indented up to 3 spaces per CommonMark.
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass
class Section:
    """One heading and the markdown body that belongs to it."""

    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip("\n")


def parse_sections(text: str, min_level: int = 2) -> tuple[str, list[Section]]:
    """Split markdown into a preamble plus a tree of heading sections.

    Headings at ``min_level`` or deeper become sections, nested by heading level.
    Everything above the first such heading is returned as the preamble (for the
    explanation docs, the H1 title and its intro paragraphs).

    Lines inside fenced code blocks are never read as headings: the docs embed
    Python snippets whose comments would otherwise be mistaken for headings and
    split the code block apart.
    """
    preamble_lines: list[str] = []
    roots: list[Section] = []
    stack: list[Section] = []
    fence: str | None = None

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            # Either way the fence line is body content, so fall through.
        elif fence is None:
            heading = _HEADING_RE.match(line)
            if heading and len(heading.group(1)) >= min_level:
                section = Section(level=len(heading.group(1)), title=heading.group(2))
                while stack and stack[-1].level >= section.level:
                    stack.pop()
                if stack:
                    stack[-1].children.append(section)
                else:
                    roots.append(section)
                stack.append(section)
                continue

        if stack:
            stack[-1].lines.append(line)
        else:
            preamble_lines.append(line)

    return "\n".join(preamble_lines).strip("\n"), roots


def render_sections(sections: Sequence[Section], *, expanded: bool = False) -> None:
    """Render each section as an expander, recursing into its child headings.

    ``with st.expander(...)`` makes the expander the active container, so the
    recursive call nests the children inside their parent.
    """
    for section in sections:
        with st.expander(section.title, expanded=expanded):
            if section.body:
                st.markdown(section.body)
            render_sections(section.children, expanded=expanded)

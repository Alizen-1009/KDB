#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest

from kb_utils import ROOT
from update_index import prefer_html_artifacts, rel_link


class ReportArtifactIndexTests(unittest.TestCase):
    def test_html_report_uses_portable_markdown_link(self) -> None:
        report = ROOT / "output" / "reports" / "示例报告.html"

        self.assertEqual(
            rel_link(report),
            "- [示例报告](../output/reports/示例报告.html)",
        )

    def test_markdown_report_keeps_obsidian_wikilink(self) -> None:
        report = ROOT / "output" / "reports" / "示例报告.md"

        self.assertEqual(
            rel_link(report),
            "- [[../output/reports/示例报告|示例报告]]",
        )

    def test_html_wins_when_legacy_markdown_copy_has_same_stem(self) -> None:
        files = [
            Path("A.md"),
            Path("A.html"),
            Path("B.md"),
        ]

        self.assertEqual(
            prefer_html_artifacts(files),
            [Path("A.html"), Path("B.md")],
        )


if __name__ == "__main__":
    unittest.main()

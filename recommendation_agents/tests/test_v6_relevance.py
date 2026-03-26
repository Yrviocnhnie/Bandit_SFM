from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from recommendation_agents.v6_relevance import parse_v6_relevance_markdown


class V6RelevanceParserTest(unittest.TestCase):
    def test_parse_v6_relevance_markdown_returns_3_3_2_for_both_heads(self) -> None:
        markdown = "\n".join(
            [
                "# v6",
                "",
                "## Scenario Defaults",
                "",
                "| scenarioId | scenarioNameZh | most relevant 3 R/O actionIds | other plausible 3 R/O actionIds | irrelevant 2 R/O actionIds | most relevant 3 app categories | other plausible 3 app categories | irrelevant 2 app categories |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| `ARRIVE_OFFICE` | 到达办公室 | `A1`<br>`A2`<br>`A3` | `A4`<br>`A5`<br>`A6` | `A7`<br>`A8` | `app1`<br>`app2`<br>`app3` | `app4`<br>`app5`<br>`app6` | `app7`<br>`app8` |",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "v6.md"
            path.write_text(markdown)
            parsed = parse_v6_relevance_markdown(path)

        self.assertEqual(parsed["ro"]["ARRIVE_OFFICE"]["most_relevant_3"], ("A1", "A2", "A3"))
        self.assertEqual(parsed["ro"]["ARRIVE_OFFICE"]["other_plausible_3"], ("A4", "A5", "A6"))
        self.assertEqual(parsed["ro"]["ARRIVE_OFFICE"]["irrelevant_2"], ("A7", "A8"))
        self.assertEqual(parsed["app"]["ARRIVE_OFFICE"]["most_relevant_3"], ("app1", "app2", "app3"))
        self.assertEqual(parsed["app"]["ARRIVE_OFFICE"]["other_plausible_3"], ("app4", "app5", "app6"))
        self.assertEqual(parsed["app"]["ARRIVE_OFFICE"]["irrelevant_2"], ("app7", "app8"))


if __name__ == "__main__":
    unittest.main()

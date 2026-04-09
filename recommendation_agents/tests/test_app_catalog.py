from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.catalog import build_app_metadata_from_catalog_markdown


class AppCatalogTest(unittest.TestCase):
    def test_build_app_metadata_from_global_catalog(self) -> None:
        payload = {
            'globalActions': [
                {'actionId': 'O_SHOW_SCHEDULE', 'actionType': 'O', 'nameEn': 'Show schedule'},
            ],
            'scenarioDefaults': [
                {
                    'scenarioId': 'ARRIVE_OFFICE',
                    'defaultActionIds': ['O_SHOW_SCHEDULE'],
                    'defaultAppCategories': ['productivity', 'news', 'music'],
                },
                {
                    'scenarioId': 'HOME_EVENING',
                    'defaultActionIds': ['O_SHOW_SCHEDULE'],
                    'defaultAppCategories': ['entertainment', 'music', 'social'],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            markdown_path = tmp_path / 'scenario_recommendation_actions_v5.md'
            markdown_path.write_text('# placeholder\n')
            (tmp_path / 'global_action_space.json').write_text(json.dumps(payload))
            output_path = tmp_path / 'app_metadata.json'

            summary = build_app_metadata_from_catalog_markdown(markdown_path, output_path)

            self.assertEqual(summary.total_scenarios, 2)
            self.assertEqual(summary.total_actions, 10)

            metadata = json.loads(output_path.read_text())
            self.assertEqual(len(metadata['global_action_ids']), 10)
            arrive_office = next(item for item in metadata['scenario_default_rankings'] if item['scenario_id'] == 'ARRIVE_OFFICE')
            self.assertEqual(arrive_office['default_action_ids'], ['productivity', 'news', 'music'])


if __name__ == '__main__':
    unittest.main()

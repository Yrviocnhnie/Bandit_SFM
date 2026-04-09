from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.catalog import build_ro_metadata_from_catalog_markdown


class CatalogTest(unittest.TestCase):
    def test_build_ro_metadata_from_global_catalog(self) -> None:
        payload = {
            'globalActions': [
                {'actionId': 'O_SHOW_SCHEDULE', 'actionType': 'O', 'nameEn': 'Show schedule'},
                {'actionId': 'O_SHOW_TODO', 'actionType': 'O', 'nameEn': 'Show todo'},
                {'actionId': 'R_PLAN_DAY_OVER_COFFEE', 'actionType': 'R', 'nameEn': 'Plan day over coffee'},
            ],
            'scenarioDefaults': [
                {
                    'scenarioId': 'ARRIVE_OFFICE',
                    'defaultActionIds': ['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'],
                    'defaultAppCategories': ['productivity', 'news', 'music'],
                },
                {
                    'scenarioId': 'HOME_EVENING',
                    'defaultActionIds': ['O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'],
                    'defaultAppCategories': ['entertainment', 'music', 'social'],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            markdown_path = tmp_path / 'scenario_recommendation_actions_v5.md'
            markdown_path.write_text('# placeholder\n')
            (tmp_path / 'global_action_space.json').write_text(json.dumps(payload))
            output_path = tmp_path / 'ro_metadata.json'

            summary = build_ro_metadata_from_catalog_markdown(markdown_path, output_path)

            self.assertEqual(summary.total_scenarios, 2)
            self.assertEqual(summary.total_actions, 3)
            self.assertEqual(summary.min_actions_per_scenario, 2)
            self.assertEqual(summary.max_actions_per_scenario, 3)

            metadata = json.loads(output_path.read_text())
            self.assertEqual(metadata['global_action_ids'], ['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'])
            self.assertNotIn('scenarios', metadata)
            arrive_office = next(item for item in metadata['scenario_default_rankings'] if item['scenario_id'] == 'ARRIVE_OFFICE')
            self.assertEqual(
                arrive_office['default_action_ids'],
                ['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'],
            )


if __name__ == '__main__':
    unittest.main()

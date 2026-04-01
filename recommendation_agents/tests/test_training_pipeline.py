from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from recommendation_agents.linucb import MaskedSharedLinearUCB, build_shared_action_feature_matrix
from recommendation_agents.schemas import ScoreRequest
from recommendation_agents.trainer import _build_grouped_dense_examples, score_v0, train_v0


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def _context(**overrides):
    payload = {
        'state_current': 'office_arriving',
        'precondition': 'commuting_walk_out',
        'state_duration_sec': 180,
        'ps_time': 'morning',
        'hour': 9,
        'cal_hasUpcoming': 1,
        'ps_dayType': 'workday',
        'ps_motion': 'stationary',
        'wifiLost': 0,
        'wifiLostCategory': 'work',
        'cal_eventCount': 2,
        'cal_inMeeting': 0,
        'cal_nextLocation': 'work',
        'ps_sound': 'quiet',
        'sms_delivery_pending': 0,
        'sms_train_pending': 0,
        'sms_flight_pending': 0,
        'sms_hotel_pending': 0,
        'sms_movie_pending': 0,
        'sms_hospital_pending': 0,
        'sms_ride_pending': 0,
        'timestep': 32400,
        'ps_location': 'work',
        'ps_phone': 'on_desk',
        'batteryLevel': 88,
        'isCharging': 1,
        'networkType': 'wifi',
        'activityState': 'sitting',
        'activityDuration': 900,
        'user_id_hash_bucket': 'b07',
        'age_bucket': '25_34',
        'sex': 'female',
        'has_kids': 0,
    }
    payload.update(overrides)
    return payload


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for the training pipeline tests')
class TrainingPipelineTest(unittest.TestCase):
    def test_shared_action_features_include_identity_block(self) -> None:
        action_ids = ['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE']
        matrix = build_shared_action_feature_matrix(action_ids)
        self.assertEqual(matrix.shape[0], 3)
        identity_block = matrix[:, 1 : 1 + len(action_ids)]
        self.assertEqual(identity_block.tolist(), [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_shared_linear_model_adds_small_context_interaction_block(self) -> None:
        model = MaskedSharedLinearUCB(
            action_ids=['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'],
            feature_dim=306,
            device='cpu',
        )
        self.assertGreater(model.context_feature_dim, model.feature_dim)
        self.assertEqual(model.context_interaction_dim, 10)

    def _write_fixture_files(self, tmp_dir: str) -> tuple[Path, Path]:
        metadata_path = Path(tmp_dir) / 'metadata.json'
        samples_path = Path(tmp_dir) / 'samples.jsonl'
        metadata = {
            'schema_version': 'ro-global-catalog',
            'global_action_ids': ['O_SHOW_SCHEDULE', 'O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'],
            'actions': [
                {'action_id': 'O_SHOW_SCHEDULE', 'display_name': 'Show schedule'},
                {'action_id': 'O_SHOW_TODO', 'display_name': 'Show todo'},
                {'action_id': 'R_PLAN_DAY_OVER_COFFEE', 'display_name': 'Plan day over coffee'},
            ],
        }
        samples = [
            {
                'event_id': 'evt-001',
                'scenario_id': 'ARRIVE_OFFICE',
                'context': _context(),
                'selected_action': 'O_SHOW_SCHEDULE',
                'reward': 1.0,
                'propensity': 1.0,
            },
            {
                'event_id': 'evt-002',
                'scenario_id': 'ARRIVE_OFFICE',
                'context': _context(hour=8, cal_eventCount=4),
                'selected_action': 'O_SHOW_SCHEDULE',
                'reward': 1.0,
                'propensity': 1.0,
            },
            {
                'event_id': 'evt-003',
                'scenario_id': 'ARRIVE_OFFICE',
                'context': _context(hour=10, ps_sound='normal'),
                'selected_action': 'O_SHOW_TODO',
                'reward': 0.5,
                'propensity': 1.0,
            },
            {
                'event_id': 'evt-004',
                'scenario_id': 'HOME_EVENING',
                'context': _context(state_current='home_evening', precondition='office_working', ps_time='evening', hour=19, ps_location='home', wifiLostCategory='home'),
                'selected_action': 'R_PLAN_DAY_OVER_COFFEE',
                'reward': 0.2,
                'propensity': 1.0,
            },
        ]
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        samples_path.write_text(''.join(json.dumps(row) + '\n' for row in samples))
        return metadata_path, samples_path

    def test_train_and_score_with_global_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
            )
            self.assertEqual(metrics.sample_count, 4)
            self.assertEqual(metrics.unique_scenarios_seen, 2)

            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(ranked[0].action_id, 'O_SHOW_SCHEDULE')
            self.assertEqual(len(ranked), 3)

    def test_score_request_can_optionally_restrict_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact'
            train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
            )
            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(
                    context=_context(),
                    shown_actions=('O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'),
                ),
                top_k=2,
                device='cpu',
            )
            self.assertEqual([item.action_id for item in ranked], ['O_SHOW_TODO', 'R_PLAN_DAY_OVER_COFFEE'])

    def test_train_and_score_with_shared_linear_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_shared'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
                model_type='shared-linear',
            )
            self.assertEqual(metrics.sample_count, 4)

            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_shared_linear_ucb_v0')

    def test_train_and_score_with_neural_linear_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_neural'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
                model_type='neural-linear',
            )
            self.assertEqual(metrics.sample_count, 4)

            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_neural_linear_ucb_v0')

    def test_grouped_dense_examples_build_expected_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            from recommendation_agents.feature_space import V0FeatureSpace
            from recommendation_agents.metadata import BanditMetadata

            examples = _build_grouped_dense_examples(
                samples_path=samples_path,
                metadata=BanditMetadata.load(metadata_path),
                feature_space=V0FeatureSpace(),
            )
            self.assertEqual(len(examples), 4)
            first = examples[0]
            self.assertEqual(first.target.shape[0], 3)
            self.assertAlmostEqual(float(first.target.max()), 1.0)

    def test_train_and_score_with_neural_scorer_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_neural_scorer'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.0,
                default_bonus=0.0,
                device='cpu',
                progress_every=10,
                model_type='neural-scorer',
            )
            self.assertEqual(metrics.sample_count, 4)
            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_neural_scorer_v0')

    def test_train_and_score_with_neural_ucb_lite_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_neural_ucb_lite'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
                model_type='neural-ucb-lite',
            )
            self.assertEqual(metrics.sample_count, 4)
            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            report = json.loads((artifact_dir / 'training_report.json').read_text())
            self.assertEqual(report['hyperparameters']['model_type'], 'neural-ucb-lite')
            self.assertIn('neural_pretrain', report['hyperparameters'])
            self.assertEqual(report['hyperparameters']['neural_pretrain']['ucb_feature_dim'], 65)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_neural_ucb_lite_v0')

    def test_train_and_score_with_neural_ucb_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_neural_ucb'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
                model_type='neural-ucb',
            )
            self.assertGreaterEqual(metrics.sample_count, 1)
            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_neural_ucb_v0')
            self.assertGreater(manifest['parameter_dim'], 0)

    def test_train_and_score_with_neural_ucb_direct_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path, samples_path = self._write_fixture_files(tmp_dir)
            artifact_dir = Path(tmp_dir) / 'artifact_neural_ucb_direct'

            metrics = train_v0(
                metadata_path=metadata_path,
                samples_path=samples_path,
                output_dir=artifact_dir,
                alpha=0.05,
                default_bonus=0.75,
                device='cpu',
                progress_every=10,
                model_type='neural-ucb-direct',
            )
            self.assertEqual(metrics.sample_count, 4)
            ranked = score_v0(
                artifact_dir=artifact_dir,
                metadata_path=metadata_path,
                request=ScoreRequest(context=_context()),
                top_k=3,
                device='cpu',
            )
            self.assertEqual(len(ranked), 3)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['model_type'], 'masked_neural_ucb_direct_v0')


if __name__ == '__main__':
    unittest.main()

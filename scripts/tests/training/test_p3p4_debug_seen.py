"""No data files or GPUs: seen-window selection, guards and paired summaries."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments.debug_audit.checkpoint_seen import (
    parser, require_execution, run, summarize, validate_seen_rows,
)


class SeenCheckpointAuditTests(unittest.TestCase):
    def manifest(self):
        return {'windows': {'5': {'train': [
            dict(episode=ep, start=start, frame_ids=list(range(start, start+6)))
            for ep in (1, 2, 4, 5, 7, 8) for start in (10, 20, 30, 40)]}}}

    def test_exact_manifest_rows_no_episode_membership_substitution(self):
        rows = validate_seen_rows(self.manifest())
        self.assertEqual(len(rows), 24)
        self.assertNotIn((2, 73), [(r['episode'], r['start']) for r in rows])
        mutations = [lambda rows: rows.append(copy.deepcopy(rows[0])),
                     lambda rows: rows.__setitem__(0, copy.deepcopy(rows[1])),
                     lambda rows: rows[0].update(episode=3),
                     lambda rows: rows[0].update(frame_ids=list(range(10, 25)))]
        for mutate in mutations:
            value = self.manifest()
            mutate(value['windows']['5']['train'])
            with self.assertRaises(ValueError):
                validate_seen_rows(value)

    def test_forbid_protected_job_before_any_cuda_use(self):
        for job in ('', '30903'):
            with patch.dict('os.environ', {'SLURM_JOB_ID': job}), patch(
                    'torch.cuda.is_available', side_effect=AssertionError('Must not inspect GPU')):
                with self.assertRaisesRegex(RuntimeError, 'new Slurm job'):
                    require_execution('expected')

    def test_guard_requires_one_gpu_clean_expected_source(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID': '99999'}), patch(
                'torch.cuda.is_available', return_value=True), patch(
                'torch.cuda.device_count', return_value=1):
            with patch('subprocess.check_output', side_effect=['abc\n', '', '']):
                self.assertEqual(require_execution('abc'), 'abc')
            with patch('subprocess.check_output', side_effect=['wrong\n', '']):
                with self.assertRaisesRegex(RuntimeError, 'SHA drift'):
                    require_execution('abc')
            with patch('subprocess.check_output', side_effect=['abc\n', ' M model.py']):
                with self.assertRaisesRegex(RuntimeError, 'tracked changes'):
                    require_execution('abc')
            with patch('subprocess.check_output', side_effect=['abc\n', '', 'scripts/training/model.py']):
                with self.assertRaisesRegex(RuntimeError, 'Frozen model'):
                    require_execution('abc')

    def test_existing_output_rejected_before_gpu_or_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'existing.json'
            output.write_text('preserve', encoding='utf-8')
            with patch('scripts.experiments.debug_audit.checkpoint_seen.require_execution',
                       side_effect=AssertionError('No execution allowed')):
                with self.assertRaises(FileExistsError):
                    run(SimpleNamespace(output=str(output)))
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')

    def test_parser_and_paired_summary(self):
        args = parser().parse_args(['--root', 'data', '--manifest', 'frozen.json',
                                    '--output', 'seen.json', '--expected-sha', 'abc'])
        self.assertEqual(args.expected_sha, 'abc')
        def metrics(value):
            return dict(cd_mm=[value]*5, rgb=[.1]*5, alpha=[.2]*5,
                        motion=[dict(displacement_rms_mm=value, centroid_error_mm=value,
                                     centered_cd_mm=value, gt_centroid_motion_mm=[3, 4, 0])]*5)
        result = summarize([dict(model=metrics(1), copy_initial=metrics(2)),
                            dict(model=metrics(3), copy_initial=metrics(2))])
        self.assertEqual(result['model']['mean_cd_mm'], 2.)
        self.assertEqual(result['windows_with_lower_mean_cd_than_copy'], 1)
        self.assertEqual(result['pred_minus_copy_cd_mm_mean'], 0.)
        self.assertEqual(result['gt_independent_set_centroid_motion_norm_mm_by_horizon'], [5.]*5)


if __name__ == '__main__':
    unittest.main()

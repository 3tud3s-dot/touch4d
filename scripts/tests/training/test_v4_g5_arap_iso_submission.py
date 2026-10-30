"""CPU-only Slurm command, durable identity and preparation gate tests."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np

from scripts.experiments.v4_g5_arap_iso import submit, prepare
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(root=str(self.root/'data'), depth_root=str(self.root/'depth'),
            g0_reference=str(self.root/'g0'), common=str(self.root/submit.COMMON_BUCKET/'run-001'),
            c_output=str(self.root/submit.BUCKETS['g5_arap']/'run-001'),
            d_output=str(self.root/submit.BUCKETS['g5_arap_iso']/'run-001'),
            reviewed_graphs=str(self.root/'reviewed'), expected_sha='a'*40)
        self.calls = []; self.next_id = 100
        self.patch = patch.object(submit, 'source_guard', return_value=(self.args.expected_sha, {})); self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.tmp.cleanup()

    def accepted(self, command, **kw):
        self.assertEqual(command[0], 'sbatch'); self.calls.append((command, kw))
        self.next_id += 1; return str(self.next_id)+';amax\n'

    def test_complete_chain_dependencies_and_one_gpu_script_contract(self):
        obj = submit.Submission(self.args)
        with patch.object(submit.subprocess, 'check_output', side_effect=self.accepted):
            jobs = obj.submit_chain()
        self.assertEqual(jobs, dict(prepare='101', c='102', d='103', evaluate='104'))
        self.assertEqual(len(self.calls), 4)
        for (command, kwargs), phase in zip(self.calls, submit.PHASES):
            self.assertIn('--kill-on-invalid-dep=yes', command)
            env = kwargs['env']; self.assertEqual(env['SOURCE'], str(Path.cwd().resolve()))
            self.assertEqual(env['EXPECTED_SHA'], self.args.expected_sha)
            self.assertEqual(env['REVIEWED_GRAPHS'], self.args.reviewed_graphs)
            self.assertNotIn('--jobid=32087', command)
            if phase in submit.VARIANTS:
                self.assertEqual(env['VARIANT'], phase); self.assertEqual(env['RESUME'], '0')
        self.assertIn('--dependency=afterok:101', self.calls[1][0]); self.assertIn('--dependency=afterok:101', self.calls[2][0])
        self.assertIn('--dependency=afterok:102:103', self.calls[3][0])
        self.assertEqual(json.loads(obj.path.read_text())['status'], 'DEPENDENCY_CHAIN_SUBMITTED')

    def test_repeated_same_submission_never_duplicates_even_after_output_exists(self):
        obj = submit.Submission(self.args)
        with patch.object(submit.subprocess, 'check_output', side_effect=self.accepted):
            expected = obj.submit_chain()
        Path(self.args.c_output).mkdir(parents=True)
        reopened = submit.Submission(self.args)
        with patch.object(submit.subprocess, 'check_output') as shell:
            self.assertEqual(reopened.submit_chain(), expected)
        shell.assert_not_called()

    def test_wrong_source_path_and_run_identity_are_rejected_without_rewrite(self):
        obj = submit.Submission(self.args); original = obj.path.read_bytes()
        for key, value in (('expected_sha', 'b'*40), ('root', str(self.root/'otherdata')),
                           ('c_output', str(self.root/submit.BUCKETS['g5_arap']/'run-other'))):
            args = copy.deepcopy(self.args); setattr(args, key, value)
            with self.assertRaises(ValueError):submit.Submission(args)
            self.assertEqual(obj.path.read_bytes(), original)

    def test_preexisting_variant_or_unregistered_results_not_overwritten(self):
        Path(self.args.c_output).mkdir(parents=True)
        marker = Path(self.args.c_output)/'user.pt'; marker.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):submit.Submission(self.args)
        self.assertEqual(marker.read_bytes(), b'preserve')
        # A separate comparison path can contain neither foreign nor old results.
        self.args.c_output = str(self.root/submit.BUCKETS['g5_arap']/'new')
        (Path(self.args.common)/'user.json').write_text('preserve')
        with self.assertRaises(FileExistsError):submit.Submission(self.args)

    def test_changed_stored_dependency_is_not_silently_reused(self):
        obj = submit.Submission(self.args)
        with patch.object(submit.subprocess, 'check_output', side_effect=self.accepted):obj.submit_chain()
        obj.state['jobs']['g5_arap']['dependencies'] = ['999']
        with self.assertRaises(ValueError):obj.submit_phase('g5_arap', ('101',))

    def cpu_preflight(self):
        manifest = dict(train=dict(windows=1514, window_keys=[[2, k] for k in range(1514)]),
            val=dict(windows=100, window_keys=[[3, k] for k in range(100)]))
        orders = [list(range(1514)) for _ in range(8)]
        hashes = [submit.digest(order) for order in orders]
        return dict(passed=True, source_sha=self.args.expected_sha, optimizer_created=False, successful_updates=0,
            data=dict(manifest=manifest, manifest_hash=submit.digest(manifest), epoch_orders=orders,
                epoch_order_hashes=hashes, observed_epoch_order_hashes=hashes),
            initialization=dict(shared_exact=True, variant_model_hashes={v:'f'*64 for v in submit.VARIANTS},
                h15_all_attributes_exact=dict(xyz=True, log_scales=True, quaternions=True, opacity_logits=True, sh_dc=True)))

    def test_same_source_complete_cpu_preflight_allows_engineering_and_preserves_logs(self):
        engineering = Path(self.args.common)/'engineering'; engineering.mkdir(parents=True)
        proof = self.cpu_preflight(); path = engineering/'preflight.json'; path.write_text(json.dumps(proof))
        log = engineering/'server_cpu_tests.log'; log.write_text('CPU tests actually passed')
        before = path.read_bytes(); obj = submit.Submission(self.args)
        self.assertTrue(obj.path.exists()); self.assertEqual(path.read_bytes(), before)
        self.assertEqual(log.read_text(), 'CPU tests actually passed')

    def test_stale_failed_or_partial_preflight_does_not_whitelist_engineering(self):
        engineering = Path(self.args.common)/'engineering'; engineering.mkdir(parents=True)
        path = engineering/'preflight.json'
        with self.assertRaises(FileExistsError):submit.Submission(self.args)
        original = self.cpu_preflight()
        for key, value in (('source_sha', 'b'*40), ('passed', False), ('successful_updates', 1)):
            proof = copy.deepcopy(original); proof[key] = value; path.write_text(json.dumps(proof))
            with self.assertRaises(ValueError):submit.Submission(self.args)
        proof = copy.deepcopy(original); proof['data']['epoch_orders'].pop(); path.write_text(json.dumps(proof))
        with self.assertRaises(ValueError):submit.Submission(self.args)

    def test_failed_or_unparseable_submission_retains_unknown_intent_before_external_call(self):
        obj = submit.Submission(self.args)
        def broken(command, **kw):
            saved = json.loads(obj.path.read_text())['jobs']['prepare']
            self.assertEqual(saved['status'], 'SUBMITTING'); self.assertIn(self.args.expected_sha, saved['comment'])
            raise subprocess.CalledProcessError(1, command, output='transport lost after submission')
        with patch.object(submit.subprocess, 'check_output', side_effect=broken):
            with self.assertRaises(subprocess.CalledProcessError):obj.submit_phase('prepare')
        self.assertEqual(json.loads(obj.path.read_text())['jobs']['prepare']['status'], 'SUBMISSION_UNKNOWN')
        self.assertEqual(obj.state['jobs']['prepare']['stdout_stderr'], 'transport lost after submission')
        self.assertNotIn('job_id', obj.state['jobs']['prepare'])

    def unknown_record(self):
        obj = submit.Submission(self.args)
        with patch.object(submit.subprocess, 'check_output', return_value='not-a-job-id'):
            with self.assertRaises(RuntimeError):obj.submit_phase('prepare')
        return obj, obj.state['jobs']['prepare']

    def test_unknown_result_reconciles_exact_name_comment_owner_not_fuzzy(self):
        obj, record = self.unknown_record()
        active = '\n'.join((f'JobId=401 JobName={record["job_name"]} Comment=wrong UserId=tcweng(1025) JobState=RUNNING',
            f'JobId=402 JobName={record["job_name"]} Comment={record["comment"]} UserId=tcweng(1025) JobState=PENDING'))
        with patch.object(submit, '_command', side_effect=(active, '')):
            self.assertEqual(obj.submit_phase('prepare'), '402')
        self.assertEqual(record['status'], 'RECONCILED')

    def test_unknown_result_can_recover_completed_accounting_job(self):
        obj, record = self.unknown_record()
        history = f'405|{record["job_name"]}|{record["comment"]}|COMPLETED|tcweng|\n'
        with patch.object(submit, '_command', side_effect=('', history)):
            self.assertEqual(obj.submit_phase('prepare'), '405')
        self.assertEqual(record['observed_state'], 'COMPLETED')

    def test_unknown_missing_or_multiple_matches_never_repeat_sbatch(self):
        obj, record = self.unknown_record()
        for active in ('', '\n'.join(f'JobId={i} JobName={record["job_name"]} Comment={record["comment"]} UserId=tcweng(1025)' for i in (410, 411))):
            with patch.object(submit, '_command', side_effect=(active, '')), patch.object(submit.subprocess, 'check_output') as shell:
                with self.assertRaisesRegex(RuntimeError, 'unique'):obj.submit_phase('prepare')
            shell.assert_not_called()

    def test_accounting_owner_mismatch_rejected(self):
        obj, record = self.unknown_record()
        raw = f'JobId=420 JobName={record["job_name"]} Comment={record["comment"]} UserId=another(999)'
        with patch.object(submit, '_command', return_value=raw):
            with self.assertRaisesRegex(RuntimeError, 'owner'):obj.submit_phase('prepare')

    def test_main_acquires_submission_flock_before_creating_any_durable_intent(self):
        actual = submit.Submission; locked = []
        def construct(args):
            self.assertEqual(locked, [True]); return actual(args)
        fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a:locked.append(True))
        with (patch.object(submit.argparse.ArgumentParser, 'parse_args', return_value=self.args),
              patch.object(submit, 'precheck', return_value={'passed':True}),
              patch.object(submit, 'Submission', side_effect=construct), patch.dict(sys.modules, {'fcntl':fcntl}),
              patch.object(submit.subprocess, 'check_output', side_effect=self.accepted), patch('builtins.print')):
            submit.main()
        self.assertEqual(len(self.calls), 4)

    def test_login_precheck_rejects_running_inside_any_allocation(self):
        with (patch.object(submit.os, 'name', 'posix'), patch.object(submit.sys, 'platform', 'linux'),
              patch.dict(os.environ, {'SLURM_JOB_ID':'32087'})):
            with self.assertRaisesRegex(RuntimeError, 'login CPU'):submit.precheck(self.args)
        with (patch.object(submit.os, 'name', 'posix'), patch.object(submit.sys, 'platform', 'linux'),
              patch.dict(os.environ, {'SLURM_JOB_ID':'99999'})):
            with self.assertRaisesRegex(RuntimeError, 'login CPU'):submit.precheck(self.args)

    def test_linux_login_owner_partition_quota_and_disk_precheck(self):
        for key in ('root', 'depth_root', 'g0_reference', 'reviewed_graphs'):
            Path(getattr(self.args, key)).mkdir(parents=True)
        for name in ('config.json', 'manifest.json'):(Path(self.args.g0_reference)/name).write_text('{}')
        concrete_path = type(self.root)
        command_outputs = ['own queued jobs', 'PartitionName=5090 State=UP MaxTime=1-00:00:00 MaxNodes=UNLIMITED',
            'Disk quotas for user tcweng (uid1025):\n Filesystem space quota limit\n /dev/data 901G 1000G 1000G\n']
        with (patch.object(submit, 'Path', concrete_path), patch.object(submit.os, 'name', 'posix'),
              patch.object(submit.sys, 'platform', 'linux'), patch.dict(os.environ, {}, clear=True),
              patch.object(submit.getpass, 'getuser', return_value='tcweng'),
              patch.object(submit, '_command', side_effect=command_outputs),
              patch.object(submit.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=200*1024**3))):
            proof = submit.precheck(self.args)
        self.assertEqual(proof['quota_headroom_bytes'], 99*1024**3)
        self.assertFalse(proof['allocation_32087_used']); self.assertIn('own queued', proof['current_jobs'])

    def test_quota_units_unknown_and_exceeded_are_not_assumed_safe(self):
        self.assertEqual(submit.quota_headroom('/dev/x 900G 1000G 1100G'), 100*1024**3)
        self.assertEqual(submit.quota_headroom('/dev/x 1024 8192 8192'), 7168*1024)
        self.assertLess(submit.quota_headroom('/dev/x 1001G* 1000G 1100G'), 0)
        with self.assertRaises(RuntimeError):submit.quota_headroom('unparseable quota result')

    def test_sbatch_scripts_are_single_gpu_fixed_runtime_and_no_training_truncation(self):
        module = Path(submit.__file__).parent
        for file in ('prepare_smoke.sbatch', 'train.sbatch', 'evaluate.sbatch'):
            source = (module/file).read_text()
            for literal in ('#SBATCH --partition=5090', '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=4',
                            '#SBATCH --mem=48G', '--no-requeue', 'exec srun ', 'cuda-12.8', '.venv/bin/python',
                            'test "$SLURM_JOB_ID" != 32087', 'git status --porcelain'):
                self.assertIn(literal, source)
            self.assertNotIn('--max-batches', source); self.assertNotIn('salloc', source)
        source = (module/'train.sbatch').read_text()
        self.assertIn('RESUME:-0', source); self.assertIn('--resume', source)
        self.assertNotIn('best.pt', source); self.assertNotIn('smoke_4updates.pt', source)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_review(self):
        old, new = self.root/'reviewed', self.root/'new'
        (old/'calibration').mkdir(parents=True); (old/'figures').mkdir(); new.mkdir()
        (old/'calibration'/'calibration.json').write_text('{}')
        review = dict(passed=True, reviewer='actual pixel review in earlier run', notes='spatial proxy, not material topology',
            calibration_sha256=sha256(old/'calibration'/'calibration.json'), graph_array_sha256={}, figure_sha256={})
        for ep, start in ((2, 141), (7, 105)):
            name = f'graph_ep{ep}_start{start}.npz'
            payload = dict(initial_world=np.array([[0., 1., 0.], [np.nan]*3], dtype=np.float32),
                initial_normalized=np.array([[0., 1., 0.], [np.nan]*3], dtype=np.float32), mask=np.array([True, False]),
                edges=np.array([[0, 0]], dtype=np.int64), rest_lengths=np.array([.1], dtype=np.float32))
            np.savez_compressed(old/'calibration'/name, **payload); np.savez_compressed(new/name, **payload)
            review['graph_array_sha256'][name] = sha256(old/'calibration'/name)
            figure = f'initial_graph_ep{ep}_start{start}.png'; (old/'figures'/figure).write_bytes(b'prior-unit-test-picture')
            review['figure_sha256'][figure] = sha256(old/'figures'/figure)
        (old/'graph_review.json').write_text(json.dumps(review))
        return old, new

    def test_same_reviewed_graph_payload_with_invalid_nan_matches_without_rewriting_prior(self):
        old, new = self.make_review(); before = (old/'graph_review.json').read_bytes()
        proof = prepare.reviewed_graph_proof(old, new)
        self.assertTrue(proof['passed']); self.assertTrue(proof['exact_initial_graph_payload'])
        self.assertEqual(len(proof['arrays']), 2); self.assertEqual((old/'graph_review.json').read_bytes(), before)

    def test_changed_payload_or_reviewed_figure_blocks_gate(self):
        old, new = self.make_review(); file = new/'graph_ep2_start141.npz'
        with np.load(file) as loaded:payload = dict(loaded)
        payload['rest_lengths'] = payload['rest_lengths']*2; np.savez_compressed(file, **payload)
        with self.assertRaisesRegex(ValueError, 'differs'):prepare.reviewed_graph_proof(old, new)
        (old/'figures'/'initial_graph_ep2_start141.png').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'figure changed'):prepare.reviewed_graph_proof(old, new)

    def test_prepare_exact_order_four_updates_gate_and_no_parent_cuda_until_children_exit(self):
        args = argparse.Namespace(root='data', depth_root='depth', g0_reference='ref', common=str(self.root/'common'),
            c_output=str(self.root/'C'), d_output=str(self.root/'D'), reviewed_graphs=str(self.root/'reviewed'), expected_sha='a'*40)
        common = Path(args.common); sequence = []
        def one(a, module, extra, label):
            sequence.append(label)
            if module == 'train':
                file = common/'engineering'/'preflight.json'; file.parent.mkdir(parents=True)
                file.write_text(json.dumps(dict(passed=True, source_sha=a.expected_sha,
                    data=dict(manifest_hash='manifest', epoch_order_hashes=['order']*8), initialization=dict(shared_exact=True))))
            elif module == 'calibrate':
                out = common/'calibration'; out.mkdir(); (out/'calibration.json').write_text('{}')
            else:
                v = extra[extra.index('--variant')+1]; out = common/'smoke'/v; out.mkdir(parents=True)
                (out/'completion.json').write_text(json.dumps(dict(passed=True, source_sha=a.expected_sha, variant=v,
                    successful_updates=4, calibration_sha256=sha256(common/'calibration'/'calibration.json'))))
            return dict(command=[module, *extra], log=label)
        def env():sequence.append('parent-environment'); return dict(cuda='12.8')
        with (patch.object(prepare, 'source_guard', return_value=(args.expected_sha, {})),
              patch.object(prepare, 'resource_guard', return_value={'job_id':'40000'}),
              patch.object(prepare, 'environment', side_effect=env), patch.object(prepare, '_one', side_effect=one),
              patch.object(prepare, 'calibration_read', return_value={'source_sha':args.expected_sha}),
              patch.object(prepare, 'reviewed_graph_proof', return_value={'passed':True, 'exact_initial_graph_payload':True}),
              patch.object(prepare, 'gate_read') as gate):
            result = prepare.preparation(args)
        self.assertEqual(sequence, ['preflight', 'calibration', 'smoke-g5_arap', 'smoke-g5_arap_iso', 'parent-environment'])
        self.assertEqual(result['manifest_hash'], 'manifest'); self.assertEqual(len(result['smoke']), 2); gate.assert_called_once()
        self.assertTrue((common/'gate.json').exists()); self.assertTrue((common/'preparation_completion.json').exists())
        self.assertFalse(Path(args.c_output).exists()); self.assertFalse(Path(args.d_output).exists())
        with self.assertRaises(FileExistsError):prepare.preparation(args)


if __name__ == '__main__':
    unittest.main()

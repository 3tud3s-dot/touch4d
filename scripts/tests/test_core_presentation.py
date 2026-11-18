"""CPU source-acceptance fixtures; no model inference or research metrics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.visualization import pack_results as p
from scripts.visualization import build_demo as b


def modern_completion(windows=5):
    return dict(status='completed_requires_visual_review', windows=windows, models=2,
                optimizer_updates=0, normalizers_and_weights_unchanged=True,
                native_continuous_parity=True, input_and_checkpoint_sha256_unchanged=True)


class SourceAcceptanceTests(unittest.TestCase):
    def write(self, root, name, value):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def cache(self, root):
        """Temporary schema fixtures only: no metrics table or predictions."""
        train = dict(windows={'train': 140, 'val': 24}, model_count=11, optimizer_updates=0,
                     native_continuous_parity=True, input_and_checkpoint_sha256_unchanged=True)
        protocol = dict(cameras=list(p.CAMERAS), horizon=15, delta=1,
                        checkpoints={'g5': {'sha256': p.G5_SHA256}, 'soma': [{'sha256': p.SOMA_SHA256}]})
        train_protocol = dict(protocol, train_starts=list(range(113, 253)), holdout_starts=list(range(268, 292)))
        cross_protocol = dict(protocol, cases=[dict(episode=1, start=n) for n in [181, 260]])
        self.write(root, 'protocol.json', {'fixture': True})
        (root/'metrics.csv').write_bytes(b'NON_RESEARCH_SOURCE_GATE_FIXTURE\n')
        paths = {'source_proofs/train_completion.json': train,
                 'source_proofs/train_protocol.json': train_protocol,
                 'source_proofs/cross_completion.json': modern_completion(6),
                 'source_proofs/cross_protocol.json': cross_protocol}
        for name, value in paths.items(): self.write(root, name, value)
        paths = list(paths) + ['protocol.json', 'metrics.csv']
        manifest = dict(schema='touch4d-verified-cache-v1',
                        files_sha256={name: p.digest(root/name) for name in paths},
                        source_completion_proofs=[
                            dict(role='resimulation', schema='legacy-h15-v1', completion='source_proofs/train_completion.json', protocol='source_proofs/train_protocol.json'),
                            dict(role='generalization', schema='cross-h15-v3', completion='source_proofs/cross_completion.json', protocol='source_proofs/cross_protocol.json')])
        return self.write(root, 'cache_sources.json', manifest), manifest

    def test_fresh_requires_real_completion_file(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError): p.validate_source(name)

    def test_fresh_accepts_only_complete_successful_five_window_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = self.write(root, 'evaluation_complete.json', modern_completion())
            accepted = p.validate_source(root)
            self.assertTrue(accepted['fresh_inference'])
            self.assertEqual(accepted['completion']['sha256'], p.digest(path))
            self.assertEqual(accepted['completion']['record'], modern_completion())

    def test_missing_failed_or_nonzero_update_gate_is_rejected(self):
        variants = [dict(modern_completion(), status='failed'), dict(modern_completion(), optimizer_updates=1),
                    dict(modern_completion(), normalizers_and_weights_unchanged=False),
                    dict(modern_completion(), native_continuous_parity=1),
                    dict(modern_completion(), input_and_checkpoint_sha256_unchanged=False),
                    dict(modern_completion(), windows=4)]
        missing = modern_completion(); missing.pop('normalizers_and_weights_unchanged'); variants.append(missing)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for record in variants:
                self.write(root, 'evaluation_complete.json', record)
                with self.assertRaises(ValueError): p.validate_source(root)

    def test_cache_manifest_is_not_an_implicit_fresh_completion(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); self.cache(root)
            with self.assertRaises(FileNotFoundError): p.validate_source(root)

    def test_explicit_cache_preserves_original_legacy_fields(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); manifest, _ = self.cache(root)
            accepted = p.validate_source(root, manifest)
            self.assertFalse(accepted['fresh_inference'])
            self.assertEqual(accepted['model_updates'], 0)
            old = accepted['source_completion_proofs'][0]['completion']['record']
            self.assertNotIn('status', old)
            self.assertNotIn('normalizers_and_weights_unchanged', old)
            self.assertFalse((root/'evaluation_complete.json').exists())

    def test_cache_tampering_or_missing_consumed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, _ = self.cache(root)
            with self.assertRaisesRegex(ValueError, 'omits'):
                p.validate_source(root, path, ['comparison_assets/missing.webp'])
            (root/'metrics.csv').write_bytes(b'TAMPERED_FIXTURE\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                p.validate_source(root, path)

    def test_cache_requires_both_original_proof_roles(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, data = self.cache(root)
            data['source_completion_proofs'][1]['role'] = 'resimulation'
            self.write(root, path.name, data)
            with self.assertRaisesRegex(ValueError, 'roles'): p.validate_source(root, path)

    def test_wrong_source_checkpoint_identity_rejected_even_with_matching_cache_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, data = self.cache(root)
            proof_path = root/'source_proofs/train_protocol.json'
            protocol = json.loads(proof_path.read_text()); protocol['checkpoints']['g5']['sha256'] = 'a'*64
            self.write(root, 'source_proofs/train_protocol.json', protocol)
            data['files_sha256']['source_proofs/train_protocol.json'] = p.digest(proof_path)
            self.write(root, path.name, data)
            with self.assertRaisesRegex(ValueError, 'checkpoints'): p.validate_source(root, path)

    def test_cache_cannot_escape_its_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, data = self.cache(root)
            data['files_sha256']['../outside'] = 'a'*64
            self.write(root, path.name, data)
            with self.assertRaisesRegex(ValueError, 'escapes'): p.validate_source(root, path)

    def test_exact_input_closure_is_required_for_real_presentation(self):
        required = p.presentation_inputs()
        self.assertEqual(len([x for x in required if x.endswith('.webp')]), 180)
        self.assertEqual(len([x for x in required if x.endswith('.npz')]), 2)
        self.assertTrue({'protocol.json', 'metrics.csv'} <= required)


class PredictionIdentityTests(unittest.TestCase):
    def records(self):
        observed = dict(xyz=np.zeros((2, 3), np.float32), log_scales=np.full((2, 3), -3., np.float32),
                        quaternions=np.tile(np.array([[1., 0, 0, 0]], np.float32), (2, 1)),
                        opacity_logits=np.full((2, 1), 100., np.float32), sh_dc=np.full((2, 3), 10., np.float32))
        predicted = {key: value.copy() for key, value in observed.items()}
        predicted.update(frame_id=np.array(225), g5_xyz=np.zeros((15, 2, 3), np.float32))
        return observed, predicted

    def test_exact_raw_initial_values_pass(self):
        observed, predicted = self.records(); p.validate_prediction_initial(predicted, observed, 225)

    def test_wrong_noninteger_or_nonscalar_frame_id_rejected(self):
        for frame in [np.array(224), np.array([225]), np.array(225.8)]:
            observed, predicted = self.records(); predicted['frame_id'] = frame
            with self.assertRaisesRegex(ValueError, 'frame_id'): p.validate_prediction_initial(predicted, observed, 225)

    def test_raw_attributes_checked_before_saturation_or_activation(self):
        for key in p.INITIAL_ATTRIBUTES:
            observed, predicted = self.records(); predicted[key].flat[0] += 1
            with self.assertRaisesRegex(ValueError, 'raw attribute'): p.validate_prediction_initial(predicted, observed, 225)

    def test_initial_dtype_and_prediction_track_shape_rejected(self):
        observed, predicted = self.records(); predicted['sh_dc'] = predicted['sh_dc'].astype(np.float64)
        with self.assertRaises(ValueError): p.validate_prediction_initial(predicted, observed, 225)
        observed, predicted = self.records(); predicted['g5_xyz'] = np.zeros((14, 2, 3), np.float32)
        with self.assertRaisesRegex(ValueError, '15-step'): p.validate_prediction_initial(predicted, observed, 225)


class HtmlByteIdentityTests(unittest.TestCase):
    def test_utf8_crlf_canonicalization_is_independent_of_source_platform(self):
        expected = '<html>\r\n布料\r\n<script>\r\n</script>\r\n</html>\r\n'.encode('utf-8')
        for newline in ('\n', '\r\n', '\r'):
            source = newline.join(['<html>', '布料', '<script>', '</script>', '</html>', ''])
            self.assertEqual(b.canonical_html_bytes(source), expected)

    def test_mixed_newlines_do_not_introduce_double_carriage_returns(self):
        source = 'a\r\nb\nc\r'
        canonical = b.canonical_html_bytes(source)
        self.assertEqual(canonical, b'a\r\nb\r\nc\r\n')
        self.assertNotIn(b'\r\r\n', canonical)
        self.assertEqual(b.canonical_html_bytes(canonical.decode('utf-8')), canonical)


if __name__ == '__main__':
    unittest.main()

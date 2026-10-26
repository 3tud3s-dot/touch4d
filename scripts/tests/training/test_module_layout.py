"""Regression coverage for public paths and migrated experiment entry points."""

import importlib
import pickle
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
ALIASES = {
    'scripts.training.fixed_fit': 'scripts.experiments.fixed_window.fixed_fit',
    'scripts.training.oracle': 'scripts.experiments.oracle.oracle',
    'scripts.training.oracle_fit': 'scripts.experiments.oracle.oracle_fit',
}


class ModuleLayoutTests(unittest.TestCase):
    def test_legacy_aliases_share_globals_and_patch_targets(self):
        for old, new in ALIASES.items():
            legacy = importlib.import_module(old)
            canonical = importlib.import_module(new)
            self.assertIs(legacy, canonical)
        legacy = importlib.import_module('scripts.training.fixed_fit')
        canonical = importlib.import_module(ALIASES['scripts.training.fixed_fit'])
        with patch('scripts.training.fixed_fit.atomic_checkpoint') as save:
            canonical.atomic_checkpoint('sentinel')
            save.assert_called_once_with('sentinel')
        self.assertIs(legacy.PositionPredictor,
                      importlib.import_module('scripts.training.model').PositionPredictor)

    def test_old_pickle_class_paths_resolve(self):
        # Trusted synthetic pickle GLOBAL records, not untrusted checkpoint data.
        for old, name in (('scripts.training.oracle', 'OracleLatent'),
                          ('scripts.training.oracle', 'OracleAnchor'),
                          ('scripts.training.fixed_fit', 'TargetCache'),
                          ('scripts.training.model', 'PositionPredictor')):
            value = pickle.loads(f'c{old}\n{name}\n.'.encode())
            self.assertIs(value, getattr(importlib.import_module(old), name))
        from scripts.training.model import PositionPredictor
        self.assertEqual(PositionPredictor.__module__, 'scripts.training.model')

    def test_source_inventory_includes_moved_implementations(self):
        from scripts.experiments.fixed_window import provenance as p
        files = p.source_files(ROOT)
        for name in ('fixed_window/fixed_fit.py', 'fixed_window/provenance.py',
                     'oracle/oracle.py', 'oracle/oracle_fit.py'):
            self.assertIn('scripts/experiments/'+name, files)
        self.assertFalse(any('/tests/' in name for name in files))
        self.assertEqual(p.inventory_digest(p.normalized_inventory(ROOT)), p.MIGRATED_FILES_SHA256)

    def test_source_migration_is_narrow_and_preserves_config_checks(self):
        from scripts.experiments.fixed_window import provenance as p
        from scripts.experiments.fixed_window.fixed_fit import digest
        current = dict(source_sha256=digest(p.source_files(ROOT)), horizon=15, gs_structure='anchor_attention')
        self.assertTrue(p.resume_config_matches(current, current, ROOT))
        for known in p.LEGACY_SOURCE_HASHES:
            old = dict(current, source_sha256=known)
            self.assertTrue(p.resume_config_matches(old, current, ROOT))
            for changed in (dict(current, horizon=5), dict(current, gs_structure='none')):
                self.assertFalse(p.resume_config_matches(old, changed, ROOT))
            inventory = dict(p.normalized_inventory(ROOT), **{'scripts/training/model.py': 'changed'})
            with patch.object(p, 'normalized_inventory', return_value=inventory):
                self.assertFalse(p.resume_config_matches(old, current, ROOT))
        self.assertFalse(p.resume_config_matches(dict(current, source_sha256='unreviewed'), current, ROOT))

    def test_fresh_process_import_orders(self):
        orders = [list(ALIASES), list(reversed(list(ALIASES.values()))),
                  ['scripts.training.evaluate', 'scripts.training.train', *ALIASES.values()]]
        for order in orders:
            code = 'import importlib; ' + '; '.join(
                f'importlib.import_module({name!r})' for name in order)
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_and_old_new_oracle_cli_help(self):
        modules = ('scripts.training.train', 'scripts.training.evaluate',
                   'scripts.training.visualize', 'scripts.training.oracle_fit',
                   'scripts.experiments.oracle.oracle_fit')
        for module in modules:
            result = subprocess.run([sys.executable, '-B', '-m', module, '--help'],
                                    cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertIn('--help', result.stdout)

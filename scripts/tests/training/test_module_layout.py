"""Retained public model path and current core command imports."""
import importlib
import pickle
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]


class ModuleLayoutTests(unittest.TestCase):
    def test_position_predictor_import_and_pickle_path(self):
        from scripts.training.model import PositionPredictor
        self.assertEqual(PositionPredictor.__module__, 'scripts.training.model')
        self.assertIs(pickle.loads(b'cscripts.training.model\nPositionPredictor\n.'), PositionPredictor)

    def test_current_core_imports(self):
        for module in ('scripts.training.cloth_model', 'scripts.training.cloth_train',
                       'scripts.training.cloth_evaluate', 'scripts.training.cloth_data'):
            importlib.import_module(module)

    def test_current_cli_help_has_no_training_side_effect(self):
        for module in ('scripts.training.cloth_train', 'scripts.training.cloth_evaluate', 'scripts.training.cloth_data'):
            result = subprocess.run([sys.executable, '-B', '-m', module, '--help'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()

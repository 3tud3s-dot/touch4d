"""Offline report assembly rejects missing evidence and refuses overwrites."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.experiments.debug_audit.report import assemble, csv_rows, main, render


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    metric = dict(cd_mm=[1.]*5, rgb=[.1]*5, alpha=[.2]*5, mean_cd_mm=1.,
                  mean_rgb=.1, mean_alpha=.2,
                  motion=[dict(displacement_rms_mm=.3, gt_centroid_motion_mm=[1., 0., 0.],
                               centroid_error_mm=.7, centered_cd_mm=.8)]*5)
    evaluation = {name: copy.deepcopy(metric) for name in ("model", "copy_initial", "freeze_action", "zero_tactile")}
    for name in ("freeze_action", "zero_tactile"):
        evaluation[name]["delta_cd_mm"] = [.01]*5
    evidence = dict(folder="evidence", horizon=5, cameras=3, frame_ids=list(range(73, 79)))
    for stage in ("diagnosis", "E1", "E2"):
        write(root/stage/"status.json", dict(status="complete", source_sha="test", job_id="test"))
        folder = root/stage/"evidence"; folder.mkdir()
        for h in range(1, 6):
            for camera in range(3):
                for mode in ("rgb", "alpha"):
                    for label in ("gt", "pred", "copy"):
                        (folder/f"h{h:02d}-c{camera}-{mode}-{label}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        initial = np.zeros((3, 3), dtype=np.float32)
        predictions = np.stack([initial + h*.001 for h in range(1, 6)])
        np.savez(folder/"geometry.npz", initial_xyz_m=initial, prediction_xyz_m=predictions,
                 frame_ids=np.arange(73, 79), **{f"gt_h{h}_xyz_m": predictions[h-1] for h in range(1, 6)})
    record = dict(label="S0-last", split="train", episode=2, start=73, completed_updates=1200,
                  metrics=evaluation, evidence=evidence, checkpoint="old.pt")
    write(root/"diagnosis/checkpoint_results.json", [record])
    write(root/"diagnosis/gradient_audit.json", {})
    for name in ("before", "after"):
        write(root/f"diagnosis/historical_hashes_{name}.json", {"old.pt": "abc"})
    oracle = dict(final=metric, baselines=dict(copy_initial=metric, centroid_translation=metric),
                  visuals=dict(final=evidence, centroid=evidence), config={},
                  milestones={str(k): metric for k in (0, 100, 200, 400)})
    write(root/"diagnosis/oracle_results.json", {"cd": oracle, "full": oracle})
    for mode in ("E1", "E2"):
        write(root/mode/"config.json", dict(initial_state_sha256={"p": "abc"}))
        write(root/mode/"updates.json", [dict(update=i, loss=1., geometry=1.) for i in range(1, 401)])
        write(root/mode/"parameter_updates.json", [])
        write(root/mode/"evaluations.json", {str(i): dict(metrics=evaluation, evidence=evidence) for i in (0, 100, 200, 400)})


class DebugReportTests(unittest.TestCase):
    def test_complete_offline_evidence_assembled_without_claimed_conclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            result = assemble(root)
            self.assertEqual(len(result["rows"]), 12)
            self.assertEqual(result["image_files_checked"], 270)
            self.assertTrue(result["paired_initial_parameters_exact"])
            self.assertIsNone(result["conclusion"])
            self.assertTrue(all(row["geometry"]["unique_prediction_states"] == 5 for row in result["rows"]))
            self.assertEqual(len(list(csv_rows(result))), 60)
            page = render(result)
            self.assertNotIn("https://", page)
            self.assertNotIn("__PAYLOAD__", page)

    def test_missing_image_and_missing_milestone_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            target = root/"E1/evidence/h05-c2-alpha-pred.png"
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                assemble(root)
            target.write_bytes(b"\x89PNG\r\n\x1a\n")
            path = root/"E1/evaluations.json"
            value = json.loads(path.read_text()); value.pop("400"); write(path, value)
            with self.assertRaisesRegex(ValueError, "milestones"):
                assemble(root)

    def test_parameter_drift_and_historical_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            write(root/"E1/config.json", dict(initial_state_sha256={"p": "different"}))
            with self.assertRaisesRegex(ValueError, "parameter hashes"):
                assemble(root)
            write(root/"E1/config.json", dict(initial_state_sha256={"p": "abc"}))
            write(root/"diagnosis/historical_hashes_after.json", {"old.pt": "different"})
            with self.assertRaisesRegex(ValueError, "checkpoint hashes"):
                assemble(root)

    def test_output_never_overwrites_and_embedded_data_escapes_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"report.html"; path.write_text("protected")
            with self.assertRaises(FileExistsError):
                main(["--root", str(root)])
            self.assertEqual(path.read_text(), "protected")
        page = render({"untrusted": "</script><script>alert(1)</script>"})
        self.assertNotIn('"untrusted":"</script>', page)
        self.assertEqual(page.count("</script>"), 1)


if __name__ == "__main__":
    unittest.main()

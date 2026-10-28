import unittest
import numpy as np
from scripts.experiments.v4_motion_field.metrics import vector_comparison,neighbor_metrics


class MotionFieldTests(unittest.TestCase):
    def test_zero_undefined(self):
        r,p=vector_comparison(np.zeros((3,3)),np.zeros((3,3)))
        self.assertEqual(r['undefined_direction_count'],3)
        self.assertIsNone(r['direction_cosine_mean']);self.assertTrue(np.isnan(p['angle_deg']).all())

    def test_translation_residual(self):
        a=np.tile([.01,0,0],(4,1));b=np.tile([.02,0,0],(4,1))
        r,_=vector_comparison(a,b)
        self.assertAlmostEqual(r['difference_rms_mm'],10)
        self.assertAlmostEqual(r['residual_difference_rms_mm'],0)
        self.assertAlmostEqual(r['magnitude_error_energy_fraction'],1)

    def test_opposite_direction(self):
        a=np.tile([.01,0,0],(4,1));r,_=vector_comparison(a,-a)
        self.assertAlmostEqual(r['direction_angle_median_deg'],180)
        self.assertAlmostEqual(r['direction_error_energy_fraction'],1)

    def test_decomposition_and_translation(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(30,3))*.01;b=rng.normal(size=(30,3))*.01
        r,_=vector_comparison(a,b)
        self.assertAlmostEqual(r['direction_error_energy_fraction']+r['magnitude_error_energy_fraction'],1)
        self.assertAlmostEqual(r['difference_rms_mm']**2,r['mean_translation_difference_mm']**2+r['residual_difference_rms_mm']**2)

    def test_neighbors_translation_and_stretch(self):
        x=np.array([[0.,0,0],[1.,0,0],[2.,0,0]]);ids=np.array([[1],[0],[1]]);valid=np.ones((3,1),bool)
        r=neighbor_metrics(x,np.ones_like(x),ids,valid)
        self.assertEqual(r['distortion_p95'],0);self.assertEqual(r['neighbor_motion_difference_rms_mm'],0)
        self.assertEqual(neighbor_metrics(x,x,ids,valid)['ratio_p95'],2)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):vector_comparison(np.ones((2,3))*np.nan,np.ones((2,3)))

    def test_rbf_control_not_actual_anchor_motion(self):
        w=np.array([[.75,.25],[.25,.75]]);d=np.array([[1.,0,0],[-1.,0,0]])
        self.assertFalse(np.array_equal(w@d,d))
        self.assertTrue(np.allclose(w@np.ones((2,3)),1))


if __name__=='__main__':unittest.main()

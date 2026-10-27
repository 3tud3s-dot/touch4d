import unittest
import numpy as np
from scripts.experiments.contact_audit.audit import nearest, tactile_stats, runs, windows, project
from scripts.experiments.contact_audit.report import infer


class ContactAuditTest(unittest.TestCase):
    def test_valid_zero_not_missing(self):
        x = np.zeros((12,32))
        a = tactile_stats(x, np.ones_like(x,bool))
        b = tactile_stats(x, np.zeros_like(x,bool))
        self.assertEqual(a['mean'], 0)
        self.assertEqual(a['valid_count'], 384)
        self.assertIsNone(b['mean'])

    def test_mask_excludes_pressure(self):
        self.assertEqual(tactile_stats(np.array([0.,999.]),np.array([True,False]))['maximum'],0)

    def test_nearest_units_and_no_identity(self):
        x=np.array([[0.,0,0],[1,0,0]])
        np.testing.assert_allclose(nearest([[.003,0,0]],x[::-1]),[.003])

    def test_invalid_nn_rejected(self):
        with self.assertRaises(ValueError): nearest([[np.nan,0,0]],[[0,0,0]])

    def test_closed_intervals(self):
        self.assertEqual(list(runs(['A','A','B','A'])),[(0,1,'A'),(2,2,'B'),(3,3,'A')])

    def test_h10_all_eleven_frames(self):
        self.assertEqual(windows(['CONTACT']*11)['counts']['all_candidate_contact'],1)
        self.assertEqual(windows(['CONTACT']*10)['counts']['all_candidate_contact'],0)
        w=windows(['CONTACT']*10+['UNCERTAIN'])
        self.assertEqual(w['counts']['all_candidate_contact'],0)
        self.assertEqual(w['confirmed_contact_windows'],0)
        self.assertTrue(w['windows'][0]['includes_uncertain'])

    def test_no_bridge_uncertain(self):
        w=windows(['CONTACT']*12+['UNCERTAIN']+['CONTACT']*12)
        self.assertEqual(w['counts']['all_candidate_contact'],4)

    def test_camera_inverse_and_behind(self):
        pose=np.eye(4);pose[0,3]=1
        uv,valid=project([[1,0,2],[1,0,-2]],np.diag([100.,100.,1.]),pose)
        np.testing.assert_allclose(uv[0],[0,0]);self.assertFalse(valid[1])

    def test_candidate_noise_missing_and_persistence(self):
        pressure=np.zeros((30,2,12,32));mask=np.ones_like(pressure,bool)
        pressure[10:20,:,:2,:2]=100
        mask[15,0]=False
        pressure[15,0]=0
        rows=[]
        for f in range(30):
            row=dict(controller_min_mm=100 if f<10 or f>=20 else 5,
                     right_pose_valid=True,controller_valid_count=30)
            for j in range(2):
                s=tactile_stats(pressure[f,j],mask[f,j])
                row[f'finger{j}_mean']=s['mean'];row[f'finger{j}_maximum']=s['maximum']
            rows.append(row)
        r=infer(rows,dict(tactile=pressure,valid_mask=mask))
        self.assertEqual(r['labels'][10],'CONTACT')
        self.assertEqual(r['labels'][15],'UNCERTAIN')
        self.assertEqual(r['labels'][0],'NO_CONTACT')
        self.assertEqual(r['labels'][29],'NO_CONTACT')
        # A short isolated peak cannot become sustained CONTACT.
        pressure[10:20]=0;pressure[12,:,:2,:2]=100
        for f in range(30):
            for j in range(2):
                s=tactile_stats(pressure[f,j],mask[f,j])
                rows[f][f'finger{j}_mean']=s['mean'];rows[f][f'finger{j}_maximum']=s['maximum']
        self.assertNotIn('CONTACT',infer(rows,dict(tactile=pressure,valid_mask=mask))['labels'])


if __name__=='__main__': unittest.main()

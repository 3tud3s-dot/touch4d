import copy
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch,add_depth_targets
from scripts.training.model import PositionPredictor
from scripts.training.losses import chamfer_samples
from scripts.experiments.condition_identifiability.conditions import ToolPredictor,sanitize,hand_inputs
from scripts.experiments.condition_identifiability.common import ARCH,synthetic_pair,intervene,state_hash,VERSION
from scripts.experiments.condition_identifiability.run import load_checkpoint

def fixture():
    b=DataProcessor('001-rope')(add_depth_targets(make_batch()))
    b['inputs']={k:v[:1].clone() for k,v in b['inputs'].items()}
    b['targets']={k:([x[:1].clone() for x in v] if isinstance(v,list) else v[:1].clone()) for k,v in b['targets'].items()}
    b['meta']['frame_ids']=b['meta']['frame_ids'][:1];b['meta']['episode_id']=torch.tensor([2]);b['meta']['scene']=['001-rope']
    b['inputs']=sanitize(hand_inputs(b['inputs'],torch.tensor([2]),['001-rope']))
    return b

def model():
    return ToolPredictor(**dict(ARCH,width=16,anchor_count=3,anchor_pool_k=2,gs_local_k=2))

class IdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_uses_exact_p2_forward(self):
        self.assertIs(ToolPredictor.forward,PositionPredictor.forward)

    def test_synthetic_only_action_differs_and_target_sign(self):
        b=fixture();saved=copy.deepcopy(b);scale=DataProcessor('001-rope').state_dict()['scale']
        a,c=synthetic_pair(b,scale)
        for k in a['inputs']:
            if k not in ('action_state','controller_points'):
                self.assertTrue(torch.equal(a['inputs'][k],c['inputs'][k]),k)
        for k in b['inputs']:self.assertTrue(torch.equal(b['inputs'][k],saved['inputs'][k]))
        for item,sign in ((a,-1),(c,1)):
            for h,t in enumerate(item['targets']['target_xyz']):
                mask=item['inputs']['gs_mask'];delta=(t-item['inputs']['gs_xyz'])[mask]*scale*1000
                torch.testing.assert_close(delta[:,2],torch.full_like(delta[:,2],sign*(h+1)*2),atol=1e-4,rtol=1e-4)
                self.assertEqual(float(delta[:,:2].abs().max()),0.)

    def test_hand_gate_prevents_bias_fallback_and_nan(self):
        b=fixture();x=b['inputs'];x['tactile'][:,:,0]=float('nan');x['action_state'][:,:,0]=float('nan')
        x=sanitize(x);m=model();z,k=m.encode_conditions(x)
        self.assertTrue(torch.isfinite(z).all());self.assertFalse(k[:,:,0].any());self.assertFalse(k[:,:,2:32].any());self.assertFalse(k[:,:,62:64].any())

    def test_local_mean_max_valid_zero_and_no_cross_finger(self):
        x=fixture()['inputs'];x['valid_mask'][:]=True;x['tactile_spatial_mask'][:]=True;x['tactile_age_us'][:]=0;x['tactile'][:]=0
        x['tactile'][:,:,2]=3.;x=sanitize(x);a=model().aggregate(x)
        torch.testing.assert_close(a[0][:,:,2],torch.full_like(a[0][:,:,2],3.))
        self.assertTrue(a[4][:,:,3].all());self.assertEqual(float(a[0][:,:,3].abs().max()),0.)
        self.assertFalse(a[4][:,:,:2].any())

    def test_swapped_synthetic_action_exactly_matches_other_input(self):
        a,b=synthetic_pair(fixture(),.15703709051)
        x=intervene(a['inputs'],'swap_action',b['inputs'])
        for k in x:self.assertTrue(torch.equal(x[k],b['inputs'][k]),k)

    def test_freeze_causal_and_zero_preserves_validity(self):
        x=fixture()['inputs'];f=intervene(x,'freeze_action');z=intervene(x,'zero_tactile')
        for k in ('action_state','controller_points'):
            self.assertTrue(torch.equal(f[k],x[k][:,:1].expand_as(x[k])))
        self.assertTrue(torch.equal(z['valid_mask'],x['valid_mask']))

    def test_future_conditions_and_targets_not_leaked(self):
        b=fixture();m=model();torch.nn.init.normal_(m.position_head[-1].weight,std=.01)
        a=m(b['inputs']);x=copy.deepcopy(b['inputs']);x['controller_points'][:,3:]+=3;x['action_state'][:,3:]+=2
        c=m(x)
        for k in a:torch.testing.assert_close(a[k][:,:3],c[k][:,:3],atol=0,rtol=0)
        b['targets']['target_xyz'][0]+=100
        c=m(b['inputs'])
        for k in a:torch.testing.assert_close(a[k],c[k],atol=0,rtol=0)

    def test_cross_step_and_condition_gradients(self):
        b=fixture();m=model();torch.nn.init.normal_(m.position_head[-1].weight,std=.02);torch.nn.init.normal_(m.fusion[-1].weight,std=.02)
        outputs=[]
        def capture(mod,inp,out):out.retain_grad();outputs.append(out)
        hook=m.position_head.register_forward_hook(capture)
        pred=m(b['inputs']);pred['xyz'][:,4].square().sum().backward();hook.remove()
        self.assertGreater(float(outputs[0].grad.abs().sum()),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.controller_encoder.parameters()),0.)

    def test_initial_copy_and_seed_pairing(self):
        b=fixture();torch.manual_seed(42);m=model();torch.manual_seed(42);n=model()
        self.assertEqual(state_hash(m),state_hash(n))
        pred=m(b['inputs'])['xyz'];expected=b['inputs']['gs_xyz'][:,None].expand_as(pred)
        self.assertTrue(torch.equal(pred,expected))

    def test_checkpoint_mode_rejected_and_exact_restore(self):
        m=model();opt=torch.optim.AdamW(m.parameters());before=state_hash(m)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'check.pt';torch.save(dict(version=VERSION,mode='real',config=m.config,model=m.state_dict(),optimizer=opt.state_dict()),p)
            with self.assertRaises(ValueError):load_checkpoint(p,m,opt,'synthetic')
            load_checkpoint(p,m,opt,'real');self.assertEqual(before,state_hash(m))

if __name__=='__main__':unittest.main()

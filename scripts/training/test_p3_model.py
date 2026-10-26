import copy
import unittest
from unittest.mock import patch
import torch
from .p3_model import P3Predictor
from .model import PositionPredictor
from .test_horizon import cast_double
from ..data_process.test_data_process import make_batch,add_depth_targets
from ..data_process import DataProcessor


def sample_inputs(h=10):
    batch=DataProcessor('001-rope')(add_depth_targets(make_batch(horizon=h)))
    x=batch['inputs']; b=x['gs_xyz'].shape[0]
    x['gs_prev_xyz']=torch.randn(b,5,3)*.1
    x['gs_prev_attr']=torch.zeros(b,5,11);x['gs_prev_attr'][...,:3]=-4;x['gs_prev_attr'][...,3]=1
    x['gs_prev_mask']=torch.ones(b,5,dtype=torch.bool);x['gs_prev_mask'][1,-2:]=False
    return batch


def small(history=2,refine=False,horizon=10):
    torch.manual_seed(42)
    return P3Predictor(history=history,template_refinement=refine,width=32,heads=4,layers=2,
                      gs_local_k=4,anchor_count=8,anchor_pool_k=4,horizon=horizon)


class P3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_shared_init_nonzero_motion_and_zero_extra_branches(self):
        batch=sample_inputs();x=batch['inputs']
        a,b,c=small(1),small(2),small(2,True)
        for key,v in a.state_dict().items():
            torch.testing.assert_close(v,b.state_dict()[key],atol=0,rtol=0)
        for key,v in b.state_dict().items():
            torch.testing.assert_close(v,c.state_dict()[key],atol=0,rtol=0)
        with torch.no_grad():
            for model in (a,b,c):model.position_head[-1].bias.copy_(torch.tensor([.01,-.02,.03]))
        outputs=[model(x) for model in (a,b,c)]
        for key in outputs[0]:
            for out in outputs[1:]:torch.testing.assert_close(out[key],outputs[0][key],atol=0,rtol=0)
        expected=x['gs_xyz'][:,None]+torch.arange(1,11)[None,:,None,None]*torch.tensor([.01,-.02,.03])
        expected=torch.where(x['gs_mask'][:,None,:,None],expected,0.)
        torch.testing.assert_close(outputs[0]['xyz'],expected,atol=2e-6,rtol=1e-6)

    def test_prefix_causality_target_independence_padding_and_fixed_geometry(self):
        x=sample_inputs()['inputs'];model=small(2,True)
        torch.nn.init.normal_(model.position_head[-1].weight,std=.02)
        torch.nn.init.normal_(model.temporal.output.weight,std=.02)
        with patch.object(model,'_geometry',wraps=model._geometry) as geometry:
            baseline=model(x)
            self.assertEqual(geometry.call_count,2)
        changed=copy.deepcopy(x)
        for key in ('action_state','controller_points','tactile'):changed[key][:,5:]+=1
        changed['target_xyz']=torch.full((4,99,3),float('nan'))
        altered=model(changed)
        for key in baseline:torch.testing.assert_close(baseline[key][:,:5],altered[key][:,:5],atol=0,rtol=0)
        self.assertFalse(torch.equal(baseline['xyz'][:,5:],altered['xyz'][:,5:]))
        for prefix in ('gs','gs_prev'):
            changed=copy.deepcopy(x)
            for suffix in ('xyz','attr'):changed[prefix+'_'+suffix][~x[prefix+'_mask']]=float('nan')
            actual=model(changed)
            for key in baseline:torch.testing.assert_close(baseline[key],actual[key],atol=0,rtol=0)

    def test_full_chain_history_and_attribute_feedback(self):
        x=cast_double(sample_inputs()['inputs']);model=small(2,True).double()
        with torch.no_grad():
            for head in (model.position_head[-1],model.temporal.output,model.refiner.scale_head[-1],model.refiner.opacity_head[-1]):
                head.weight.normal_(std=.02)
        x['gs_prev_xyz'].requires_grad_();x['gs_prev_attr'].requires_grad_()
        out,states=model(x,return_states=True)
        grad=torch.autograd.grad(out['xyz'][:,-1].square().sum(),
            [s['xyz'] for s in states[:-1]]+[states[0]['attr'],x['gs_prev_xyz'],x['gs_prev_attr']],retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grad))
        self.assertTrue((grad[-3][...,:3].abs().sum()>0) and (grad[-3][...,7].abs().sum()>0))
        self.assertFalse(torch.equal(out['xyz'][:,4],out['xyz'][:,9]))
        self.assertLessEqual(float((out['log_scales']-x['gs_attr'][:,None,:,:3]).abs().max()),.262365)
        self.assertLessEqual(float((out['opacity_logits']-x['gs_attr'][:,None,:,7:8]).abs().max()),.500001)

    def test_previous_order_is_not_correspondence(self):
        x=sample_inputs()['inputs'];model=small(2)
        torch.nn.init.normal_(model.position_head[-1].weight,std=.01)
        torch.nn.init.normal_(model.temporal.output.weight,std=.02)
        out=model(x);changed=copy.deepcopy(x);order=torch.tensor([4,2,0,3,1])
        for key in ('gs_prev_xyz','gs_prev_attr','gs_prev_mask'):changed[key]=changed[key][:,order]
        actual=model(changed)
        for key in out:torch.testing.assert_close(out[key],actual[key],atol=3e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()

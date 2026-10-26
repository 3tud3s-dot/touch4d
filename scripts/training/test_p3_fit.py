"""P3 checkpoint mode protection and exact optimizer continuation, disk-free."""
import copy
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import torch
from .test_p3_model import small,sample_inputs
from .p3_fit import checkpoint,read_state,restore,recover_best,remaining_updates,construct
from .fixed_fit import digest


class P3FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def state(self,model,n=19,history=None,best=None):
        cfg=dict(model.config,scene='001-rope',variant='M3')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        return opt,cfg,copy.deepcopy(checkpoint(model,opt,cfg,[],n,history or [],best,'test',dict(seconds=0.)))

    def test_resume_optimizer_next_step_and_wrong_modes(self):
        x=sample_inputs()['inputs'];model=small(2,True);opt,cfg,_=self.state(model)
        def step(m,o):
            o.zero_grad(set_to_none=True);out=m(x)
            loss=out['xyz'].square().mean()+out['log_scales'].square().mean()+out['opacity_logits'].square().mean()
            loss.backward();o.step()
        step(model,opt)
        state=copy.deepcopy(checkpoint(model,opt,cfg,[],1,[],None,'test',dict(seconds=0.)))
        restored=construct(cfg);ropt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
        restore(state,restored,ropt,cfg)
        step(model,opt);step(restored,ropt)
        self.assertEqual(digest(model.state_dict()),digest(restored.state_dict()))
        self.assertEqual(digest(opt.state_dict()),digest(ropt.state_dict()))
        for key,value in [('horizon',5),('history',1),('template_refinement',False),('variant','M1')]:
            bad=dict(cfg,**{key:value})
            with self.assertRaises(ValueError):restore(state,restored,ropt,bad)
        self.assertEqual(list(remaining_updates(399)),[400]);self.assertEqual(list(remaining_updates(400)),[])
        self.assertEqual(remaining_updates(17).start,18)

    def test_pending_evaluation_and_incomplete_config_fail_closed(self):
        model=small(2,True);opt,cfg,state=self.state(model,n=20)
        with patch('torch.load',return_value=state):
            self.assertEqual(read_state('unused',cfg)['completed_updates'],20)
        bad=copy.deepcopy(state);bad['completed_updates']=39 # missing evaluation at20
        with patch('torch.load',return_value=bad):
            with self.assertRaises(ValueError):read_state('unused')
        bad=copy.deepcopy(state);bad['config'].pop('history')
        with patch('torch.load',return_value=bad):
            with self.assertRaises(ValueError):read_state('unused')

    def test_newer_better_checkpoint_metadata_is_not_overwritten(self):
        model=small(2,True);old=dict(update=20,rank=[2.,2.,.1]);new=dict(update=40,rank=[1.,1.,.05])
        _,cfg,state=self.state(model,n=40,history=[dict(update=20)],best=old)
        saved=copy.deepcopy(state);saved.update(best_candidate=new,history=[dict(update=20),dict(update=40)])
        with patch.object(Path,'is_file',return_value=True),patch('scripts.training.p3_fit.read_state',return_value=saved):
            self.assertEqual(recover_best('unused',state),new)
            state['completed_updates']=39
            with self.assertRaises(ValueError):recover_best('unused',state)
        with patch.object(Path,'is_file',return_value=False):
            with self.assertRaises(ValueError):recover_best('unused',state)

    def test_old_P2_structural_forward_exact(self):
        import subprocess
        from .gs_structure import StructuralGSBlock
        from .gs_decoder import build_initial_anchor_field
        code=subprocess.check_output(['git','show','fdd4013:scripts/training/gs_structure.py'],text=True)
        namespace={'__name__':'scripts.training._old_structure','__package__':'scripts.training'}
        exec(compile(code,'fdd4013/gs_structure.py','exec'),namespace)
        old=namespace['StructuralGSBlock'](32,4);new=StructuralGSBlock(32,4)
        with torch.no_grad():old.output.weight.normal_(std=.1)
        new.load_state_dict(old.state_dict())
        xyz=torch.randn(2,13,3);keep=torch.ones(2,13,dtype=torch.bool);keep[1,-5:]=False
        features=torch.randn(2,13,32);field=build_initial_anchor_field(xyz,keep,8,4)
        torch.testing.assert_close(old(features,xyz,keep,field),new(features,xyz,keep,field),atol=0,rtol=0)

if __name__=='__main__':unittest.main()

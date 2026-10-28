import ast
import copy
import inspect
from pathlib import Path
import subprocess
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.v4_model import V4Stage1Predictor, fixed_field
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.fixed_window.fixed_fit import digest
from scripts.experiments.v4_h15_condition_ablation import interventions as iv
from scripts.experiments.v4_h15_condition_ablation.run import geometric_metrics

class FrozenH15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        torch.manual_seed(42)
        self.m=V4Stage1Predictor(horizon=15,prediction_interval=1).eval()
        # Activate synthetic test parameters once, before frozen-state assertions.
        torch.nn.init.normal_(self.m.motion[-1].weight,std=.003)
        self.m.requires_grad_(False)
        self.p=DataProcessor('001-rope');self.raw=synthetic_batch(points=24,horizon=15,prediction_interval=1)
        self.b=prepare_batch(self.raw,self.p,self.m);self.x=self.b['inputs']

    def test_interventions_identical_to_previous_audit(self):
        old=subprocess.check_output(['git','show','0934d8c:scripts/experiments/v4_attention_gradient/core.py'],text=True,encoding='utf8')
        funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
        for name in ('observe','altered_inputs','fixed_regions','rms_mm'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(iv,name))).body[0],include_attributes=False),funcs[name])

    def test_h15_normal_hook_identity_parameters_and_attributes(self):
        before=digest(self.m.state_dict());xhash=digest(self.x)
        with torch.inference_mode():
            plain=self.m(self.x)
            with iv.observe(self.m) as cap:out,t=self.m(self.x,return_trace=True)
        self.assertEqual(out['xyz'].shape,(2,15,24,3));self.assertEqual(len(cap['motion_input']),15)
        for k in out:torch.testing.assert_close(out[k],plain[k],rtol=0,atol=0)
        for k,a,b in [('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)]:
            expected=torch.where(self.x['gs_mask'][...,None],self.x['gs_attr'][...,a:b],0)[:,None].expand_as(out[k])
            torch.testing.assert_close(out[k],expected,rtol=0,atol=0)
        self.assertEqual(before,digest(self.m.state_dict()));self.assertEqual(xhash,digest(self.x))
        self.assertEqual(len(self.m.motion._forward_pre_hooks),0)

    def test_freeze_exact_mask_and_tactile_semantics(self):
        x=copy.deepcopy(self.x);x['controller_mask'][:,0,1,0]=False;x['action_valid'][:,0,0]=False
        before=digest(x);f=iv.altered_inputs(x,'freeze_action')
        for key in x:
            if key not in ('action_state','controller_points','action_valid','controller_mask'):
                if isinstance(x[key],torch.Tensor):torch.testing.assert_close(x[key],f[key],rtol=0,atol=0,equal_nan=True)
        for key in ('action_state','controller_points'):torch.testing.assert_close(f[key],x[key][:,:1].expand_as(x[key]),rtol=0,atol=0)
        for key in ('action_valid','controller_mask'):self.assertTrue(torch.equal(f[key],x[key]&x[key][:,:1]))
        self.assertEqual(before,digest(x))

    def test_zero_only_twelve_fields_at_each_invocation(self):
        seen=[]
        handle=self.m.motion.register_forward_pre_hook(lambda m,args:seen.append(args[0].clone()))
        with torch.inference_mode():
            with iv.observe(self.m,zero_tool=True) as cap:pred=self.m(self.x)
        handle.remove()
        for original,modified in zip(seen,cap['motion_input']):
            torch.testing.assert_close(original[...,:128],modified[...,:128],rtol=0,atol=0)
            torch.testing.assert_close(original[...,140:],modified[...,140:],rtol=0,atol=0)
            self.assertFalse(modified[...,128:140].any())
        self.assertTrue(torch.isfinite(pred['xyz']).all());self.assertFalse(pred['xyz'].requires_grad)

    def test_hook_cleanup_on_error(self):
        with self.assertRaisesRegex(RuntimeError,'expected'):
            with iv.observe(self.m,zero_tool=True):raise RuntimeError('expected')
        self.assertFalse(self.m.motion._forward_pre_hooks);self.assertFalse(self.m.cross[-1]._forward_hooks)

    def test_causal_h15_and_disabled_hand(self):
        with torch.inference_mode():
            ref=self.m(self.x)['xyz'];x=copy.deepcopy(self.x)
            x['controller_points'][:,10:]+=10;x['action_state'][:,10:]+=2;x['tactile'][:,10:]+=5
            got=self.m(x)['xyz'];torch.testing.assert_close(got[:,:10],ref[:,:10],rtol=0,atol=0)
            x=copy.deepcopy(self.x);x['controller_points'][0,:,0]=float('nan');x['action_state'][0,:,0]=float('nan')
            torch.testing.assert_close(self.m(x)['xyz'][0],ref[0],rtol=0,atol=0)

    def test_fixed_regions_padding_and_gt_order_independence(self):
        # Metrics use one sample and arbitrary independent target cardinality/order.
        x=dict(gs_xyz=torch.tensor([[[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0],[99.,0,0]]]),
            gs_mask=torch.tensor([[True,True,True,True,False]]),gs_attr=torch.tensor([[[0.,0,0,1,0,0,0,0,0,0,0]]*5]),
            controller_points=torch.zeros(1,1,2,30,3),controller_mask=torch.ones(1,1,2,30,dtype=torch.bool),operation_hands=torch.tensor([[False,True]]))
        f=fixed_field(x['gs_xyz'],x['gs_mask']);regions,_=iv.fixed_regions({'inputs':x},f)
        self.assertEqual(regions['near'].tolist(),[0]);self.assertEqual(regions['far'].tolist(),[3]);self.assertNotIn(4,regions['all'])
        pred={'xyz':x['gs_xyz'][:,None], 'log_scales':x['gs_attr'][:,None,:,:3], 'quaternions':x['gs_attr'][:,None,:,3:7]}
        gt=torch.tensor([[[0.,1,0],[1.,1,0],[2.,1,0],[3.,1,0],[999.,0,0]]]);mask=torch.tensor([[True,True,True,True,False]])
        b={'inputs':x,'targets':{'target_xyz':[gt],'target_mask':[mask]}}
        a=geometric_metrics(pred,b,pred,regions,.001)[0]
        self.assertAlmostEqual(a['cd_mm'],1.);self.assertAlmostEqual(a['near_pred_to_gt_mm'],1.);self.assertEqual(a['all_change_from_normal_mm'],0.)
        b['targets']={'target_xyz':[gt[:,:4].flip(1)],'target_mask':[mask[:,:4]]}
        self.assertEqual(a,geometric_metrics(pred,b,pred,regions,.001)[0])

    def test_no_optimizer_or_backward_in_diagnostic_sources(self):
        root=Path(__file__).parents[2]/'experiments/v4_h15_condition_ablation'
        for path in root.glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('backward','step','Adam','AdamW','SGD'))

if __name__=='__main__':unittest.main()

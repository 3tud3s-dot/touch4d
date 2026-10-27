import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch
from scripts.training.model import PositionPredictor
from scripts.training.gs_decoder import build_initial_anchor_field,interpolate_anchor_displacements
from scripts.experiments.tool_anchor_b.model import ARCH,AnchorBPredictor,ToolAnchorAttention
from scripts.experiments.tool_anchor_b.conditions import ToolPredictor,hand_inputs,sanitize
from scripts.experiments.tool_anchor_b.run import fresh,payload,restore

def fixture():
    b=DataProcessor('001-rope')(make_batch(horizon=15))
    b['inputs']=sanitize(hand_inputs(b['inputs'],torch.tensor([2,5]),['001-rope','001-rope']))
    return b

def small(mode='candidate'):
    return fresh(mode,config=dict(ARCH,width=16,anchor_count=4,anchor_pool_k=2,gs_local_k=2))

def active(m):
    torch.nn.init.normal_(m.position_head[-1].weight,std=.02)
    torch.nn.init.normal_(m.gs_structure.output.weight,std=.01)
    torch.nn.init.normal_(m.fusion[-1].weight,std=.01)

class ToolAnchorBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_base_forward_and_common_initialization(self):
        a,b=small('base'),small();self.assertIs(ToolPredictor.forward,PositionPredictor.forward)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        x=fixture()['inputs'];pa,pb=a(x),b(x)
        for key in pa:self.assertTrue(torch.equal(pa[key],pb[key]))

    def test_h15_shapes_padding_finite_and_inherited_attributes(self):
        x=fixture()['inputs'];m=small();active(m);p=m(x)
        self.assertEqual(p['xyz'].shape,(2,15,x['gs_xyz'].shape[1],3))
        for key,v in p.items():
            self.assertTrue(torch.isfinite(v).all(),key)
            self.assertEqual(float(v.masked_select((~x['gs_mask'])[:,None,:,None]).abs().sum()),0.)
        torch.testing.assert_close(p['log_scales'],x['gs_attr'][:,None,:,:3].expand_as(p['log_scales']),atol=0,rtol=0)

    def test_single_hand_and_fallback_masks(self):
        x=fixture()['inputs'];m=small();z,k=m.encode_conditions(x)
        self.assertFalse(k[0,:,0].any());self.assertFalse(k[0,:,2:32].any());self.assertFalse(k[0,:,62:64].any())
        x['controller_mask'][:,:,1]=False;x['valid_mask'][:,:,2:]=True;x['tactile_spatial_mask'][:,:,2:]=True
        x['tactile_age_us'][:]=0;x['tactile'][:]=0
        z,k=m.encode_conditions(x);self.assertTrue(k[:,:,64:].all());self.assertFalse(k[0,:,62:64].any())

    def test_missing_gs_sample_does_not_contaminate_other_sample(self):
        x=fixture()['inputs'];m=small();active(m);before=m(x)
        x['gs_mask'][0]=False;x['gs_xyz'][0]=float('nan');x['gs_attr'][0]=float('nan')
        after=m(x)
        for k,v in after.items():
            self.assertTrue(torch.isfinite(v).all());self.assertEqual(float(v[0].detach().abs().sum()),0.)
            torch.testing.assert_close(v[1],before[k][1],atol=0,rtol=0)

    def test_tactile_zero_observation_and_invalid_nan(self):
        x=fixture()['inputs'];x['valid_mask'][:]=True;x['tactile_spatial_mask'][:]=True;x['tactile_age_us'][:]=0;x['tactile'][:]=0
        x=sanitize(x);values=small().aggregate(x)
        self.assertTrue(values[4][:,:,2:].all());self.assertEqual(float(values[0].abs().max()),0.)
        x['tactile'][:,:,2]=float('nan');x=sanitize(x);values=small().aggregate(x)
        self.assertFalse(values[4][:,:,2].any());self.assertTrue(torch.isfinite(values[0]).all())

    def test_a0_endpoints_and_first_step(self):
        x=fixture()['inputs'];x['controller_points'][:,:,1]+=torch.arange(15)[None,:,None,None]*.001
        x['controller_mask'][:,:,1]=True;x['controller_mask'][0,0,1,0]=False;x['controller_points'][0,0,1,0]=float('nan')
        x=sanitize(x);v=small().controller_inputs(x)
        self.assertEqual(float(v[:,0,...,3:6].abs().sum()),0.)
        self.assertEqual(float(v[0,:,1,0,3:].abs().sum()),0.)
        torch.testing.assert_close(v[1,:,1,:,3:6],x['controller_points'][1,:,1]-x['controller_points'][1,:1,1])

    def local_inputs(self):
        m=ToolAnchorAttention(16,4)
        for layer in (m.query,m.key):torch.nn.init.zeros_(layer.weight)
        for p in m.bias.parameters():torch.nn.init.zeros_(p)
        q=torch.zeros(1,2,16);centers=torch.zeros(1,2,3);ak=torch.tensor([[True,False]])
        z=torch.randn(1,3,16);pos=torch.tensor([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]]);tk=torch.tensor([[True,True,False]])
        d=torch.tensor([[[.1,0.,0.],[0.,.2,0.],[9.,9.,9.]]]);sigma=torch.ones(1)
        return m,[q,centers,ak,z,pos,tk,d,sigma]

    def test_exact_log_prior_normalization_and_motion(self):
        m,args=self.local_inputs();out,e=m(*args,True)
        expected=torch.softmax(torch.tensor([0.,0.,-.5,float('-inf')]),0)
        torch.testing.assert_close(e['alpha'][0,0],expected.expand(4,4))
        torch.testing.assert_close(e['alpha'].sum(-1),torch.ones(1,2,4))
        torch.testing.assert_close(e['motion'][0,0],expected[1]*args[6][0,0]+expected[2]*args[6][0,1])
        self.assertEqual(float(out[0,1].abs().sum()),0.)
        self.assertTrue((e['alpha'][0,1,:,0]==1).all())

    def test_all_invalid_and_distant_null_retreat(self):
        m,args=self.local_inputs();args[5][:]=False;args[3][:]=float('nan');args[4][:]=float('nan');args[6][:]=float('nan')
        out,e=m(*args,True);self.assertTrue(torch.isfinite(out).all());self.assertEqual(float(out.abs().sum()),0.)
        self.assertTrue((e['alpha'][...,0]==1).all())
        m,args=self.local_inputs();args[4][:]=10000
        out,e=m(*args,True);self.assertTrue((e['alpha'][...,0]==1).all());self.assertEqual(float(out.abs().sum()),0.)

    def test_spatial_units_and_motion_scaling(self):
        m,args=self.local_inputs();_,e=m(*args,True)
        scaled=[v.clone() for v in args]
        for i in (1,4,6,7):scaled[i]*=3
        _,s=m(*scaled,True)
        torch.testing.assert_close(e['alpha'],s['alpha']);torch.testing.assert_close(s['motion'],e['motion']*3)

    def test_fixed_field_and_live_anchor_coordinates(self):
        x=fixture()['inputs'];m=small();active(m)
        with patch('scripts.experiments.tool_anchor_b.model.build_initial_anchor_field',wraps=build_initial_anchor_field) as build:
            p,trace,field=m(x,True)
        self.assertEqual(build.call_count,1);self.assertFalse(field.weights.requires_grad)
        torch.testing.assert_close(field.weights.sum(-1)[x['gs_mask']],torch.ones_like(field.weights.sum(-1)[x['gs_mask']]))
        batch=torch.arange(2)[:,None]
        for h,row in enumerate(trace):
            previous=x['gs_xyz'] if h==0 else p['xyz'][:,h-1]
            expected=previous[batch,field.anchor_indices]*field.anchor_mask[...,None]
            torch.testing.assert_close(row['centers'],expected,atol=0,rtol=0)

    def test_cumulative_d0_not_incremental_and_no_motion_bypass(self):
        x=fixture()['inputs'];m=small();m.position_head[-1].bias.data.fill_(.01)
        p=m(x);expected=(x['gs_xyz']+.01)*x['gs_mask'][...,None]
        for h in range(15):torch.testing.assert_close(p['xyz'][:,h],expected)
        m.position_head[-1].bias.data.zero_();x['controller_points']+=100
        p=m(x);torch.testing.assert_close(p['xyz'],x['gs_xyz'][:,None].expand_as(p['xyz']))

    def test_future_conditions_gt_and_earlier_prefix(self):
        b=fixture();m=small();active(m);x=b['inputs'];p=m(x);changed=copy.deepcopy(x)
        for name in ('action_state','controller_points','tactile'):changed[name][:,8:]+=2
        q=m(changed)
        for key in p:torch.testing.assert_close(p[key][:,:8],q[key][:,:8],atol=0,rtol=0)
        b['targets']['target_xyz'][0]+=100
        q=m(x)
        for key in p:torch.testing.assert_close(p[key],q[key],atol=0,rtol=0)

    def test_all_steps_and_spatial_modules_receive_gradient(self):
        x=fixture()['inputs'];m=small();active(m)
        # Near-but-not-coincident tools avoid testing only a numerically null neighborhood.
        x['controller_mask'][:,:,1]=True
        x['controller_points'][:,:,1]=x['gs_xyz'][:,:1,None]+torch.arange(15)[None,:,None,None]*.001
        p,trace,field=m(x,True)
        for row in trace:row['anchor_delta'].retain_grad()
        p['xyz'][:,-1].square().sum().backward()
        self.assertTrue(all(row['anchor_delta'].grad is not None and row['anchor_delta'].grad.abs().sum()>0 for row in trace))
        for prefix in ('tool_attention.bias','tool_attention.value','tool_attention.motion_fusion','anchor_to_gs','gs_encoder','position_head'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None),0.,prefix)

    def test_history_remains_visible_to_global_attention(self):
        x=fixture()['inputs'];m=small();seen=[]
        hook=m.blocks[0].register_forward_pre_hook(lambda mod,args:seen.append(args[1].shape[1]))
        m(x);hook.remove();self.assertEqual(seen,[1+66*(k+1) for k in range(15)])

    def test_checkpoint_roundtrip_and_wrong_mode_rejection(self):
        for mode in ('base','candidate'):
            m=small(mode);active(m);opt=torch.optim.AdamW(m.parameters());x=fixture()['inputs'];p=m(x)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'test.pt';torch.save(payload(m,opt,mode,4,'test'),path)
                n=small(mode);no=torch.optim.AdamW(n.parameters());restore(path,n,no,mode);q=n(x)
                for key in p:torch.testing.assert_close(p[key],q[key],atol=0,rtol=0)
                with self.assertRaises(ValueError):restore(path,n,no,'candidate' if mode=='base' else 'base')

if __name__=='__main__':unittest.main()

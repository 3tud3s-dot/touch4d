import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_action_identifiability.data import make_pair, normalize, action_variant, load_initial
from scripts.experiments.v4_action_identifiability.run import update, evaluate
from scripts.training.v4_runtime import build_model, save_checkpoint, load_checkpoint
from scripts.training.v4_model import fixed_field
from scripts.tests.dataloader.test_dataloader import MemoryReconstruction


def fixture():
    old=synthetic_batch(80,1)
    initial={k:v[0].clone() for k,v in old['inputs'].items() if isinstance(v,torch.Tensor) and k not in ('gs_mask','gs_lengths','operation_hands')}
    return make_pair(initial,list(range(1700000000000000,1700000000000000+11*50000,50000)),{'kind':'CPU fixture'})


class PairedActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_only_action_differs_rigid_translation_zero_pressure_and_causality(self):
        raw,w,region,m=fixture(); x=raw['inputs']
        self.assertEqual(set(m['differing_input_fields']),{'action_state','controller_points'})
        self.assertTrue(torch.equal(x['action_state'][:,0],x['action_state'][0:1,0].expand(2,-1,-1)))
        cp=x['controller_points'][:,:,1]-x['controller_points'][:,0:1,1]
        hand=x['action_state'][:,:,1,:3]-x['action_state'][:,0:1,1,:3]
        torch.testing.assert_close(cp,hand[:,:,None].expand_as(cp),atol=4e-8,rtol=0)
        self.assertTrue(torch.equal(raw['targets']['target_xyz'][0][0],x['gs_xyz'][0]))
        for h in range(10):
            dt=raw['targets']['target_xyz'][h]-x['gs_xyz']
            torch.testing.assert_close(dt[0],-dt[1],atol=6e-8,rtol=0)
            # Compare the frozen construction exactly. Subtracting two ~0.5m
            # float32 positions first introduces an extra ULP of cancellation.
            expected=x['gs_xyz'][0].clone()
            expected[:,2]+=torch.tensor(m['action_z_m'][h])*w
            torch.testing.assert_close(raw['targets']['target_xyz'][h][0],expected,atol=0,rtol=0)
            expected_hand=x['action_state'][0,0,1,2]+torch.tensor(m['action_z_m'][h])
            torch.testing.assert_close(x['action_state'][0,h,1,2],expected_hand,atol=0,rtol=0)
        model=build_model('v4_stage1');batch,p=normalize(raw,model)
        _,_,f,mask,_=model.conditions.features(batch['inputs'],1)
        self.assertFalse(mask[:,:30].any());self.assertTrue(mask[:,30:].all())
        self.assertTrue((f[:,30:,7:9]==0).all());self.assertTrue((f[:,30:,11:13]==1).all())

    def test_equal_anchors_no_branch_tag_and_nonzero_head_first_step_equal(self):
        raw,*_=fixture();model=build_model('v4_stage1');batch,p=normalize(raw,model)
        field=fixed_field(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
        self.assertTrue(torch.equal(field.anchor_indices[0],field.anchor_indices[1]))
        self.assertTrue(torch.equal(field.weights[0],field.weights[1]))
        torch.nn.init.normal_(model.motion[-1].weight,std=.002)
        xyz=model(batch['inputs'])['xyz']
        torch.testing.assert_close(xyz[0,0],xyz[1,0],rtol=0,atol=0)
        changed=copy.deepcopy(batch['inputs']);changed['action_state'][:,6:,:,2]+=.2;changed['controller_points'][:,6:,...,2]+=.2
        torch.testing.assert_close(model(changed)['xyz'][:,:6],xyz[:,:6],rtol=0,atol=0)

    def test_static_and_swap_consistency(self):
        raw,_,a,_=fixture();model=build_model('v4_stage1');batch,p=normalize(raw,model)
        torch.nn.init.normal_(model.motion[-1].weight,std=.002)
        real=model(batch['inputs'])['xyz'];swapped=model(action_variant(batch,'swap')['inputs'])['xyz']
        torch.testing.assert_close(swapped,real.flip(0),rtol=0,atol=0)
        static=model(action_variant(batch,'static')['inputs'])['xyz']
        torch.testing.assert_close(static[0],static[1],rtol=0,atol=0)
        half=action_variant(batch,'half')['inputs']
        torch.testing.assert_close(half['action_state'][:,0],batch['inputs']['action_state'][:,0])

    def test_loss_gradient_update_checkpoint_and_metrics(self):
        raw,w,a,_=fixture();torch.manual_seed(42);model=build_model('v4_stage1');batch,p=normalize(raw,model)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        records=[update(model,opt,batch) for _ in range(3)]
        for group in ('controller','motion','attention_score','gs_encoder'):
            self.assertTrue(any(r['raw_gradients'][group]['gradient_norm']>0 for r in records[1:]),group)
        metrics,pred=evaluate(model,batch,p,a)
        self.assertEqual(len(metrics['cd_mm']),2);self.assertEqual(len(metrics['cd_mm'][0]),10)
        self.assertEqual(metrics['target_separation_mm'][0],0.)
        self.assertGreater(metrics['target_affected_signed_separation_z_mm'][-1],18.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'check.pt';save_checkpoint(path,model,opt,3,source_sha='test',processor_state=p.state_dict(),run_state={'diagnostic_loss':'CD only'})
            clone=build_model('v4_stage1');co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
            state=load_checkpoint(path,clone,co,processor_state=p.state_dict());self.assertEqual(state['successful_steps'],3)
            torch.testing.assert_close(model(batch['inputs'])['xyz'],clone(batch['inputs'])['xyz'],rtol=0,atol=0)
            update(model,opt,batch);update(clone,co,batch)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)

    def test_reader_never_opens_future_gs(self):
        source=MemoryReconstruction(frames=90)
        from scripts.dataloader.dataset import read_gs as original
        frames=[]
        def guarded(info,frame):
            self.assertEqual(frame,73);frames.append(frame);return original(info,frame)
        with source.mounted(),patch('scripts.experiments.v4_action_identifiability.data.read_gs',side_effect=guarded):
            initial,meta=load_initial(source.root)
        self.assertEqual(frames,[73]);self.assertFalse(meta['future_real_gs_read'])

    def test_geometry_plot_and_report_contract(self):
        # The synthetic template is allowed to use point order in visualization only.
        from scripts.experiments.v4_action_identifiability.report import scatter,curve
        raw,*_=fixture();x=raw['inputs']['gs_xyz'][0].numpy()
        self.assertIn('<svg',scatter(x,x,x,'test',(0,2)))
        self.assertIn('polyline',curve([('test',[(0,0.),(400,1.)],'blue')],'test'))

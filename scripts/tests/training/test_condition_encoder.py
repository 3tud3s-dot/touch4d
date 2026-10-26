"""CPU-only contract/gradient tests for the opt-in condition experiment."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch
from scripts.training.model import PositionPredictor
from scripts.experiments.condition_encoder.model import (ConditionPredictor, MODES, hand_inputs, sanitize)
from scripts.experiments.condition_encoder.screen import (paired_model, select, save_checkpoint, restore, perturb)


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        self.processor=DataProcessor('001-rope')
        self.batch=self.processor(make_batch())
        self.x=hand_inputs(self.batch['inputs'],torch.tensor([1,5]),['001-rope']*2)
        self.arch=dict(width=16,heads=4,layers=2,controller_features=7,horizon=5,
            gs_local_attention='knn',gs_local_k=3,gs_decoder='anchors',anchor_count=4,
            anchor_pool_k=3,gs_structure='anchor_attention',output_mode='gs_geometry')

    def model(self,mode): return paired_model(mode,**self.arch)[0]

    def all_valid(self):
        x=copy.deepcopy(self.x)
        x['action_valid'].fill_(True);x['controller_mask'].fill_(True)
        x['valid_mask'].fill_(True);x['valid_mask'][...,-1]=False
        x['tactile_spatial_mask']=x['valid_mask'].clone()
        x['tactile_age_us'].fill_(20000)
        x['operation_hands'].fill_(True)
        return x

    def test_paired_shared_initialization(self):
        for mode in MODES: self.model(mode)
        a,b=self.model('T2'),self.model('T3')
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k in a.state_dict(): self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))

    def test_b0_exact_old_model_with_gated_inputs(self):
        new=self.model('B0');old=PositionPredictor(**self.arch);old.load_state_dict(new.state_dict())
        torch.nn.init.normal_(new.position_head[-1].weight,std=.01);old.load_state_dict(new.state_dict())
        x=sanitize(self.x)
        for k,v in new(self.x).items():self.assertTrue(torch.equal(v,old(x)[k]))

    def test_adjacent_and_initial_validity(self):
        x=self.all_valid();x['controller_points'][:,0]=0
        for j in range(1,5):x['controller_points'][:,j]=j*j
        a=self.model('A1').controller_inputs(x);b=self.model('B0').controller_inputs(x)
        self.assertTrue((a[:,0,...,3:]==0).all())
        self.assertTrue((a[:,3,...,3:6]==5).all());self.assertTrue((b[:,3,...,3:6]==9).all())
        x['controller_mask'][:,1]=False
        a=self.model('A1').controller_inputs(x)
        self.assertTrue((a[:,2,...,3:]==0).all())
        self.assertTrue((a[:,2,...,:3]==4).all())

    def test_hand_masks_all_modes_after_embeddings(self):
        x=self.all_valid();x['operation_hands'][0,0]=False
        for mode in MODES:
            tokens,keep=self.model(mode).encode_conditions(x)
            ids=[0,*range(2,32),62,63]
            self.assertFalse(keep[0,:,ids].any());self.assertTrue((tokens[0,:,ids]==0).all())
            self.assertTrue(keep[1,:,0].all()) # stationary dual hand never removed

    def test_invalid_nan_and_age(self):
        x=self.all_valid();x['action_state'][0,0,0]=float('nan')
        x['controller_points'][0,0,0,0]=float('nan');x['tactile'][0,0,0,0,0]=float('nan')
        x['tactile_age_us'][0,1,0]=75001
        for mode in MODES:
            tokens,keep=self.model(mode).encode_conditions(x)
            self.assertTrue(torch.isfinite(tokens).all());self.assertFalse(keep[0,0,0])
        self.assertFalse(sanitize(x)['valid_mask'][0,1,0].any())

    def test_valid_zero_constant_local_max_and_coverage(self):
        x=sanitize(self.all_valid());x['tactile'].zero_()
        for mode in ('T2','T3'):
            model=self.model(mode);mean,maximum,cov,age,obs,maxobs=model.aggregate(x)
            self.assertTrue(obs.all());self.assertTrue(maxobs.all());self.assertTrue((mean==0).all())
            self.assertTrue((cov==1).all());self.assertTrue(torch.allclose(age,torch.full_like(age,20000/75000)))
            x['tactile'].fill_(3)
            mean,maximum,*_=model.aggregate(x)
            self.assertTrue(torch.allclose(mean,torch.full_like(mean,3)));self.assertTrue((maximum==3).all())
            x['tactile'].zero_()
        x['tactile'][0,0,0,2,8]=9
        mean,maximum,*_=self.model('T3').aggregate(x)
        self.assertEqual(int((maximum[0,0,0]>0).sum()),1)
        self.assertTrue((maximum[0,0,1:]==0).all())
        self.assertTrue((self.model('T2').aggregate(x)[1][0,0,0]==9).all())

    def test_local_empty_max_and_fallback(self):
        x=self.all_valid();model=self.model('T3')
        x['valid_mask'].zero_();x['valid_mask'][...,0,0]=True
        x=sanitize(x)
        mean,maximum,cov,age,obs,maxobs=model.aggregate(x)
        self.assertTrue(obs.all());self.assertTrue((maxobs.sum(-1)==1).all())
        x=self.all_valid();x['controller_mask'].zero_()
        for mode in ('T2','T3'):
            x['operation_hands'].fill_(True)
            model=self.model(mode);_,keep=model.encode_conditions(x)
            self.assertTrue(keep[...,62:].all());self.assertFalse(keep[...,2:62].any())
            x['operation_hands'][0,0]=False
            _,keep=model.encode_conditions(x);self.assertFalse(keep[0,:,62:64].any())

    def test_spatial_missing_fallback_and_zero_residual(self):
        x=self.all_valid()
        a,b=self.model('T2'),self.model('T3')
        self.assertTrue(torch.equal(a.encode_conditions(x)[0],b.encode_conditions(x)[0]))
        x['tactile_spatial_mask'].zero_()
        for model in (a,b):
            _,keep=model.encode_conditions(x);self.assertTrue(keep[...,62:].all())

    def test_causality_nonzero_heads_and_gradients(self):
        x=self.all_valid()
        for mode in MODES:
            model=self.model(mode);torch.nn.init.normal_(model.position_head[-1].weight,std=.03)
            if mode in ('T2','T3'):torch.nn.init.normal_(model.fusion[-1].weight,std=.03)
            source=copy.deepcopy(x);source['tactile'].requires_grad_();source['controller_points'].requires_grad_()
            out=model(source)['xyz'];changed=copy.deepcopy(x)
            changed['controller_points'][:,3:]+=50;changed['tactile'][:,3:]+=30
            self.assertTrue(torch.equal(out[:,:3],model(changed)['xyz'][:,:3]))
            changed=dict(x,target_xyz=torch.randn(2,5,30,3))
            self.assertTrue(torch.equal(model(x)['xyz'],model(changed)['xyz']))
            out[:,-1].square().mean().backward()
            self.assertGreater(source['controller_points'].grad.abs().sum().item(),0)
            self.assertGreater(source['tactile'].grad.abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_null_only_and_no_input_mutation(self):
        x=self.all_valid();x['operation_hands'].zero_();original=copy.deepcopy(x)
        for mode in MODES:
            model=self.model(mode);tokens,keep=model.encode_conditions(x)
            self.assertFalse(keep.any());self.assertTrue((tokens==0).all())
            self.assertTrue(torch.isfinite(model(x)['xyz']).all())
        for k in x:self.assertTrue(torch.equal(x[k],original[k]))

    def test_checkpoint_reject_mode_horizon_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            model=self.model('T2');opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            path=Path(folder)/'last.pt'
            save_checkpoint(path,model,opt,self.processor,'abc',10,[])
            self.assertEqual(restore(path,model,opt,self.processor,'abc'),10)
            for altered in (self.model('T3'),ConditionPredictor(condition_mode='T2',**dict(self.arch,horizon=10))):
                with self.assertRaises(ValueError):restore(path,altered,opt,self.processor,'abc')
            with self.assertRaises(ValueError):restore(path,model,opt,self.processor,'bad')

    def test_windows_and_perturbation_keep_zero_validity(self):
        windows=[(e,s) for e in [1,2,4,5,7,8] for s in range(101)]
        selected=select(windows);self.assertEqual(len(selected),24);self.assertEqual(selected,select(windows))
        x=self.all_valid();p=perturb(x,'zero_tactile')
        self.assertTrue(torch.equal(p['valid_mask'],x['valid_mask']));self.assertTrue((p['tactile']==0).all())


if __name__=='__main__':unittest.main()

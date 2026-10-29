import unittest
import torch
from scripts.experiments.v4_g5_signed_local.propagation import (
    build_anchor_graph, SignedAnchorPropagation)


class AnchorGraphTests(unittest.TestCase):
    def test_initial_self_and_eight_others_deterministic_ties(self):
        xyz = torch.zeros(1, 12, 3)
        xyz[0, :, 0] = torch.arange(12.)
        keep = torch.ones(1, 12, dtype=torch.bool)
        indices, mask = build_anchor_graph(xyz, keep)
        self.assertEqual(tuple(indices.shape), (1, 12, 9))
        self.assertTrue(mask.all())
        for a in range(12):
            expected = [a] + sorted([b for b in range(12) if b != a],
                                     key=lambda b: (abs(a-b), b))[:8]
            self.assertEqual(indices[0, a].tolist(), expected)
        again, _ = build_anchor_graph(xyz, keep)
        self.assertTrue(torch.equal(indices, again))

    def test_padding_nan_small_and_single_anchor(self):
        xyz = torch.tensor([[[0.,0,0],[1.,0,0],[float('nan')]*3],
                            [[3.,0,0],[float('nan')]*3,[float('nan')]*3]])
        keep = torch.tensor([[True,True,False],[True,False,False]])
        indices, mask = build_anchor_graph(xyz, keep)
        self.assertEqual(indices[0, 0, :2].tolist(), [0,1])
        self.assertEqual(indices[0, 1, :2].tolist(), [1,0])
        self.assertEqual(mask.sum(-1).tolist(), [[2,2,0],[1,0,0]])
        for b in range(2):
            self.assertTrue(keep[b, indices[b][mask[b]]].all())
        one, valid = build_anchor_graph(torch.zeros(1,1,3),torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(one.tolist(), [[[0]]]);self.assertTrue(valid.all())


class SignedPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        torch.manual_seed(245)
        self.model = SignedAnchorPropagation(128)
        self.keep = torch.tensor([[True,True,True,False]])
        self.hands = torch.tensor([[True,True]])
        self.xyz = torch.randn(1,4,3)
        self.graph,self.mask = build_anchor_graph(self.xyz,self.keep)
        self.q = torch.randn(1,4,128)
        self.token = torch.randn(1,2,128)
        self.basis = torch.randn(1,4,2,18)

    def call(self,seed,**changes):
        args = dict(query=self.q,hand_tokens=self.token,basis18=self.basis,
                    direct_seed_per_hand=seed,initial_xyz=self.xyz,current_xyz=self.xyz+.01,
                    anchor_valid=self.keep,hand_valid=self.hands,
                    indices=self.graph,neighbor_mask=self.mask)
        args.update(changes)
        return self.model(**args)

    def test_weights_only_valid_neighbors_and_constant_signed_seeds_unchanged(self):
        seed = torch.tensor([[[[1.,-2.,3.],[-1.,2.,-3.]]]]).expand(1,4,2,3).clone()
        output,weight = self.call(seed)
        torch.testing.assert_close(output[:,:3],seed[:,:3],atol=0,rtol=0)
        self.assertTrue(torch.all(output[:,3] == 0))
        torch.testing.assert_close(weight[:,:3].sum(-1),torch.ones(1,3,2),atol=1e-6,rtol=0)
        self.assertTrue(torch.all(weight[:,3] == 0))
        self.assertTrue(torch.all(weight.masked_select(~self.mask[:,:,None].expand_as(weight)) == 0))

    def test_heterogeneous_signed_seeds_really_mix(self):
        seed = torch.zeros(1,4,2,3);seed[0,0,0,0]=1;seed[0,1,1,1]=-2
        output,weight = self.call(seed)
        self.assertGreater(float(output[0,1,0,0].detach()),0.)
        self.assertLess(float(output[0,0,1,1].detach()),0.)
        # The graph's rows are permuted self-first; use actual indexed seeds.
        neighbor=seed[torch.arange(1)[:,None,None],self.graph].permute(0,1,3,2,4)
        expected=seed+.5*(weight[...,None]*(neighbor-seed[:,:,:,None])).sum(-2)
        expected[:,3]=0
        torch.testing.assert_close(output,expected,rtol=0,atol=0)

    def test_hands_independent_and_missing_hand_no_nan(self):
        seed=torch.randn(1,4,2,3);before,weight=self.call(seed)
        changed=seed.clone();changed[:,:,0]*=-8
        after,_=self.call(changed)
        torch.testing.assert_close(before[:,:,1],after[:,:,1],rtol=0,atol=0)
        changed_token=self.token.clone();changed_token[:,0]*=-3
        changed_basis=self.basis.clone();changed_basis[:,:,0]*=7
        after,changed_weights=self.call(seed,hand_tokens=changed_token,basis18=changed_basis)
        torch.testing.assert_close(before[:,:,1],after[:,:,1],rtol=0,atol=0)
        torch.testing.assert_close(weight[:,:,1],changed_weights[:,:,1],rtol=0,atol=0)
        token=self.token.clone();token[:,0]=float('nan')
        basis=self.basis.clone();basis[:,:,0]=float('nan')
        seed[:,:,0]=float('nan')
        out,w=self.call(seed,hand_tokens=token,basis18=basis,hand_valid=torch.tensor([[False,True]]))
        self.assertTrue(torch.isfinite(out).all());self.assertTrue(torch.all(out[:,:,0] == 0))
        self.assertTrue(torch.all(w[:,:,0] == 0))

    def test_padding_nan_and_gradient_isolation_score_and_live_geometry(self):
        q=self.q.clone();q[:,3]=float('nan');q.requires_grad_()
        xyz=self.xyz.clone();xyz[:,3]=float('nan');xyz.requires_grad_()
        basis=self.basis.clone();basis[:,3]=float('nan');basis.requires_grad_()
        seed=torch.randn(1,4,2,3);seed[:,3]=float('nan');seed.requires_grad_()
        out,weight=self.call(seed,query=q,basis18=basis,current_xyz=xyz)
        (out.square()*torch.tensor([1.,2.,3.,0.])[None,:,None,None]).sum().backward()
        for value in (q,xyz,basis,seed):
            self.assertIsNotNone(value.grad);self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue(torch.all(value.grad[:,3] == 0))
        grads=[p.grad for p in self.model.score.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        self.assertGreater(float(xyz.grad[:,:3].abs().sum()),0.)

    def test_one_anchor_and_all_invalid_graph_rows(self):
        keep=torch.tensor([[True,False,False,False]])
        graph,mask=build_anchor_graph(self.xyz,keep)
        seed=torch.randn(1,4,2,3)
        out,_=self.call(seed,anchor_valid=keep,indices=graph,neighbor_mask=mask)
        torch.testing.assert_close(out[:,0],seed[:,0],rtol=0,atol=0)
        self.assertTrue(torch.all(out[:,1:] == 0))
        empty=torch.zeros_like(keep);graph,mask=build_anchor_graph(self.xyz,empty)
        out,w=self.call(seed,anchor_valid=empty,indices=graph,neighbor_mask=mask)
        self.assertTrue(torch.all(out == 0));self.assertTrue(torch.all(w == 0))


if __name__=='__main__':unittest.main()

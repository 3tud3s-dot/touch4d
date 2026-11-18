import unittest
import torch
from scripts.experiments.local_response.head import EdgeResponseHead
from scripts.experiments.local_response.correction import preserve_transport


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]],dtype=torch.float64)
        self.edges=torch.tensor([[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]])
        self.mask=torch.ones(1,6,dtype=torch.bool)
        self.valid=torch.ones(1,4,dtype=torch.bool)
        self.head=EdgeResponseHead(hidden_dim=8,min_length=1e-8).double()

    def call(self,y,x=None,edges=None,mask=None):
        return self.head(self.x if x is None else x,y,self.edges if edges is None else edges,self.mask if mask is None else mask,self.valid)

    def activate(self):
        with torch.no_grad():self.head.net[-1].bias.fill_(.4)

    def test_zero_init_identity_and_head_gradient(self):
        y=self.x*1.2
        q=self.call(y);self.assertEqual(q.abs().max(),0)
        (q*(y-self.x)).sum().backward()
        self.assertGreater(self.head.net[-1].weight.grad.abs().sum(),0)
        self.assertTrue(torch.isfinite(self.head.net[-1].weight.grad).all())

    def test_rigid_transforms_do_not_trigger(self):
        self.activate()
        r=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=self.x.dtype)
        q=self.call(self.x@r.T+torch.tensor([2.,3.,4.]))
        torch.testing.assert_close(q,torch.zeros_like(q),atol=1e-14,rtol=0)

    def test_nonrigid_positive_gain_restores(self):
        self.activate()
        for scale in (1.2,.8):
            q=self.call(self.x*scale)
            self.assertGreater(q.abs().sum(),0)
            self.assertLess((q*(self.x*scale-self.x)).sum(),0)
        y=self.x.clone();y[...,0]+=.2*y[...,1]
        self.assertGreater(self.call(y).abs().sum(),0)

    def test_rotation_translation_equivariance(self):
        self.activate();y=self.x*1.2
        r=torch.linalg.qr(torch.tensor([[1.,2,3],[2,1,1],[3,1,2]],dtype=self.x.dtype))[0]
        torch.testing.assert_close(self.call(y@r.T+5,x=self.x@r.T+5),self.call(y)@r.T,atol=1e-14,rtol=1e-10)

    def test_edge_order_orientation_invariance(self):
        self.activate();y=self.x*1.2
        torch.testing.assert_close(self.call(y,edges=self.edges.flip(1).flip(2)),self.call(y))

    def test_empty_graph(self):
        self.activate()
        q=self.call(self.x,edges=self.edges[:,:0],mask=self.mask[:,:0])
        self.assertEqual(q.abs().sum(),0)
        self.assertEqual(self.call(self.x,edges=torch.full_like(self.edges,-1),mask=torch.zeros_like(self.mask)).abs().sum(),0)

    def test_bad_graphs_rejected(self):
        for bad in (torch.tensor([[[0,0]]]),torch.tensor([[[0,1],[1,0]]]),torch.tensor([[[0,7]]])):
            with self.assertRaises(ValueError):self.call(self.x,edges=bad,mask=torch.ones(bad.shape[:2],dtype=torch.bool))

    def test_collapse_and_nan_rejected(self):
        y=self.x.clone();y[:,1]=y[:,0]
        with self.assertRaises(ValueError):self.call(y)
        y=self.x.clone();y[:,0]=float('nan')
        with self.assertRaises(ValueError):self.call(y)

    def test_padding_and_disconnected_anchor(self):
        self.activate();self.valid[:,-1]=False;self.x[:,-1]=float('nan')
        q=self.call(self.x*1.2,edges=self.edges[:,:1],mask=self.mask[:,:1])
        self.assertTrue(torch.isfinite(q).all());self.assertEqual(q[:,2:].abs().sum(),0)

    def test_positions_gradient(self):
        self.activate();x=self.x.clone().requires_grad_();y=(self.x*1.2).requires_grad_()
        f=lambda a,b:self.call(b,x=a)
        self.assertTrue(torch.autograd.gradcheck(f,(x,y)))

    def test_transport_guard_composition(self):
        self.activate();u=self.x*.2;w=torch.tensor([[[.7,.1,.1,.1],[.1,.2,.3,.4]]],dtype=self.x.dtype)
        q=self.call(self.x+u)
        out,_,_=preserve_transport(u,q,w,torch.ones(1,2,dtype=torch.bool),self.valid,max_ratio=.25)
        torch.testing.assert_close((w@out).mean(1),(w@u).mean(1),atol=1e-14,rtol=0)

    def test_float32(self):
        self.head.float();self.x=self.x.float();self.activate()
        y=(self.x*1.2).requires_grad_();self.call(y).square().sum().backward()
        self.assertTrue(torch.isfinite(y.grad).all())

    def test_hinge_bending_without_chord_edges(self):
        self.activate()
        x=torch.tensor([[[0.,0,0],[1,0,0],[2,0,0],[3,0,0]]],dtype=self.x.dtype)
        y=torch.tensor([[[0.,0,0],[1,0,0],[1,1,0],[1,2,0]]],dtype=self.x.dtype)
        edges=torch.tensor([[[0,1],[1,2],[2,3]]])
        q=self.call(y,x=x,edges=edges,mask=torch.ones(1,3,dtype=torch.bool))
        torch.testing.assert_close(q,torch.zeros_like(q),atol=1e-14,rtol=0)

    def test_collinear_stretch_and_batched_masks(self):
        self.activate()
        x=torch.tensor([[[0.,0,0],[1,0,0],[2,0,0],[3,0,0]]],dtype=self.x.dtype).expand(2,-1,-1)
        edges=self.edges.expand(2,-1,-1);mask=self.mask.expand(2,-1).clone();mask[1]=False
        q=self.head(x,x*1.2,edges,mask,self.valid.expand(2,-1))
        self.assertTrue(torch.isfinite(q).all());self.assertGreater(q[0].abs().sum(),0);self.assertEqual(q[1].abs().sum(),0)


if __name__=='__main__':unittest.main()

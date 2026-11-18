import unittest
import torch
from scripts.experiments.local_response.correction import preserve_transport


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(42)
        self.u = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
        self.q = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
        self.w = torch.rand(2, 7, 4, generator=g, dtype=torch.float64)
        self.w /= self.w.sum(-1, keepdim=True)
        self.gm = torch.ones(2, 7, dtype=torch.bool)
        self.am = torch.ones(2, 4, dtype=torch.bool)

    def run_head(self, u=None, q=None, ratio=.25):
        return preserve_transport(self.u if u is None else u, self.q if q is None else q,
                                  self.w, self.gm, self.am, max_ratio=ratio)

    def test_identity(self):
        out, delta, _ = self.run_head(q=torch.zeros_like(self.q))
        self.assertTrue(torch.equal(out, self.u))
        self.assertEqual(delta.abs().max(), 0)

    def test_gs_weighted_not_anchor_mean(self):
        out, delta, _ = self.run_head(ratio=100)
        torch.testing.assert_close((self.w @ delta).mean(1), torch.zeros(2,3,dtype=self.u.dtype), atol=1e-14,rtol=0)
        torch.testing.assert_close((self.w @ out).mean(1),(self.w @ self.u).mean(1))
        self.assertGreater(delta.mean(1).abs().max().item(), 1e-3)

    def test_bound(self):
        _, d, _ = self.run_head()
        original = self.w @ self.u
        relative = original-original.mean(1,keepdim=True)
        self.assertTrue((torch.linalg.vector_norm((self.w@d).flatten(1),dim=1) <= .25*torch.linalg.vector_norm(relative.flatten(1),dim=1)+1e-12).all())

    def test_translation_and_zero_budget(self):
        u = torch.ones_like(self.u)*2
        torch.testing.assert_close(self.run_head(u=u)[0],u,atol=1e-14,rtol=0)
        self.assertTrue(torch.equal(self.run_head(ratio=0)[0],self.u))

    def test_rotation_equivariance(self):
        r=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=self.u.dtype)
        torch.testing.assert_close(self.run_head(u=self.u@r.T,q=self.q@r.T)[0],self.run_head()[0]@r.T)

    def test_padding_nan_and_empty(self):
        self.am[:,-1]=False;self.gm[:,-1]=False
        self.u[:,-1]=float('nan');self.q[:,-1]=float('nan')
        self.w[:,:,-1]=0;self.w/=self.w.sum(-1,keepdim=True);self.w[:,-1]=float('nan')
        out,d,_=self.run_head();self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close((torch.nan_to_num(self.w)@d)[:,:-1].mean(1),torch.zeros(2,3,dtype=self.u.dtype),atol=1e-14,rtol=0)
        self.gm[:]=False;self.w[:]=float('nan')
        self.assertEqual(self.run_head()[1].abs().max(),0)

    def test_invalid_input_rejected(self):
        self.w[0,0,0]=-1
        with self.assertRaises(ValueError):self.run_head()
        with self.assertRaises(ValueError):self.run_head(ratio=-1)

    def test_nonfinite_valid_rejected(self):
        self.q[0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.run_head()

    def test_gradient(self):
        u=self.u.clone().requires_grad_();q=self.q.clone().requires_grad_()
        f=lambda a,b:self.run_head(u=a,q=b)[0]
        self.assertTrue(torch.autograd.gradcheck(f,(u,q)))
        f(u,q).square().sum().backward()
        for x in (u,q):self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum(),0)

    def test_zero_initialization_gradient(self):
        q=torch.zeros_like(self.q,requires_grad=True)
        out,_,_=self.run_head(q=q)
        (out*self.q).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(q.grad.abs().sum(),0)


if __name__ == '__main__': unittest.main()

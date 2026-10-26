import unittest
import torch
from .gs_temporal import TemporalAnchorFusion
from .gs_template_refiner import TemplateRefiner
from .gs_decoder import build_initial_anchor_field


class TemporalFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)

    def test_zero_init_permutation_padding_and_independent_lengths(self):
        layer = TemporalAnchorFusion(16, 4)
        a, b = torch.randn(2, 3, 16), torch.randn(2, 7, 16)
        x, y = torch.randn(2, 3, 3), torch.randn(2, 7, 3)
        am = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        bm = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [0]*7], dtype=torch.bool)
        self.assertTrue((layer(a,b,x,y,am,bm) == 0).all())
        torch.nn.init.normal_(layer.output.weight, std=.1)
        baseline = layer(a,b,x,y,am,bm)
        order = torch.tensor([6,2,4,0,3,5,1])
        torch.testing.assert_close(baseline, layer(a,b[:,order],x,y[:,order],am,bm[:,order]))
        a[~am]=float('nan'); x[~am]=float('nan'); b[~bm]=float('nan'); y[~bm]=float('nan')
        torch.testing.assert_close(baseline, layer(a,b,x,y,am,bm), atol=0, rtol=0)
        self.assertTrue((baseline[1] == 0).all())

    def test_relative_xyz_and_both_frame_feature_gradients(self):
        layer = TemporalAnchorFusion(16,4).double()
        torch.nn.init.normal_(layer.output.weight, std=.1)
        a,b,x,y=[torch.randn(*shape,dtype=torch.float64,requires_grad=True)
                 for shape in ((1,3,16),(1,5,16),(1,3,3),(1,5,3))]
        out=layer(a,b,x,y,torch.ones(1,3,dtype=torch.bool),torch.ones(1,5,dtype=torch.bool))
        grads=torch.autograd.grad(out.square().sum(),(a,b,x,y))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads))

    def test_attribute_bounds_and_zero_initialization(self):
        keep=torch.tensor([[True,True,True,False]])
        xyz=torch.randn(1,4,3); field=build_initial_anchor_field(xyz,keep,3,2)
        attr=torch.randn(1,4,11); attr[~keep]=0
        head=TemplateRefiner(16); features=torch.randn(1,3,16)
        torch.testing.assert_close(head(features,field,attr,keep),attr,atol=0,rtol=0)
        with torch.no_grad():
            head.scale_head[-1].bias.fill_(1e6); head.opacity_head[-1].bias.fill_(-1e6)
        out=head(features,field,attr,keep)
        self.assertLessEqual(float((out[...,:3]-attr[...,:3]).abs().max()),head.SCALE_BOUND+1e-6)
        self.assertLessEqual(float((out[...,7:8]-attr[...,7:8]).abs().max()),.500001)
        torch.testing.assert_close(out[...,3:7],attr[...,3:7],atol=0,rtol=0)

if __name__=='__main__':unittest.main()

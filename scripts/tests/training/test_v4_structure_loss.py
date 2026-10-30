"""Frozen initial GS graph/weak edge-length penalty, CPU-only numerical checks."""
import copy
import math
import unittest
from unittest.mock import patch

import torch
from scripts.training.model import initial_knn
from scripts.experiments.v4_g5_struct_state.structure import (
    build_structure_graph, structure_loss, structure_loss_per_window,
    structure_statistics, EPSILON)


def line(n=20, dtype=torch.float64):
    x = torch.arange(n, dtype=dtype)*.01
    return torch.stack((x, x.square()*.3, x*0), -1)[None]


class StructureLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.xyz = line()
        self.mask = torch.ones(self.xyz.shape[:2], dtype=torch.bool)
        self.graph = build_structure_graph(self.xyz, self.mask)

    def test_calls_original_initial_knn_and_canonical_mutual_unique_edges(self):
        with patch('scripts.experiments.v4_g5_struct_state.structure.initial_knn', wraps=initial_knn) as original:
            graph = build_structure_graph(self.xyz, self.mask)
        original.assert_called_once()
        self.assertEqual(original.call_args.args[2], 16)
        ids, valid = initial_knn(self.xyz, self.mask, 16)
        candidates = {(i, int(j)) for i in range(self.xyz.shape[1]) for j, ok in zip(ids[0,i],valid[0,i]) if ok and i != int(j)}
        edges = [tuple(v) for v in graph.edges[0,graph.edge_mask[0]].tolist()]
        self.assertTrue(edges)
        self.assertEqual(edges, sorted(set(edges)))
        for i,j in edges:
            self.assertLess(i,j)
            self.assertIn((i,j),candidates); self.assertIn((j,i),candidates)
            rest = float(torch.linalg.vector_norm(self.xyz[0,i]-self.xyz[0,j]))
            scale = float(min(graph.local_spacing[0,i],graph.local_spacing[0,j]))
            self.assertGreater(rest,max(EPSILON,.1*scale)); self.assertLessEqual(rest,2*scale)

    def test_identity_translation_rotation_have_zero_loss(self):
        theta=.73
        rotation=torch.tensor([[math.cos(theta),-math.sin(theta),0.],[math.sin(theta),math.cos(theta),0.],[0.,0.,1.]],dtype=self.xyz.dtype)
        for xyz in (self.xyz, self.xyz+torch.tensor([.5,-.2,1.]), self.xyz@rotation.T+torch.tensor([.5,-.2,1.])):
            self.assertLess(float(structure_loss(xyz[:,None].expand(-1,15,-1,-1),self.graph)),1e-26)

    def test_tolerance_interval_is_unpenalized(self):
        for factor in (.8,.9,1.,1.1,1.2):
            value=structure_loss((self.xyz*factor)[:,None],self.graph)
            self.assertLess(float(value),1e-26)

    def test_stretch_and_compression_have_finite_nonzero_gradients(self):
        for factor in (.5,1.6):
            prediction=(self.xyz*factor)[:,None].clone().requires_grad_()
            loss=structure_loss(prediction,self.graph);loss.backward()
            self.assertGreater(float(loss.detach()),0)
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(float(prediction.grad.abs().sum()),0)

    def test_formula_is_smooth_l1_of_each_hinge_and_window_balanced(self):
        a=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]]],dtype=torch.float64)
        masks=torch.tensor([[True,True,False],[True,True,True]])
        initial=torch.cat((a,torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]],dtype=torch.float64)),0)
        graph=build_structure_graph(initial,masks)
        prediction=initial[:,None].clone();prediction[0]*=1.5;prediction[1]*=.5
        per=structure_loss_per_window(prediction,graph)
        # Both excesses are .3; SmoothL1 beta .1 gives .3-.05=.25.
        torch.testing.assert_close(per,torch.tensor([.25,.25],dtype=torch.float64),rtol=0,atol=1e-14)
        torch.testing.assert_close(structure_loss(prediction,graph),per.mean(),rtol=0,atol=0)

    def test_even_candidate_spacing_uses_midpoint_median(self):
        xyz=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]]],dtype=torch.float64)
        graph=build_structure_graph(xyz,torch.ones(1,3,dtype=torch.bool))
        torch.testing.assert_close(graph.local_spacing,torch.tensor([[2.5,2.,3.5]],dtype=torch.float64),rtol=0,atol=0)

    def test_duplicate_points_and_near_zero_lengths_do_not_enter_graph(self):
        xyz=torch.tensor([[[0.,0.,0.],[0.,0.,0.],[1e-8,0.,0.],[1.,0.,0.]]],dtype=torch.float64)
        graph=build_structure_graph(xyz,torch.ones(1,4,dtype=torch.bool))
        self.assertTrue((graph.rest_lengths[graph.edge_mask]>EPSILON).all())
        self.assertGreater(graph.stats[0]['near_zero_removed_directed'],0)
        prediction=xyz[:,None].clone().requires_grad_()
        loss=structure_loss(prediction,graph);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_all_duplicate_singleton_and_zero_point_graphs_are_differentiable_zero(self):
        for xyz in (torch.zeros(1,5,3,dtype=torch.float64),torch.zeros(1,1,3,dtype=torch.float64),torch.empty(1,0,3,dtype=torch.float64)):
            mask=torch.ones(xyz.shape[:2],dtype=torch.bool);graph=build_structure_graph(xyz,mask)
            prediction=xyz[:,None].clone().requires_grad_();loss=structure_loss(prediction,graph);loss.backward()
            self.assertEqual(float(loss.detach()),0);self.assertEqual(int(graph.edge_mask.sum()),0)
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_padding_and_invalid_nan_do_not_leak_loss_or_gradient(self):
        xyz=torch.cat((self.xyz,torch.full((1,4,3),float('nan'),dtype=self.xyz.dtype)),1)
        mask=torch.cat((self.mask,torch.zeros(1,4,dtype=torch.bool)),1)
        graph=build_structure_graph(xyz,mask)
        prediction=(xyz*1.5)[:,None].clone().requires_grad_()
        loss=structure_loss(prediction,graph);loss.backward()
        torch.testing.assert_close(loss,structure_loss((self.xyz*1.5)[:,None],self.graph),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:,:,~mask[0]].abs().sum()),0)
        self.assertTrue((graph.edges[graph.edge_mask]<20).all())

    def test_empty_edge_window_does_not_create_nan_or_reweight_other_windows(self):
        xyz=self.xyz.expand(2,-1,-1).clone();mask=self.mask.expand(2,-1).clone();mask[1]=False;xyz[1]=float('nan')
        graph=build_structure_graph(xyz,mask)
        prediction=(xyz*1.5)[:,None].clone().requires_grad_()
        per=structure_loss_per_window(prediction,graph)
        self.assertEqual(float(per[1].detach()),0)
        self.assertGreater(float(per[0].detach()),0)
        torch.testing.assert_close(structure_loss(prediction,graph),per[0]/2,rtol=0,atol=0)
        structure_loss(prediction,graph).backward();self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_far_gap_edges_filtered_without_inserting_rope_order_or_bridges(self):
        xyz=torch.cat((line(10),line(10)+torch.tensor([2.,0.,0.])),1)
        graph=build_structure_graph(xyz,torch.ones(1,20,dtype=torch.bool))
        edges=graph.edges[0,graph.edge_mask[0]]
        self.assertFalse(((edges[:,0]<10)&(edges[:,1]>=10)).any())
        self.assertGreater(graph.stats[0]['length_filtered_undirected'],0)

    def test_graph_is_fixed_initial_only_and_does_not_recompute_knn_in_loss(self):
        initial=self.xyz.clone().requires_grad_();graph=build_structure_graph(initial,self.mask)
        frozen=copy.deepcopy(graph)
        prediction=(self.xyz*1.7)[:,None].clone().requires_grad_()
        with patch('scripts.experiments.v4_g5_struct_state.structure.initial_knn',side_effect=AssertionError('No graph rebuild')):
            structure_loss(prediction,graph).backward()
        self.assertIsNone(initial.grad)
        self.assertTrue(torch.equal(graph.edges,frozen.edges));self.assertTrue(torch.equal(graph.rest_lengths,frozen.rest_lengths))
        self.assertTrue(torch.equal(graph.gs_mask,frozen.gs_mask))

    def test_repeatability_with_equal_distances_and_statistics(self):
        # More than 16 equidistant non-self candidates stresses the actual top-k
        # boundary; tie identities follow original initial_knn on this backend.
        axes=torch.cat((torch.eye(3,dtype=torch.float64),-torch.eye(3,dtype=torch.float64)),0).repeat(5,1)
        xyz=torch.cat((torch.zeros(1,3,dtype=torch.float64),axes),0)[None]
        mask=torch.ones(1,31,dtype=torch.bool);a=build_structure_graph(xyz,mask);b=build_structure_graph(xyz,mask)
        self.assertTrue(torch.equal(a.edges,b.edges));self.assertTrue(torch.equal(a.rest_lengths,b.rest_lengths))
        rows=structure_statistics((xyz*1.5)[:,None].expand(-1,15,-1,-1),a)
        self.assertEqual(len(rows),15)
        self.assertTrue(all(r['stretch_fraction']==1 and r['compression_fraction']==0 for r in rows))
        self.assertTrue(all(abs(r['ratio_mean']-1.5)<1e-14 for r in rows))

    def test_stats_empty_graph_uses_null_quantiles_not_nan(self):
        xyz=torch.zeros(1,1,3);graph=build_structure_graph(xyz,torch.ones(1,1,dtype=torch.bool))
        row=structure_statistics(xyz[:,None],graph)[0]
        self.assertEqual(row['edges'],0);self.assertIsNone(row['ratio_p50']);self.assertEqual(row['loss_struct_unweighted'],0)

    def test_valid_nonfinite_and_wrong_gs_or_anchor_shape_are_rejected(self):
        bad=self.xyz.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):build_structure_graph(bad,self.mask)
        with self.assertRaises(FloatingPointError):structure_loss(bad[:,None],self.graph)
        with self.assertRaises(ValueError):structure_loss(torch.zeros(1,15,128,3),self.graph)
        with self.assertRaises(ValueError):structure_loss(torch.zeros(1,0,20,3),self.graph)


if __name__=='__main__':unittest.main()

"""Mathematical CPU contracts for the frozen-graph C/D regularizers."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.experiments.v4_g5_arap_iso import arap, structure


def points(dtype=torch.float64):
    return torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1.3, 0.],
                          [0., 0., .7], [1., 1., 1.]]], dtype=dtype)


def graph_from_edges(xyz, edges, mask=None, edge_mask=None):
    """Small explicit initial graph, never built from predictions or GT."""
    b, n, _ = xyz.shape
    if mask is None:
        mask = torch.ones((b, n), dtype=torch.bool)
    ids = torch.tensor(edges, dtype=torch.long).reshape(1, -1, 2).expand(b, -1, -1).clone()
    if edge_mask is None:
        edge_mask = torch.ones(ids.shape[:2], dtype=torch.bool)
    batch = torch.arange(b)[:, None]
    lengths = torch.linalg.vector_norm(xyz[batch, ids[..., 0]]-xyz[batch, ids[..., 1]], dim=-1)
    lengths = torch.where(edge_mask, lengths, torch.ones_like(lengths))
    return structure.StructureGraph(ids, edge_mask, lengths.detach(), mask, xyz.new_zeros((b, n)), tuple({} for _ in range(b)))


def complete_graph(xyz):
    return graph_from_edges(xyz, [(i, j) for i in range(xyz.shape[1]) for j in range(i+1, xyz.shape[1])])


def rotate_z(angle, dtype=torch.float64):
    a = torch.tensor(angle, dtype=dtype)
    c, s = a.cos(), a.sin()
    return torch.stack((c, -s, a*0, s, c, a*0, a*0, a*0, a*0+1)).reshape(3, 3)


class FrozenGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_reviewed_graph_source_is_exact_unchanged_copy(self):
        path = Path(structure.__file__)
        digest = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        self.assertEqual(digest, 'c618ff57842697cea7a20f65ca3fe58d010dc0a1a600095b786d954929b929e7')

    def test_mutual_graph_unique_without_self_padding_or_zero_length(self):
        xyz = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.],
                             [float('nan'), float('nan'), float('nan')]]], dtype=torch.float64)
        mask = torch.tensor([[True, True, True, True, False]])
        g = structure.build_structure_graph(xyz, mask)
        edges = g.edges[0, g.edge_mask[0]]
        self.assertTrue((edges[:, 0] < edges[:, 1]).all())
        self.assertEqual(len(edges), len(torch.unique(edges, dim=0)))
        self.assertTrue((edges < 4).all())
        self.assertTrue((g.rest_lengths[g.edge_mask] > structure.EPSILON).all())
        self.assertFalse(g.edges.requires_grad); self.assertFalse(g.rest_lengths.requires_grad)
        self.assertGreater(g.stats[0]['near_zero_removed_directed'], 0)

    def test_graph_only_initial_and_frozen_for_entire_rollout(self):
        x = points().requires_grad_(); g = structure.build_structure_graph(x, torch.ones((1, 5), dtype=torch.bool))
        before = tuple(t.clone() for t in (g.edges, g.edge_mask, g.rest_lengths, g.local_spacing))
        arap.combined_regularizers(torch.stack((x*1.4, x*.6, x.flip(1)), 1), x, g)
        for a, b in zip(before, (g.edges, g.edge_mask, g.rest_lengths, g.local_spacing)):
            self.assertTrue(torch.equal(a, b))
        self.assertIsNone(x.grad)

    def test_explicit_length_filter_open_lower_closed_upper_and_midpoint_spacing(self):
        # Controlled ORIGINAL candidate kNN fixture: all other valid neighbors.
        x = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [20., 0., 0.]]], dtype=torch.float64)
        ids = torch.arange(4).view(1, 1, 4).expand(1, 4, 4)
        with patch.object(structure, 'initial_knn', return_value=(ids, torch.ones_like(ids, dtype=torch.bool))):
            g = structure.build_structure_graph(x, torch.ones((1, 4), dtype=torch.bool))
        self.assertTrue(torch.equal(g.local_spacing, torch.tensor([[2., 1., 2., 19.]], dtype=x.dtype)))
        self.assertTrue(torch.equal(g.edges[0, g.edge_mask[0]], torch.tensor([[0, 1], [0, 2], [1, 2]])))


class RegularizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_identity_and_pure_translation_are_zero(self):
        x = points(); g = complete_graph(x)
        p = torch.stack((x, x+3., x-4., x+.1), 1).requires_grad_()
        a, i, _ = arap.combined_regularizers(p, x, g)
        self.assertLess(float(a.detach()), 1e-27); self.assertLess(float(i.detach()), 1e-27)
        (a+i).backward(); self.assertTrue(torch.isfinite(p.grad).all())

    def test_independent_rigid_rotation_every_step_zero_and_correct_direction(self):
        x = points(); g = complete_graph(x)
        q1, q2 = rotate_z(.6), rotate_z(-.4)
        p = torch.stack((x@q1.T+2., x@q2.T-1.), 1)
        a, i, _ = arap.combined_regularizers(p, x, g)
        self.assertLess(float(a.detach()), 1e-27); self.assertLess(float(i.detach()), 1e-27)
        stats = arap.local_rigid_rotations(p, x, g)
        self.assertTrue(torch.allclose(stats['rotations'][0, 0], q1.expand(5, 3, 3), atol=1e-12, rtol=0))
        relative = q2@q1.T
        self.assertTrue(torch.allclose(stats['rotations'][0, 1], relative.expand(5, 3, 3), atol=1e-12, rtol=0))

    def test_reflection_is_not_accepted_as_SO3_rigid_motion(self):
        x = points(); g = complete_graph(x)
        p = x.clone(); p[..., 0].neg_(); p = p[:, None]
        a, i, _ = arap.combined_regularizers(p, x, g)
        self.assertGreater(float(a.detach()), .01); self.assertLess(float(i.detach()), 1e-27)
        s = arap.local_rigid_rotations(p, x, g)
        self.assertTrue(torch.allclose(torch.linalg.det(s['rotations']), torch.ones((1, 1, 5), dtype=x.dtype), atol=1e-12))
        self.assertGreater(int(s['kabsch_reflection_correction_count'].sum()), 0)

    def test_shear_is_penalized_and_produces_finite_gradient(self):
        x = points(); g = complete_graph(x)
        matrix = torch.tensor([[1., .4, 0.], [0., 1., 0.], [0., 0., 1.]], dtype=x.dtype)
        p = (x@matrix.T)[:, None].requires_grad_()
        a, i, _ = arap.combined_regularizers(p, x, g)
        self.assertGreater(float(a.detach()), 0.); self.assertGreater(float(i.detach()), 0.)
        (a+i).backward(); self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.norm()), 0.)

    def test_arap_compares_adjacent_times_not_initial_every_time(self):
        x = points(); g = complete_graph(x)
        p = torch.stack((x*1.4, x*1.4, x*1.4), 1)
        a, i, s = arap.combined_regularizers(p, x, g)
        self.assertGreater(float(s['arap_per_horizon'][0, 0]), 0.)
        self.assertLess(float(s['arap_per_horizon'][0, 1:].abs().max()), 1e-27)
        self.assertTrue(torch.allclose(s['iso_per_horizon'][0], i.expand(3), atol=1e-15, rtol=0))
        self.assertAlmostEqual(float(a.detach()), float(s['arap_per_horizon'][0, 0])/3, places=14)

    def test_arap_previous_and_current_predictions_both_receive_gradient(self):
        x = points(); g = complete_graph(x)
        p = torch.stack((x*1.2, x*.9), 1).requires_grad_()
        a, _, _ = arap.combined_regularizers(p, x, g)
        a.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad[:, 0].norm()), 0.)
        self.assertGreater(float(p.grad[:, 1].norm()), 0.)
        # With only the second transition retained, its source still learns.
        p2 = p.detach().clone().requires_grad_()
        old = p2[:, :1].clone()
        a2 = arap.arap_loss_per_window(p2, x, g)
        second_transition = a2*2-arap.arap_loss(old, x, g)
        second_transition.sum().backward()
        self.assertGreater(float(p2.grad[:, 0].norm()), 0.)

    def test_current_rotations_are_detached_but_envelope_gradient_matches_finite_difference(self):
        x = points(); g = complete_graph(x)
        generator = torch.Generator().manual_seed(91)
        p = (x[:, None].expand(-1, 2, -1, -1)+torch.randn((1, 2, 5, 3), generator=generator, dtype=x.dtype)*.07).requires_grad_()
        self.assertFalse(arap.local_rigid_rotations(p, x, g)['rotations'].requires_grad)
        self.assertTrue(torch.autograd.gradcheck(lambda v: arap.arap_loss(v, x, g), (p,), eps=1e-6, atol=1e-7, rtol=1e-4))

    def test_per_center_fixed_initial_squared_denominator_not_graph_weighted(self):
        x = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [8., 0., 0.]]], dtype=torch.float64)
        g = graph_from_edges(x, [(0, 1), (0, 2)])  # Last center is isolated.
        p = (x*1.3)[:, None]
        a, _, s = arap.combined_regularizers(p, x, g)
        denominator = torch.tensor([10., 1., 9.], dtype=x.dtype)
        expected = ((.3**2)*denominator/(denominator+arap.ARAP_DENOM_EPSILON)).mean()
        self.assertAlmostEqual(float(a.detach()), float(expected), places=14)
        self.assertEqual(s['center_count'].tolist(), [3])

    def test_iso_exact_log_smooth_l1_and_no_length_tolerance_dead_zone(self):
        x = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], dtype=torch.float64)
        g = graph_from_edges(x, [(0, 1)])
        for scale in (.6, .95, 1.05, 1.4):
            p = (x*scale)[:, None]
            value = arap.iso_loss(p, x, g)
            log = torch.log(torch.tensor((2*scale+1e-6)/(2+1e-6), dtype=x.dtype))
            expected = F.smooth_l1_loss(log, log*0, beta=.1)
            self.assertAlmostEqual(float(value), float(expected), places=15)
            self.assertGreater(float(value), 0.)

    def test_iso_uses_initial_rest_not_previous_time_and_counts_undirected_once(self):
        x = points(); g = complete_graph(x)
        p = torch.stack((x*1.1, x*1.3), 1)
        a, i, s = arap.combined_regularizers(p, x, g)
        manual = torch.stack([arap.iso_loss(p[:, j:j+1], x, g) for j in range(2)]).mean()
        self.assertTrue(torch.equal(i, manual)); self.assertEqual(s['edge_count'].tolist(), [10])

    def test_both_compression_and_stretch_have_finite_nonzero_gradients(self):
        x = points(); g = complete_graph(x)
        for scale in (.5, 1.5):
            p = (x*scale)[:, None].requires_grad_()
            a, i, _ = arap.combined_regularizers(p, x, g)
            for loss in (a, i):
                grad = torch.autograd.grad(loss, p, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.norm()), 0.)

    def test_collinear_and_planar_neighborhoods_not_rank_masked(self):
        for x, rank in ((torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]], dtype=torch.float64), 1),
                        (torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]], dtype=torch.float64), 2)):
            g = complete_graph(x); p = (x*1.3)[:, None].requires_grad_()
            a, _, s = arap.combined_regularizers(p, x, g, capture_solver_failures=True)
            self.assertGreater(float(a.detach()), 0.); self.assertEqual(int(s[f'kabsch_rank{rank}_count'].sum()), x.shape[1])
            a.backward(); self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.norm()), 0.)

    def test_collapsed_current_edges_are_counted_not_silently_removed(self):
        x = points(); g = complete_graph(x); p = torch.zeros((1, 1, 5, 3), dtype=x.dtype, requires_grad=True)
        a, i, s = arap.combined_regularizers(p, x, g)
        self.assertGreater(float(a.detach()), .9); self.assertTrue(torch.isfinite(i)); self.assertEqual(int(s['kabsch_rank0_count'].sum()), 5)
        (a+i).backward(); self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_edges_single_point_and_all_invalid_have_differentiable_zero(self):
        for x, mask in ((torch.zeros((1, 1, 3), dtype=torch.float64), torch.ones((1, 1), dtype=torch.bool)),
                        (torch.full((1, 3, 3), float('nan'), dtype=torch.float64), torch.zeros((1, 3), dtype=torch.bool))):
            g = structure.build_structure_graph(x, mask)
            p = x[:, None].expand(-1, 3, -1, -1).clone().requires_grad_()
            a, i, s = arap.combined_regularizers(p, x, g)
            self.assertEqual(float((a+i).detach()), 0.); (a+i).backward()
            self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))
            self.assertEqual(int(s['center_count'].sum()), 0)

    def test_padding_nan_isolation_and_zero_padding_gradient(self):
        x = torch.cat((points(), torch.full((1, 2, 3), float('nan'), dtype=torch.float64)), 1)
        mask = torch.tensor([[True]*5+[False]*2]); g = structure.build_structure_graph(x, mask)
        p = (x[:, None].expand(-1, 2, -1, -1)*1.2).clone().requires_grad_()
        a, i, _ = arap.combined_regularizers(p, x, g); (a+i).backward()
        self.assertTrue(torch.isfinite(a+i)); self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.equal(p.grad[:, :, 5:], torch.zeros_like(p.grad[:, :, 5:])))
        control = arap.combined_regularizers(p[:, :, :5].detach(), x[:, :5], structure.build_structure_graph(x[:, :5], mask[:, :5]))
        self.assertTrue(torch.equal(a, control[0])); self.assertTrue(torch.equal(i, control[1]))

    def test_nonfinite_valid_prediction_or_initial_and_invalid_graph_fail_explicitly(self):
        x = points(); g = complete_graph(x); p = x[:, None].clone(); p[0, 0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):arap.combined_regularizers(p, x, g)
        x2 = x.clone(); x2[0, 1, 0] = float('nan')
        with self.assertRaises(ValueError):arap.combined_regularizers(x[:, None], x2, g)
        bad = graph_from_edges(x, [(0, 0)])
        with self.assertRaises(ValueError):arap.combined_regularizers(x[:, None], x, bad)
        with self.assertRaises(ValueError):arap.combined_regularizers(x, x, g)

    def test_window_balancing_with_different_edge_counts_and_empty_window(self):
        x = points().expand(3, -1, -1).clone()
        ids = [(i, j) for i in range(5) for j in range(i+1, 5)]
        mask = torch.tensor([[True]*10, [True]+[False]*9, [False]*10])
        g = graph_from_edges(x, ids, edge_mask=mask)
        p = x[:, None]*torch.tensor([1.1, 1.4, 2.], dtype=x.dtype)[:, None, None, None]
        a, i, s = arap.combined_regularizers(p, x, g)
        self.assertTrue(torch.equal(a, s['arap_per_window'].mean())); self.assertTrue(torch.equal(i, s['iso_per_window'].mean()))
        self.assertEqual(float(s['arap_per_window'][2]), 0.); self.assertEqual(float(s['iso_per_window'][2]), 0.)
        for batch in range(2):
            single = structure.StructureGraph(g.edges[batch:batch+1], g.edge_mask[batch:batch+1], g.rest_lengths[batch:batch+1],
                g.gs_mask[batch:batch+1], g.local_spacing[batch:batch+1], ({},))
            av, iv, _ = arap.combined_regularizers(p[batch:batch+1], x[batch:batch+1], single)
            self.assertTrue(torch.equal(av, s['arap_per_window'][batch])); self.assertTrue(torch.equal(iv, s['iso_per_window'][batch]))

    def test_svd_batch_retry_and_identity_fallback_explicit_count_no_mask(self):
        x = points(); g = complete_graph(x); p = (x*1.2)[:, None].requires_grad_()
        original = torch.linalg.svd
        calls = []
        def solver(value, **kw):
            calls.append(value.shape)
            if len(calls) in (1, 2):raise RuntimeError('controlled backend failure')
            return original(value, **kw)
        with patch.object(torch.linalg, 'svd', side_effect=solver):
            a, _, s = arap.combined_regularizers(p, x, g, capture_solver_failures=True)
        self.assertEqual(int(s['kabsch_batch_retry_count'].sum()), 5)
        self.assertEqual(int(s['kabsch_fallback_count'].sum()), 1)
        self.assertEqual(s['solver_failures']['indices'].tolist(), [[0, 0, 0]])
        self.assertEqual(s['solver_failures']['covariance'].shape, (1, 3, 3))
        self.assertTrue(torch.equal(s['solver_failures']['rotation'], torch.eye(3, dtype=x.dtype)[None]))
        self.assertGreater(float(a.detach()), 0.); a.backward(); self.assertTrue(torch.isfinite(p.grad).all())

    def test_nonfinite_solver_result_uses_counted_identity_fallback(self):
        x = points(); g = complete_graph(x); original = torch.linalg.svd
        def solver(value, **kw):
            u, s, vh = original(value, **kw)
            u = u.clone(); u[0, 0, 0] = float('nan')
            return u, s, vh
        with patch.object(torch.linalg, 'svd', side_effect=solver):
            a, _, stats = arap.combined_regularizers((x*1.2)[:, None], x, g)
        self.assertTrue(torch.isfinite(a)); self.assertEqual(int(stats['kabsch_fallback_count'].sum()), 1)

    def test_float32_rank_deficient_paths_have_finite_outputs_and_gradients(self):
        x = points(torch.float32); g = complete_graph(x)
        p = (x[:, None].expand(-1, 15, -1, -1)*1.1).clone().requires_grad_()
        a, i, stats = arap.combined_regularizers(p, x, g); (a+i).backward()
        self.assertEqual(stats['arap_per_horizon'].shape, (1, 15)); self.assertTrue(torch.isfinite(a+i))
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    unittest.main()

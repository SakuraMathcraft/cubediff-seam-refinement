"""Checkpoint-free checks for the standalone research package."""

import unittest

import torch

from cubediff_seams.geometry import face_rays, overlap_band_diagnostic
from cubediff_seams.graph import LocalTrustGraph, OverlapGraph
from cubediff_seams.method import ScheduledGraphMethod
from cubediff_seams.method_api import MethodContext


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_metric_agrees_on_common_rays_and_detects_rotated_top(self):
        rays = face_rays(129)
        faces = (rays / rays.norm(dim=-1, keepdim=True) + 1).permute(0, 3, 1, 2) / 2
        result = overlap_band_diagnostic(faces)
        self.assertEqual(len(result['edges']), 12)
        self.assertLess(max(result['edges'].values()), 1e-4)
        faces[4] = torch.rot90(faces[4], 2, (-2, -1))
        self.assertGreater(overlap_band_diagnostic(faces)['mean_rgb_l1'], .05)

    def test_graph_adjoint_and_constrained_descent(self):
        rng = torch.Generator().manual_seed(19)
        graph = OverlapGraph(32, 'cpu')
        clean = torch.randn(6, 4, 32, 32, generator=rng)
        x = graph.flatten(clean) * graph.variable_mask[:, None]
        y = torch.randn(len(graph.indices), 4, generator=rng)
        torch.testing.assert_close((graph.residual(x) * y).sum(), (x * graph.transpose(y)).sum())
        corrected, record = LocalTrustGraph(graph).solve(clean, 1)
        self.assertLess(record['objective_after'], record['objective_before'])
        self.assertLessEqual(record['max_pixel_limit_ratio'], 1 + 1e-5)
        self.assertTrue(torch.equal(graph.flatten(corrected)[~graph.variable_mask],
                                    graph.flatten(clean)[~graph.variable_mask]))

    def test_method_changes_only_late_overlap_support(self):
        rng = torch.Generator().manual_seed(20)
        timesteps = tuple(range(981, 0, -20))
        alphas = torch.linspace(.999, .01, 1000)
        ctx = MethodContext(torch.randn(1, 4, 64, 64, generator=rng), alphas, timesteps, alphas[0], 20)
        method = ScheduledGraphMethod(ctx)
        sample = torch.randn(5, 4, 64, 64, generator=rng).half()
        epsilon = torch.randn(5, 4, 64, 64, generator=rng).half()
        outside = ~method.mask.expand_as(epsilon)
        for index, timestep in enumerate(timesteps):
            actual = method.update(epsilon, sample, timestep, index)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual[outside], epsilon[outside]))
            if index < 35:
                self.assertTrue(torch.equal(actual, epsilon))
        self.assertFalse(torch.equal(actual, epsilon))
        self.assertEqual(len(method.diagnostics), 15)


if __name__ == '__main__':
    unittest.main()

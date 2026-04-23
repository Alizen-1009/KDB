import unittest

import torch

from bytedance_pressure_round_modules import (
    SimpleMHA,
    SimpleMLA,
    SimpleMoE,
    hierarchical_block_reduce_sum,
    tree_reduce_sum,
)


class TestByteDancePressureRoundModules(unittest.TestCase):
    def test_mha_shape(self):
        x = torch.randn(2, 8, 32)
        model = SimpleMHA(dim=32, num_heads=4)
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_moe_shape(self):
        x = torch.randn(2, 8, 32)
        model = SimpleMoE(dim=32, hidden_dim=64, num_experts=4, top_k=2)
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_mla_shape(self):
        x = torch.randn(2, 8, 32)
        model = SimpleMLA(dim=32, num_heads=4, latent_dim=12)
        y, latent = model(x)
        self.assertEqual(y.shape, (2, 8, 32))
        self.assertEqual(latent.shape, (2, 8, 12))
        self.assertTrue(torch.isfinite(y).all())

    def test_tree_reduce_sum_matches_torch(self):
        x = torch.randn(1025)
        expected = x.sum()
        actual = tree_reduce_sum(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_hierarchical_reduce_sum_matches_torch(self):
        x = torch.randn(4097)
        expected = x.sum()
        actual = hierarchical_block_reduce_sum(x, block_size=256)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

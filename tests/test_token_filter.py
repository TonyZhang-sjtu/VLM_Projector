"""Check spatial alignment and gradient preservation at the selection boundary."""
import unittest

import torch

from src.data.token_filter import select_tokens


class TokenFilterTest(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.arange(12.0).reshape(4, 3).requires_grad_()
        self.coords = torch.arange(12).reshape(4, 3)

    def test_default_preserves_every_token_and_gradient(self):
        result = select_tokens(self.tokens, self.coords)
        self.assertIs(result.tokens, self.tokens)
        self.assertIs(result.coords, self.coords)
        self.assertEqual(result.kept_indices.tolist(), [0, 1, 2, 3])
        result.tokens.sum().backward()
        torch.testing.assert_close(self.tokens.grad, torch.ones_like(self.tokens))

    def test_custom_selection_keeps_coordinates_and_gradients_aligned(self):
        result = select_tokens(self.tokens, self.coords, policy="custom",
                               selector=lambda t, c: c[:, 0] >= 6)
        self.assertEqual(result.kept_indices.tolist(), [2, 3])
        torch.testing.assert_close(result.coords, self.coords[2:])
        result.tokens.sum().backward()
        expected = torch.zeros_like(self.tokens)
        expected[2:] = 1
        torch.testing.assert_close(self.tokens.grad, expected)

    def test_invalid_or_implicit_selection_is_rejected(self):
        selectors = [lambda t, c: torch.ones(4),
                     lambda t, c: torch.ones(3, dtype=torch.bool),
                     lambda t, c: torch.zeros(4, dtype=torch.bool)]
        for selector in selectors:
            with self.assertRaises(ValueError):
                select_tokens(self.tokens, self.coords, policy="custom", selector=selector)
        with self.assertRaises(ValueError):
            select_tokens(self.tokens, self.coords, selector=selectors[0])


if __name__ == "__main__":
    unittest.main()

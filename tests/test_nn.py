# -*- coding: utf-8 -*-
"""Тесты сети. Запуск: python -m unittest discover -s tests"""

import io
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets  # noqa: E402
import nn  # noqa: E402


class TestMath(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        probs = nn.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_softmax_is_stable_for_big_values(self):
        # без вычитания максимума exp(1000) даёт бесконечность
        probs = nn.softmax([1000.0, 1001.0])
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_loss_grows_with_wrong_answer(self):
        sure_right, _ = nn.loss_and_grad([10.0, 0.0], 0)
        sure_wrong, _ = nn.loss_and_grad([0.0, 10.0], 0)
        self.assertLess(sure_right, 0.01)
        self.assertGreater(sure_wrong, 5.0)

    def test_gradient_direction(self):
        # градиент по логитам должен уменьшать логит правильного класса
        _, grad = nn.loss_and_grad([0.0, 0.0], 1)
        self.assertLess(grad[1], 0.0)
        self.assertGreater(grad[0], 0.0)


class TestShapes(unittest.TestCase):
    def test_dense_forward_length(self):
        layer = nn.Dense(3, 5, __import__("random").Random(1))
        self.assertEqual(len(layer.forward([0.1, 0.2, 0.3])), 5)

    def test_backward_returns_input_gradient(self):
        layer = nn.Dense(3, 5, __import__("random").Random(1))
        layer.forward([0.1, 0.2, 0.3])
        self.assertEqual(len(layer.backward([1.0, 0.0, 0.0, 0.0, 0.0])), 3)

    def test_mlp_rejects_single_size(self):
        with self.assertRaises(ValueError):
            nn.MLP([2])


class TestGradient(unittest.TestCase):
    """Самый важный тест: ручной backward сверяется с численной производной."""

    def test_analytic_matches_numeric(self):
        net = nn.MLP([2, 4, 3], seed=3, hidden="tanh")
        x, y = [0.3, -0.7], 1

        # аналитический градиент
        logits = net.forward(x)
        _, grad = nn.loss_and_grad(logits, y)
        net.backward(grad)
        first = net.layers[0]

        eps = 1e-5
        for i in range(len(first.w)):
            for j in range(len(first.w[i])):
                original = first.w[i][j]

                first.w[i][j] = original + eps
                loss_plus, _ = nn.loss_and_grad(net.forward(x), y)

                first.w[i][j] = original - eps
                loss_minus, _ = nn.loss_and_grad(net.forward(x), y)

                first.w[i][j] = original

                numeric = (loss_plus - loss_minus) / (2 * eps)
                analytic = first.gw[i][j]
                self.assertAlmostEqual(
                    analytic, numeric, places=6,
                    msg=f"вес w[{i}][{j}]: аналитически {analytic:.8f}, численно {numeric:.8f}")


class TestLearning(unittest.TestCase):
    def test_xor_is_learned(self):
        net = nn.MLP([2, 8, 8, 2], seed=2, hidden="tanh")
        nn.train(net, datasets.xor(), epochs=500, lr=0.5, batch_size=4, verbose_every=0)
        self.assertEqual(nn.accuracy(net, datasets.xor()), 1.0)

    def test_loss_decreases(self):
        net = nn.MLP([2, 6, 2], seed=5, hidden="tanh")
        history = nn.train(net, datasets.xor(), epochs=200, lr=0.5,
                           batch_size=4, verbose_every=0)
        self.assertLess(history[-1], history[0])

    def test_spirals_partially_learned(self):
        data = datasets.two_spirals(points_per_class=40, seed=1)
        net = nn.MLP([2, 10, 10, 2], seed=1, hidden="tanh")
        nn.train(net, data, epochs=250, lr=0.3, batch_size=16, verbose_every=0)
        self.assertGreater(nn.accuracy(net, data), 0.85)

    def test_accuracy_is_fraction(self):
        net = nn.MLP([2, 4, 2], seed=1)
        value = nn.accuracy(net, datasets.xor())
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class TestDataset(unittest.TestCase):
    def test_spirals_are_balanced(self):
        data = datasets.two_spirals(points_per_class=30)
        labels = [label for _, label in data]
        self.assertEqual(len(data), 60)
        self.assertEqual(labels.count(0), 30)
        self.assertEqual(labels.count(1), 30)

    def test_spirals_are_deterministic(self):
        self.assertEqual(datasets.two_spirals(seed=42), datasets.two_spirals(seed=42))

    def test_limits_wrap_points(self):
        data = [[(-1.0, 2.0), 0], [(1.0, -2.0), 1]]
        x0, x1, y0, y1 = datasets.limits(data, margin=0.5)
        self.assertLess(x0, -1.0)
        self.assertGreater(x1, 1.0)
        self.assertLess(y0, -2.0)
        self.assertGreater(y1, 2.0)


class TestRender(unittest.TestCase):
    def test_svg_is_written_and_has_points(self):
        import tempfile
        import train as trainer

        data = datasets.two_spirals(points_per_class=10)
        net = nn.MLP([2, 4, 2], seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.svg")
            width, height, size = trainer.render_boundary(
                net, data, datasets.limits(data), path, grid=12, cell=6)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()

        self.assertGreater(size, 500)
        self.assertIn("<svg", text)
        self.assertEqual(text.count("<circle"), 20)
        self.assertTrue(math.isclose(width, 12 * 6 + 12))


if __name__ == "__main__":
    unittest.main(verbosity=2)

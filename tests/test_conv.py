# -*- coding: utf-8 -*-
"""Тесты свёрточной сети. Запуск: python -m unittest discover -s tests"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets  # noqa: E402
import nn  # noqa: E402


class TestConvShapes(unittest.TestCase):
    def test_output_size_and_filters(self):
        conv = nn.Conv2D(1, 9, 9, kernel=3, filters=4, rng=random.Random(1))
        maps = conv.forward([datasets.blank(9)])

        self.assertEqual(len(maps), 4)          # по карте на фильтр
        self.assertEqual(len(maps[0]), 7)       # 9 - 3 + 1
        self.assertEqual(len(maps[0][0]), 7)

    def test_backward_returns_image_gradient(self):
        conv = nn.Conv2D(1, 9, 9, kernel=3, filters=4, rng=random.Random(2))
        maps = conv.forward([datasets.blank(9)])
        grad = conv.backward(maps)

        self.assertEqual(len(grad), 1)
        self.assertEqual(len(grad[0]), 9)
        self.assertEqual(len(grad[0][0]), 9)

    def test_kernel_larger_than_image_is_rejected(self):
        with self.assertRaises(ValueError):
            nn.Conv2D(1, 4, 4, kernel=5, filters=2, rng=random.Random(1))

    def test_convnet_accepts_flat_image(self):
        conv = nn.Conv2D(1, 9, 9, kernel=3, filters=3, rng=random.Random(3))
        cnn = nn.ConvNet(conv, nn.GlobalMaxPool(),
                         nn.MLP([3, 4, 2], seed=3, hidden="tanh"))

        probs = cnn.probabilities(datasets.blank(9))
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_parameter_count(self):
        conv = nn.Conv2D(1, 9, 9, kernel=3, filters=8, rng=random.Random(4))
        cnn = nn.ConvNet(conv, nn.GlobalMaxPool(),
                         nn.MLP([8, 16, 2], seed=4, hidden="tanh"))

        # 8·1·3·3 + 8 смещений + (8·16+16) + (16·2+2) = 258
        self.assertEqual(cnn.parameters(), 258)


class TestPool(unittest.TestCase):
    def test_takes_maximum(self):
        pool = nn.GlobalMaxPool()
        self.assertEqual(pool.forward([[[1.0, 5.0], [3.0, 2.0]]]), [5.0])

    def test_gradient_goes_to_the_maximum(self):
        pool = nn.GlobalMaxPool()
        pool.forward([[[1.0, 5.0], [3.0, 2.0]]])
        grad = pool.backward([2.0])

        self.assertEqual(grad, [[[0.0, 2.0], [0.0, 0.0]]])

    def test_pooled_features_ignore_shift(self):
        # 3×3-шаблон, найденный в другом месте карты, даёт то же число
        conv = nn.Conv2D(1, 9, 9, kernel=3, filters=3, rng=random.Random(5))
        pool = nn.GlobalMaxPool()

        base = pool.forward(conv.forward([datasets.cross(size=9)]))
        moved = pool.forward(conv.forward([datasets.cross(size=9, center=(4, 5))]))

        for a, b in zip(base, moved):
            self.assertAlmostEqual(a, b, places=9)


class TestConvGradient(unittest.TestCase):
    """Ручной backward свёртки сверяется с численной производной."""

    def test_conv_weights_match_numeric(self):
        rng = random.Random(7)
        conv = nn.Conv2D(1, 5, 5, kernel=3, filters=2, rng=rng,
                         activation="tanh")
        cnn = nn.ConvNet(conv, nn.GlobalMaxPool(),
                         nn.MLP([2, 4, 2], seed=5, hidden="tanh"))

        image = [[rng.uniform(-1.0, 1.0) for _ in range(5)] for _ in range(5)]
        label = 1

        logits = cnn.forward(image)
        _, grad = nn.loss_and_grad(logits, label)
        cnn.backward(grad)

        eps = 1e-5
        for f in range(conv.n_filters):
            for ky in range(conv.k):
                for kx in range(conv.k):
                    original = conv.w[f][0][ky][kx]

                    conv.w[f][0][ky][kx] = original + eps
                    plus, _ = nn.loss_and_grad(cnn.forward(image), label)

                    conv.w[f][0][ky][kx] = original - eps
                    minus, _ = nn.loss_and_grad(cnn.forward(image), label)

                    conv.w[f][0][ky][kx] = original

                    numeric = (plus - minus) / (2 * eps)
                    analytic = conv.gw[f][0][ky][kx]
                    self.assertAlmostEqual(
                        analytic, numeric, places=5,
                        msg=f"фильтр {f}, вес [{ky}][{kx}]: "
                            f"аналитически {analytic:.8f}, численно {numeric:.8f}")


class TestConvLearning(unittest.TestCase):
    def test_cnn_learns_cross_and_square(self):
        data = datasets.shapes(size=7, per_class=16, shift=0, noise=0.05, seed=3)
        conv = nn.Conv2D(1, 7, 7, kernel=3, filters=4, rng=random.Random(2))
        cnn = nn.ConvNet(conv, nn.GlobalMaxPool(),
                         nn.MLP([4, 8, 2], seed=2, hidden="tanh"))

        nn.train(cnn, data, epochs=40, lr=0.2, batch_size=8, verbose_every=0)
        self.assertGreaterEqual(nn.accuracy(cnn, data), 0.9)


class TestImageData(unittest.TestCase):
    def test_shapes_are_balanced_and_deterministic(self):
        data = datasets.shapes(per_class=12, seed=9)
        labels = [label for _, label in data]

        self.assertEqual(len(data), 24)
        self.assertEqual(labels.count(0), 12)
        self.assertEqual(labels.count(1), 12)
        self.assertEqual(data, datasets.shapes(per_class=12, seed=9))

    def test_flatten_is_row_by_row(self):
        image = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(datasets.flatten(image), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_grids_are_square(self):
        for image, _ in datasets.shapes(size=9, per_class=2):
            self.assertEqual(len(image), 9)
            self.assertTrue(all(len(row) == 9 for row in image))


if __name__ == "__main__":
    unittest.main(verbosity=2)

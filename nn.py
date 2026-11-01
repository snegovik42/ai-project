# -*- coding: utf-8 -*-
"""Минимальная нейросеть на чистом Python.

Ни numpy, ни torch — только математика и списки. Этого хватает, чтобы
обучить сеть на двух спиралях или на картинках и увидеть, как она
проводит границу между классами.

Идея устройства:
    Dense         — один слой: веса, смещения, прямой проход и ручной backward;
    MLP           — цепочка слоёв, прямой и обратный проход по цепочке;
    Conv2D        — шаблон 3×3 скользит по картинке и даёт карту откликов;
    GlobalMaxPool — максимум по карте: неважно, где шаблон нашёлся;
    loss          — softmax + перекрёстная энтропия.
"""

import math
import random


# ---------------------------------------------------------------------------
#  Слой
# ---------------------------------------------------------------------------

class Dense:
    """Полносвязный слой: y = activation(W·x + b)."""

    def __init__(self, n_in, n_out, rng, activation="tanh"):
        self.activation = activation

        # инициализация: чем больше входов, тем меньше веса — иначе ReLU взрывается
        if activation == "relu":
            scale = math.sqrt(2.0 / n_in)
        else:
            scale = math.sqrt(1.0 / n_in)

        self.w = [[rng.gauss(0.0, scale) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out

        # накопленные градиенты (сбрасываются после шага)
        self.gw = [[0.0] * n_in for _ in range(n_out)]
        self.gb = [0.0] * n_out

        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        z = []
        for row, bias in zip(self.w, self.b):
            total = bias
            for weight, value in zip(row, x):
                total += weight * value
            z.append(total)
        self._z = z

        if self.activation == "relu":
            return [value if value > 0.0 else 0.0 for value in z]
        if self.activation == "tanh":
            return [math.tanh(value) for value in z]
        return z                     # linear

    def backward(self, grad_out):
        """Принимает dL/d(выход слоя), возвращает dL/d(вход)."""
        if self.activation == "relu":
            grad_z = [g if z > 0.0 else 0.0 for g, z in zip(grad_out, self._z)]
        elif self.activation == "tanh":
            # производная tanh: 1 - tanh²
            grad_z = [g * (1.0 - math.tanh(z) ** 2) for g, z in zip(grad_out, self._z)]
        else:
            grad_z = grad_out

        for i, g in enumerate(grad_z):
            row = self.w[i]
            g_row = self.gw[i]
            for j in range(len(row)):
                g_row[j] += g * self._x[j]
            self.gb[i] += g

        # dL/dx = Wᵀ · grad_z
        n_in = len(self.w[0])
        grad_in = [0.0] * n_in
        for i, g in enumerate(grad_z):
            row = self.w[i]
            for j in range(n_in):
                grad_in[j] += row[j] * g
        return grad_in

    def step(self, lr):
        for i, row in enumerate(self.w):
            g_row = self.gw[i]
            for j in range(len(row)):
                row[j] -= lr * g_row[j]
                g_row[j] = 0.0
            self.b[i] -= lr * self.gb[i]
            self.gb[i] = 0.0


# ---------------------------------------------------------------------------
#  Свёрточный слой
# ---------------------------------------------------------------------------

class Conv2D:
    """Свёртка: маленький шаблон (ядро) скользит по картинке.

    Вход  — стопка карт (каналы, высота, ширина), для серой картинки канал один.
    Выход — по одной карте на каждый фильтр: где шаблон нашёлся, там большое число.
    Весов у слоя всего `фильтры · каналы · k · k`, а не «вход × выход»:
    один и тот же шаблон проверяет всё изображение.
    """

    def __init__(self, in_channels, in_h, in_w, kernel, filters, rng, activation="relu"):
        if kernel > min(in_h, in_w):
            raise ValueError("ядро больше картинки")

        self.activation = activation
        self.k = kernel
        self.c_in = in_channels
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = in_h - kernel + 1
        self.out_w = in_w - kernel + 1
        self.n_filters = filters

        # инициализация как у Dense: чем больше входов у нейрона, тем меньше веса
        scale = math.sqrt(2.0 / (in_channels * kernel * kernel))
        self.w = [[[[rng.gauss(0.0, scale) for _ in range(kernel)]
                    for _ in range(kernel)]
                   for _ in range(in_channels)]
                  for _ in range(filters)]
        self.b = [0.0] * filters

        self.gw = [[[[0.0] * kernel for _ in range(kernel)]
                    for _ in range(in_channels)]
                   for _ in range(filters)]
        self.gb = [0.0] * filters

        self._x = None
        self._z = None

    def forward(self, x):
        """x: список каналов, каждый канал — список строк."""
        self._x = x
        k = self.k
        z = []

        for f in range(self.n_filters):
            kernel = self.w[f]
            maps = []
            for oy in range(self.out_h):
                row = []
                for ox in range(self.out_w):
                    total = self.b[f]
                    for c in range(self.c_in):
                        channel = x[c]
                        kc = kernel[c]
                        for ky in range(k):
                            src = channel[oy + ky]
                            krow = kc[ky]
                            for kx in range(k):
                                total += krow[kx] * src[ox + kx]
                    row.append(total)
                maps.append(row)
            z.append(maps)

        self._z = z
        if self.activation == "relu":
            return [[[v if v > 0.0 else 0.0 for v in row] for row in m] for m in z]
        if self.activation == "tanh":
            return [[[math.tanh(v) for v in row] for row in m] for m in z]
        return z

    def backward(self, grad_out):
        if self.activation == "relu":
            grad_z = [[[g if z > 0.0 else 0.0 for g, z in zip(grow, zrow)]
                       for grow, zrow in zip(gmap, zmap)]
                      for gmap, zmap in zip(grad_out, self._z)]
        elif self.activation == "tanh":
            grad_z = [[[g * (1.0 - math.tanh(z) ** 2) for g, z in zip(grow, zrow)]
                       for grow, zrow in zip(gmap, zmap)]
                      for gmap, zmap in zip(grad_out, self._z)]
        else:
            grad_z = grad_out

        k = self.k
        grad_in = [[[0.0] * self.in_w for _ in range(self.in_h)]
                   for _ in range(self.c_in)]

        for f in range(self.n_filters):
            gz = grad_z[f]
            kernel = self.w[f]
            gw_f = self.gw[f]
            for oy in range(self.out_h):
                gz_row = gz[oy]
                for ox in range(self.out_w):
                    g = gz_row[ox]
                    if g == 0.0:
                        continue
                    self.gb[f] += g
                    for c in range(self.c_in):
                        kc = kernel[c]
                        gwc = gw_f[c]
                        gch = grad_in[c]
                        src_ch = self._x[c]
                        for ky in range(k):
                            krow = kc[ky]
                            gw_row = gwc[ky]
                            grow = gch[oy + ky]
                            src_row = src_ch[oy + ky]
                            for kx in range(k):
                                gw_row[kx] += g * src_row[ox + kx]
                                grow[ox + kx] += krow[kx] * g

        return grad_in

    def step(self, lr):
        for f in range(self.n_filters):
            kernel = self.w[f]
            gw_f = self.gw[f]
            for c in range(self.c_in):
                for ky in range(self.k):
                    for kx in range(self.k):
                        kernel[c][ky][kx] -= lr * gw_f[c][ky][kx]
                        gw_f[c][ky][kx] = 0.0
            self.b[f] -= lr * self.gb[f]
            self.gb[f] = 0.0


class GlobalMaxPool:
    """Из каждой карты признаков берём один максимум.

    Заодно решается главная проблема: неважно, где нашёлся шаблон —
    в левом верхнем углу или в правом нижнем, число одно и то же.
    """

    def __init__(self):
        self._arg = None
        self._shape = None

    def forward(self, x):
        self._shape = (len(x), len(x[0]), len(x[0][0]))
        self._arg = []
        out = []

        for channel in x:
            best = None
            best_y = best_x = 0
            for y, row in enumerate(channel):
                for xi, value in enumerate(row):
                    if best is None or value > best:
                        best, best_y, best_x = value, y, xi
            out.append(best if best is not None else 0.0)
            self._arg.append((best_y, best_x))

        return out

    def backward(self, grad):
        channels, height, width = self._shape
        grad_in = [[[0.0] * width for _ in range(height)] for _ in range(channels)]
        for c, g in enumerate(grad):
            y, x = self._arg[c]
            grad_in[c][y][x] = g
        return grad_in

    def step(self, lr):
        pass                        # учить нечего: параметров нет


# ---------------------------------------------------------------------------
#  Сеть
# ---------------------------------------------------------------------------

class MLP:
    """Полносвязная сеть из нескольких слоёв."""

    def __init__(self, sizes, seed=1, hidden="tanh"):
        if len(sizes) < 2:
            raise ValueError("нужно хотя бы два числа в sizes")

        rng = random.Random(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            self.layers.append(
                Dense(sizes[i], sizes[i + 1], rng, "linear" if last else hidden))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def predict(self, x):
        logits = self.forward(x)
        return max(range(len(logits)), key=lambda i: logits[i])

    def probabilities(self, x):
        return softmax(self.forward(x))

    def parameters(self):
        return sum(len(layer.w) * len(layer.w[0]) + len(layer.b)
                   for layer in self.layers)


class ConvNet:
    """Свёртка → максимум по карте → обычная полносвязная «голова».

    Свёртка и пулинг работают с картинкой, голова — с вектором чисел,
    поэтому её можно взять готовую: это тот же MLP.
    """

    def __init__(self, conv, pool, head):
        self.conv = conv
        self.pool = pool
        self.head = head

    def forward(self, image):
        maps = self.conv.forward(_as_channels(image))
        features = self.pool.forward(maps)
        return self.head.forward(features)

    def backward(self, grad):
        grad = self.head.backward(grad)
        grad = self.pool.backward(grad)
        return self.conv.backward(grad)

    def step(self, lr):
        self.conv.step(lr)
        self.pool.step(lr)
        self.head.step(lr)

    def predict(self, image):
        logits = self.forward(image)
        return max(range(len(logits)), key=lambda i: logits[i])

    def probabilities(self, image):
        return softmax(self.forward(image))

    def parameters(self):
        return (self.conv.n_filters * self.conv.c_in * self.conv.k * self.conv.k
                + self.conv.n_filters + self.head.parameters())


def _as_channels(image):
    """Серая картинка (список строк) превращается в картинку с одним каналом."""
    if image and image[0] and isinstance(image[0][0], list):
        return image
    return [image]


# ---------------------------------------------------------------------------
#  Функция потерь
# ---------------------------------------------------------------------------

def softmax(logits):
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def loss_and_grad(logits, label):
    """Перекрёстная энтропия и её градиент по логитам."""
    probs = softmax(logits)
    loss = -math.log(max(probs[label], 1e-12))

    # градиент softmax+CE по логитам — просто «вероятности минус one-hot»
    grad = probs[:]
    grad[label] -= 1.0
    return loss, grad


# ---------------------------------------------------------------------------
#  Обучение
# ---------------------------------------------------------------------------

def train(net, data, epochs=300, lr=0.3, batch_size=16, verbose_every=50, log=print):
    """Обычный SGD по мини-батчам. Возвращает историю потерь."""
    history = []
    rng = random.Random(7)
    order = list(range(len(data)))

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        total_loss = 0.0

        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            for idx in batch:
                x, y = data[idx]
                logits = net.forward(x)
                loss, grad = loss_and_grad(logits, y)
                total_loss += loss
                net.backward(grad)
            net.step(lr / len(batch))     # усредняем по батчу

        avg = total_loss / len(data)
        history.append(avg)

        if verbose_every and epoch % verbose_every == 0:
            log(f"эпоха {epoch:4d}   потеря {avg:.4f}   точность {accuracy(net, data):.1%}")

    return history


def accuracy(net, data):
    right = sum(1 for x, y in data if net.predict(x) == y)
    return right / len(data)

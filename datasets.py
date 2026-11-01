# -*- coding: utf-8 -*-
"""Данные для обучения: две спирали и XOR. Никаких скачиваний — всё генерируется."""

import math
import random


def two_spirals(points_per_class=80, turns=1.0, noise=0.04, seed=1):
    """Две переплетённые спирали — задача, где линейная модель беспомощна."""
    rng = random.Random(seed)
    data = []

    for i in range(points_per_class):
        radius = i / points_per_class
        angle = i / points_per_class * turns * 2.0 * math.pi

        # вторая спираль — поворот на 180°: те же лучи, но с другой стороны
        for sign, label in ((1.0, 0), (-1.0, 1)):
            x = sign * radius * math.cos(angle) + rng.gauss(0.0, noise)
            y = sign * radius * math.sin(angle) + rng.gauss(0.0, noise)
            data.append(([x, y], label))

    rng.shuffle(data)
    return data


def xor():
    """Четыре точки, которые нельзя разделить одной прямой."""
    return [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 0),
    ]


def limits(data, margin=0.25):
    xs = [point[0][0] for point in data]
    ys = [point[0][1] for point in data]
    return (min(xs) - margin, max(xs) + margin,
            min(ys) - margin, max(ys) + margin)


# ---------------------------------------------------------------------------
#  Картинки: плюс и рамка
# ---------------------------------------------------------------------------

def blank(size=9, value=0.0):
    return [[value] * size for _ in range(size)]


def _put(grid, y, x, value=1.0):
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        grid[y][x] = value


def cross(size=9, center=None, arm=2, value=1.0):
    """Плюс: горизонтальная и вертикальная полоски."""
    grid = blank(size)
    cy, cx = center or (size // 2, size // 2)
    for d in range(-arm, arm + 1):
        _put(grid, cy, cx + d, value)
        _put(grid, cy + d, cx, value)
    return grid


def square(size=9, center=None, half=2, value=1.0):
    """Рамка: контур квадрата."""
    grid = blank(size)
    cy, cx = center or (size // 2, size // 2)
    for d in range(-half, half + 1):
        _put(grid, cy - half, cx + d, value)
        _put(grid, cy + half, cx + d, value)
        _put(grid, cy + d, cx - half, value)
        _put(grid, cy + d, cx + half, value)
    return grid


def shapes(size=9, per_class=80, shift=0, noise=0.08, seed=1, arm=2, half=2):
    """Датасет картинок: 0 — плюс, 1 — рамка.

    shift — на сколько клеток центр может гулять от середины. Обучение на
    shift=0 даёт модели, которая видела фигуры только по центру.
    """
    rng = random.Random(seed)
    data = []

    for label, maker in ((0, cross), (1, square)):
        extra = {"arm": arm} if maker is cross else {"half": half}
        for _ in range(per_class):
            dy = rng.randint(-shift, shift)
            dx = rng.randint(-shift, shift)
            center = (size // 2 + dy, size // 2 + dx)
            grid = maker(size, center=center, **extra)

            # немного шума по яркости: иначе задача слишком стерильная
            for row in grid:
                for x in range(size):
                    row[x] += rng.gauss(0.0, noise)

            data.append((grid, label))

    rng.shuffle(data)
    return data


def flatten(image):
    """Картинка (список строк) → плоский вектор для обычного MLP."""
    return [value for row in image for value in row]

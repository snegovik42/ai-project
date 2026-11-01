#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Обучение сети на двух спиралях + картинка границы решений в SVG.

    python train.py                 # 300 эпох, как в README
    python train.py --epochs 800    # точнее, но дольше
    python train.py --out docs/boundary.svg
"""

import argparse
import os
import time

import datasets
import nn

# цвета классов: 0 — фиолетовый, 1 — бирюзовый (по 5 уровней уверенности)
CLASS_COLORS = {
    0: ["#33224A", "#432C60", "#523676", "#623F8C", "#7149A2"],
    1: ["#123B36", "#175049", "#1C655C", "#217A6F", "#268F82"],
}


def render_boundary(net, data, bounds, path, grid=96, cell=7, pad=6):
    """Рисует уверенность сети на сетке и накладывает точки данных."""
    x0, x1, y0, y1 = bounds
    size = grid * cell
    width = size + pad * 2
    height = size + pad * 2 + 44

    parts = [f'<rect width="{width}" height="{height}" fill="#14111B"/>']

    for gy in range(grid):
        # склеиваем соседние одинаковые клетки в один прямоугольник:
        # без этого файл весит сотни килобайт вместо десятков
        runs = []
        prev = None
        start = 0

        for gx in range(grid):
            cx = x0 + (gx + 0.5) * (x1 - x0) / grid
            cy = y1 - (gy + 0.5) * (y1 - y0) / grid
            probs = net.probabilities([cx, cy])

            label = 0 if probs[0] >= probs[1] else 1
            confidence = max(probs)
            level = max(0, min(4, int((confidence - 0.5) * 8)))
            key = (label, level)

            if key != prev:
                if prev is not None:
                    runs.append((prev, start, gx))
                prev = key
                start = gx

        runs.append((prev, start, grid))

        for (label, level), a, b in runs:
            parts.append(
                f'<rect x="{pad + a * cell}" y="{pad + gy * cell}" '
                f'width="{(b - a) * cell}" height="{cell}" '
                f'fill="{CLASS_COLORS[label][level]}" opacity="0.6"/>')

    for (point, label) in data:
        px = pad + (point[0] - x0) / (x1 - x0) * size
        py = pad + (y1 - point[1]) / (y1 - y0) * size
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="#EFEAF7" '
            f'stroke="#0F0C16" stroke-width="1.1"/>')

    parts.append(
        f'<text x="{pad}" y="{height - 18}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" fill="#A99FC0">граница решений · сеть 2-10-10-2 · '
        f'ярче = увереннее</text>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Граница решений нейросети на двух спиралях">\n  '
           + "\n  ".join(parts) + "\n</svg>\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return width, height, len(svg)


def main():
    parser = argparse.ArgumentParser(description="Нейросеть на двух спиралях")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--points", type=int, default=80, help="точек на класс")
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--out", default="docs/boundary.svg")
    args = parser.parse_args()

    data = datasets.two_spirals(points_per_class=args.points)
    net = nn.MLP([2, args.hidden, args.hidden, 2], seed=1, hidden="tanh")

    print(f"точек: {len(data)}   сеть: 2-{args.hidden}-{args.hidden}-2   "
          f"эпох: {args.epochs}")

    started = time.time()
    history = nn.train(net, data, epochs=args.epochs, lr=args.lr, verbose_every=50)
    spent = time.time() - started

    train_acc = nn.accuracy(net, data)
    print(f"\nточность на обучающих данных: {train_acc:.1%}")
    print(f"потеря: {history[0]:.4f} → {history[-1]:.4f}")
    print(f"время: {spent:.1f} с")

    width, height, size = render_boundary(net, data, datasets.limits(data), args.out)
    print(f"картинка: {args.out} ({width}×{height}, {size / 1024:.0f} КБ)")

    if train_acc < 0.85:
        print("\nвнимание: точность ниже 85% — попробуй --epochs 600")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

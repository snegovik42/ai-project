#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Свёрточная сеть на картинках: плюс или рамка.

    python conv_demo.py                 # обучение и docs/conv.svg
    python conv_demo.py --epochs 100    # дольше, но можно сравнить

Обычный MLP видит пиксели по отдельности и запоминает фигуру в центре.
Свёртка ищет маленький шаблон в любом месте картинки, а глобальный
максимум выбирает лучший отклик — поэтому сдвиг фигуры её не смущает.
Обе сети учатся только на центрированных картинках, а проверяются
на сдвинутых: в этом и весь смысл сравнения.
"""

import argparse
import os
import random
import time

import datasets
import nn

# палитра та же, что в train.py: тёмный фон, фиолетовый и бирюзовый
BG = "#14111B"
PANEL = "#1D1827"
GRID = "#2A2338"
INK = "#EFEAF7"
MUTED = "#A99FC0"
LILAC = "#7149A2"
TEAL = "#268F82"
BRIGHT_TEAL = "#5AD9C6"
WRONG = "#D96A7A"

NAME = {0: "плюс", 1: "рамка"}
WIDTH = 976
HEIGHT = 700


# ---------------------------------------------------------------------------
#  Маленькие помощники для SVG
# ---------------------------------------------------------------------------

def _mix(a, b, t):
    """Смешивает два цвета #rrggbb: t=0 — это a, t=1 — это b."""
    t = max(0.0, min(1.0, t))
    lo = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    hi = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02X}"
                         for x, y in zip(lo, hi))


def _text(x, y, body, size=12, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{body}</text>')


def _grid(parts, x, y, image, cell, color):
    """Рисует картинку клетками; color(значение) задаёт цвет клетки."""
    for row_i, row in enumerate(image):
        for col_i, value in enumerate(row):
            parts.append(
                f'<rect x="{x + col_i * cell}" y="{y + row_i * cell}" '
                f'width="{cell - 1}" height="{cell - 1}" fill="{color(value)}"/>')

    width = len(image[0]) * cell
    height = len(image) * cell
    parts.append(f'<rect x="{x - 0.5}" y="{y - 0.5}" width="{width}" '
                 f'height="{height}" fill="none" stroke="{GRID}"/>')


def _image_color(value):
    # пиксель 0 — тёмный, 1 — светлый; шум слегка подсвечивает фон
    return _mix(PANEL, "#8FE9D8", value)


def _kernel_color(value, scale):
    color = TEAL if value >= 0.0 else LILAC
    return _mix(PANEL, color, abs(value) / scale if scale else 0.0)


def _guess(net, item, flatten=False):
    """Класс и уверенность сети; MLP просит плоский вектор, CNN — картинку."""
    x = datasets.flatten(item) if flatten else item
    probs = net.probabilities(x)
    label = max(range(len(probs)), key=lambda i: probs[i])
    return label, probs[label]


# ---------------------------------------------------------------------------
#  Обучение
# ---------------------------------------------------------------------------

def build_cnn(size=9, filters=8, hidden=16, seed=1):
    conv = nn.Conv2D(1, size, size, kernel=3, filters=filters,
                     rng=random.Random(seed))
    head = nn.MLP([filters, hidden, 2], seed=seed, hidden="tanh")
    return nn.ConvNet(conv, nn.GlobalMaxPool(), head)


def flatten_data(data):
    return [(datasets.flatten(image), label) for image, label in data]


# ---------------------------------------------------------------------------
#  Картинка
# ---------------------------------------------------------------------------

def render_conv(cnn, mlp, centered, shifted, scores, path):
    """Собирает SVG: примеры, фильтры, карты откликов и точность на сдвигах."""
    n = cnn.conv.n_filters
    parts = [f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>']

    parts.append(_text(24, 38, "Свёртка: шаблон ищется где угодно", 21, INK, "600"))
    parts.append(_text(24, 60, f"плюс или рамка · картинка 9×9 · {n} фильтров 3×3 · "
                               "обучение только на центрированных фигурах", 12.5, MUTED))
    parts.append(f'<line x1="24" y1="76" x2="{WIDTH - 24}" y2="76" stroke="{GRID}"/>')

    # --- входы -------------------------------------------------------------
    parts.append(_text(24, 104, "1. что видит сеть", 13.5, INK, "600"))
    parts.append(_text(24, 124, "примеры из тестовой выборки", 11, MUTED))

    samples = ((centered, 136, "плюс по центру", "так учились"),
               (shifted, 286, "плюс сдвинут", "так не учились"))

    for image, top, title, note in samples:
        _grid(parts, 24, top, image, 14, _image_color)
        parts.append(_text(162, top + 20, title, 13, INK, "600"))
        parts.append(_text(162, top + 38, note, 10.5, MUTED))

        for j, (name, net, flat) in enumerate((("CNN", cnn, False),
                                               ("MLP", mlp, True))):
            label, confidence = _guess(net, image, flat)
            color = TEAL if label == 0 else WRONG      # обе картинки — плюсы
            parts.append(_text(162, top + 66 + j * 20,
                               f"{name} → {NAME[label]} {confidence:.0%}",
                               11, color, "600"))

    # --- фильтры -----------------------------------------------------------
    maps = cnn.conv.forward([centered])
    pooled = cnn.pool.forward(maps)
    order = sorted(range(cnn.conv.n_filters), key=lambda f: -pooled[f])

    parts.append(_text(360, 104, "2. выученные фильтры 3×3", 13.5, INK, "600"))
    parts.append(_text(360, 124, "бирюза — плюс, фиолетовый — минус", 11, MUTED))

    scale = max(abs(v) for filt in cnn.conv.w for ch in filt
                for row in ch for v in row) or 1.0

    for slot, f in enumerate(order[:8]):
        col, row = slot % 4, slot // 4
        x, y = 360 + col * 78, 140 + row * 98
        _grid(parts, x, y, cnn.conv.w[f][0], 20,
              lambda v, s=scale: _kernel_color(v, s))
        parts.append(_text(x, y + 74, f"b = {cnn.conv.b[f]:+.2f}", 9.5, MUTED))

    conv_weights = cnn.conv.n_filters * cnn.conv.c_in * cnn.conv.k * cnn.conv.k
    parts.append(_text(360, 344, f"свёртка: {n} фильтров × 3×3 = "
                                 f"{conv_weights} веса", 11, MUTED))
    parts.append(_text(360, 366, "пулинг: максимум по карте, учить нечего", 11, MUTED))
    parts.append(_text(360, 388, f"голова {n}-16-2: "
                                 f"{cnn.head.parameters()} весов", 11, MUTED))
    parts.append(_text(360, 410, f"итого {cnn.parameters()} параметров",
                       11, BRIGHT_TEAL, "600"))

    # --- карты откликов ----------------------------------------------------
    parts.append(_text(690, 104, "3. карты откликов", 13.5, INK, "600"))
    parts.append(_text(690, 124, "ярче — шаблон нашёлся сильнее", 11, MUTED))

    feature_scale = max((max(max(row) for row in m) for m in maps), default=0.0) or 1.0

    for slot, f in enumerate(order[:8]):
        col, row = slot % 2, slot // 2
        x, y = 694 + col * 96, 140 + row * 82
        _grid(parts, x, y, maps[f], 9,
              lambda v, s=feature_scale: _mix(PANEL, BRIGHT_TEAL, v / s))
        parts.append(_text(x, y + 72, f"max {pooled[f]:.2f}", 9.5, MUTED))

    # --- точность на сдвигах ----------------------------------------------
    parts.append(_text(24, 480, "4. что будет, если сдвинуть фигуру", 13.5, INK, "600"))
    parts.append(_text(24, 498, "обе сети учились только на фигурах по центру; "
                                "ниже — точность на новых картинках", 11, MUTED))

    parts.append(f'<rect x="700" y="468" width="12" height="12" rx="2" fill="{TEAL}"/>')
    parts.append(_text(718, 478, "свёрточная сеть", 11, MUTED))
    parts.append(f'<rect x="830" y="468" width="12" height="12" rx="2" fill="{LILAC}"/>')
    parts.append(_text(848, 478, "обычный MLP", 11, MUTED))

    left, top, width = 170, 534, 480
    bottom = top + 3 * 42 - 6

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = left + width * frac
        parts.append(f'<line x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" '
                     f'y2="{bottom}" stroke="{GRID}"/>')
        parts.append(_text(gx, top - 16, f"{frac:.0%}", 9.5, MUTED,
                           anchor="middle"))

    for i, name in enumerate(("по центру", "сдвиг 1", "сдвиг 2")):
        y = top + i * 42
        parts.append(_text(24, y + 10, name, 12, INK))

        for j, (key, color) in enumerate((("cnn", TEAL), ("mlp", LILAC))):
            value = scores[key][i]
            bar_y = y + j * 18
            parts.append(f'<rect x="{left}" y="{bar_y}" '
                         f'width="{value * width:.1f}" height="13" rx="3" '
                         f'fill="{color}"/>')
            parts.append(_text(left + value * width + 7, bar_y + 10.5,
                               f"{value:.0%}", 10.5, INK))

    parts.append(_text(24, 676, f"параметров: свёрточная {cnn.parameters()} · "
                                f"MLP {mlp.parameters()}", 11, MUTED))
    parts.append(_text(WIDTH - 24, 676, "вход 9×9 → 8 карт 7×7 → 8 чисел → 2 класса",
                       11, MUTED, anchor="end"))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
           f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
           f'aria-label="Свёрточная сеть ищет плюс или рамку в любом месте картинки">\n  '
           + "\n  ".join(parts) + "\n</svg>\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return WIDTH, HEIGHT, len(svg)


# ---------------------------------------------------------------------------
#  Запуск
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Свёрточная сеть на картинках")
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=60,
                        help="эпох для свёрточной сети")
    parser.add_argument("--epochs-mlp", type=int, default=100,
                        help="эпох для обычного MLP")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="docs/conv.svg")
    args = parser.parse_args()

    train = datasets.shapes(per_class=args.per_class, shift=0, seed=args.seed)
    tests = [datasets.shapes(per_class=args.test_per_class, shift=shift,
                             seed=args.seed + 10 + shift)
             for shift in (0, 1, 2)]

    cnn = build_cnn(filters=args.filters, seed=args.seed)
    mlp = nn.MLP([81, 32, 2], seed=args.seed, hidden="tanh")

    print(f"картинок: {len(train)}   фильтров: {args.filters}   "
          f"CNN {args.epochs} эпох, MLP {args.epochs_mlp} эпох")

    started = time.time()
    cnn_history = nn.train(cnn, train, epochs=args.epochs, lr=args.lr,
                           batch_size=16, verbose_every=0)
    cnn_time = time.time() - started

    flat_train = flatten_data(train)
    started = time.time()
    mlp_history = nn.train(mlp, flat_train, epochs=args.epochs_mlp, lr=args.lr,
                           batch_size=16, verbose_every=0)
    mlp_time = time.time() - started

    scores = {
        "cnn": [nn.accuracy(cnn, data) for data in tests],
        "mlp": [nn.accuracy(mlp, flatten_data(data)) for data in tests],
    }

    print(f"\n{'':16}{'параметры':>10}   {'центр':>6}   "
          f"{'сдвиг 1':>7}   {'сдвиг 2':>7}")
    for name, net, values in (("свёрточная", cnn, scores["cnn"]),
                              ("MLP", mlp, scores["mlp"])):
        print(f"{name:16}{net.parameters():10d}   "
              f"{values[0]:6.1%}   {values[1]:7.1%}   {values[2]:7.1%}")

    print(f"время: CNN {cnn_time:.1f} с, MLP {mlp_time:.1f} с")
    print(f"потеря: CNN {cnn_history[-1]:.4f}, MLP {mlp_history[-1]:.4f}")

    # для картинки берём плюс по центру и плюс, на котором MLP ошиблась
    centered = next(image for image, label in tests[0] if label == 0)
    shifted = next((image for image, label in tests[2]
                    if label == 0 and mlp.predict(datasets.flatten(image)) != 0),
                   None)
    if shifted is None:
        shifted = next(image for image, label in tests[2] if label == 0)

    width, height, size = render_conv(cnn, mlp, centered, shifted, scores, args.out)
    print(f"\nкартинка: {args.out} ({width}×{height}, {size / 1024:.0f} КБ)")

    if scores["cnn"][2] <= scores["mlp"][2]:
        print("\nвнимание: сдвиг не дал разницы — проверь гиперпараметры")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

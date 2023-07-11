import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pandas as pd


class EllipseGen:
    def __init__(self, a: float, b: float, x0: float, y0: float, angle: float):
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.angle = angle

    def get_points(self, samples=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.linspace(0, 2 * pi, 100)
        x = self.x0 + self.a * np.cos(t) * np.cos(self.angle) - self.b * np.sin(t) * np.sin(self.angle)
        y = self.y0 + self.a * np.cos(t) * np.sin(self.angle) + self.b * np.sin(t) * np.cos(self.angle)
        samples = np.array(random.sample(list(zip(x, y)), samples))
        return samples[:, 0], samples[:, 1], np.ones_like(samples[:, 1])

    # def generate_ellipse_points():


def add_noize(input_points: Tuple[np.ndarray, np.ndarray, np.ndarray], samples=10, eps=0.2):
    x_input, y_input, signal_input = input_points

    x_range = min(x_input) - eps, max(x_input) + eps
    y_range = min(y_input) - eps, max(y_input) + eps
    x = np.random.uniform(x_range[0], x_range[1], samples)
    y = np.random.uniform(y_range[0], y_range[1], samples)
    signal = np.zeros_like(x)

    result_x = np.append(x_input, x)
    result_y = np.append(y_input, y)
    result_s = np.append(signal_input, signal)
    return result_x, result_y, result_s


if __name__ == "__main__":
    tests = []
    for i in range(100):
        a = random.uniform(0.5, 3)
        b = random.uniform(0.5, 3)
        x0 = random.uniform(-3, 3)
        y0 = random.uniform(-3, 3)
        angel = random.uniform(0, 2 * pi)
        tests.append([a, b, x0, y0, angel])

    for i, test in enumerate(tests):
        el = EllipseGen(*test)
        points = el.get_points(samples=10)
        x, y, s = add_noize(points, samples=5)
        df = pd.DataFrame({
            # f"INDEX": range(len(s)),
            f"x": x,
            f"y": y,
            f"signal": s.astype(int)
        })
        df.to_csv(f'dataset/examples/ellipse_{i}.csv', index=False)
        plt.scatter(x, y)
        plt.savefig(f'dataset/examples/sample_{i}.png')
        plt.close()

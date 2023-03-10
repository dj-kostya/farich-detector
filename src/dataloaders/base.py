from abc import abstractmethod, ABC
from typing import Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use


class IDataloader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[pd.DataFrame]:
        pass

    @staticmethod
    def plot_solution(df: pd.DataFrame):
        if 'signal' in df:
            sh = df['signal'].to_numpy()
        else:
            sh = np.ones(len(df['t_c']), bool)
        plt.figure(figsize=(8, 5), dpi=120)
        ax = plt.axes(projection='3d')
        ax.scatter3D(df['t_c'][sh], df['x_c'][sh], df['y_c'][sh], 'r.', label='Signal')
        if not sh.all():
            ax.scatter3D(df['t_c'][~sh], df['x_c'][~sh], df['y_c'][~sh], 'k.', label='Noise')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.set_title('data example')
        plt.legend()
        return ax

    @staticmethod
    def plot_xy_solution(df: pd.DataFrame):
        if 'signal' in df:
            sh = df['signal'].to_numpy()
        else:
            sh = np.ones(len(df['t_c']), bool)
        plt.figure(figsize=(8, 5), dpi=120)
        ax = plt.axes()
        ax.plot(df['x_c'][sh], df['y_c'][sh], 'r.', label='Signal')
        if not sh.all():
            ax.plot(df['x_c'][~sh], df['y_c'][~sh], 'k.', label='Noise')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('data example')
        plt.legend()
        return ax

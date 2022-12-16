import pandas as pd
from abc import ABC, abstractmethod


class ISolution(ABC):
    @abstractmethod
    def run(self, df: pd.DataFrame):
        pass

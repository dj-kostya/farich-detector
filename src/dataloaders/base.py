from abc import abstractmethod, ABC
from typing import Iterator

import pandas as pd


class BaseDataloader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[pd.DataFrame]:
        pass

import numpy as np
import torch
from torch.utils.data import Dataset

from psycopg2 import connect
from psycopg2.extras import NamedTupleCursor


class MatrixDataset(Dataset):

    def __init__(self, connection_str: str, table_name: str) -> None:
        super().__init__()
        self.conn = connect(connection_str)
        self.table_name = table_name
        self.cache = {}

    def __len__(self):
        with self.conn.cursor(cursor_factory=NamedTupleCursor) as curs:
            curs.execute(
                f"select count(row_number) as cnt from {self.table_name}")
            results = curs.fetchone()
            return results.cnt

    def __getitem__(self, idx):
        with self.conn.cursor(cursor_factory=NamedTupleCursor) as curs:
            curs.execute(
                f"select x_c, y_c, t_c, signal from {self.table_name} where row_number={idx+1};")
            results = curs.fetchone()
            sig_indexes = torch.tensor([
                results.x_c,
                results.y_c,
                results.t_c,
            ])
            val = torch.from_numpy(np.ones_like(results.t_c))
            result = torch.sparse_coo_tensor(sig_indexes, val, (241, 241, 501))
            return result

    def __getitems__(self, indxes):
        indxes = frozenset(indxes)
        if indxes in self.cache:
            return self.cache[indxes]
        results = []
        with self.conn.cursor(cursor_factory=NamedTupleCursor) as curs:
            curs.execute(
                f"select x_c, y_c, t_c, signal from {self.table_name} where (row_number - 1) in %(indexes)s;", {
                    "indexes": tuple(indxes)
                })
            for el in curs:
                sig_indexes = torch.tensor([
                    [0] * len(el.x_c),
                    el.x_c,
                    el.y_c,
                    el.t_c,
                ])
                val = torch.from_numpy(np.ones_like(el.t_c)).type(torch.float16)
                matrix = torch.sparse_coo_tensor(
                    sig_indexes, val, (1, 240, 240, 50))
                signal_val = torch.tensor(el.signal, dtype=torch.int8)
                signal = torch.sparse_coo_tensor(
                    sig_indexes, signal_val, (1, 240, 240, 50))
                results.append((matrix, signal))
        self.cache[indxes]=results
        return results

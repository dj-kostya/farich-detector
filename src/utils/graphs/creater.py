import pandas as pd
import numpy as np
from torch_geometric.data import Data




def split_df_to_graphs(df, point_radius=35, window=5, stride=2):
    graphs = []
    total_signal = df.signal.sum()
    if total_signal == 0:
        return []
    max_t = df.t_c.max()
    for from_t, to_t in get_indexes(0, max_t, window=window, stride=stride):
        entry_df = df[(df.t_c >= from_t) & (
            df.t_c <= to_t)].reset_index(drop=True)
        entry_df['join'] = 0
        entry_df['index'] = entry_df.index
        current_signal = entry_df.signal.sum()
        joined_df = entry_df.merge(entry_df, on='join', how='outer')
        joined_df['distance'] = joined_df.apply(lambda x: np.sqrt(
            (x.x_c_x - x.x_c_y)**2 + (x.y_c_x - x.y_c_y)**2 + (x.t_c_x - x.t_c_y)**2), axis=1)

        joined_df_with_distance = joined_df[(joined_df.distance > 0) & (
            joined_df.distance <= point_radius)]

        edjes = pd.DataFrame(
            [joined_df_with_distance.index_x, joined_df_with_distance.index_y]).to_numpy().T
        nodes = entry_df[['x_c', 'y_c', 't_c']].to_numpy()
        graph = Data(x=nodes, edge_index=edjes, y=current_signal / total_signal)
        graphs.append(graph)
    return graphs


def get_indexes(a: int, b: int, window: int, stride: int)->list[tuple[int, int]]:
    res = []
    from_idx = a
    while from_idx <= b:
        res.append((from_idx, from_idx + window))
        from_idx += stride
    return res


if __name__ == "__main__":
    for i in get_indexes(0, 10, 5, 2):
        print(i)

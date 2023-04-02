

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from multiprocessing import Pool
from src.utils.graphs import split_df_to_graphs
from tqdm import tqdm
from pathlib import Path





if __name__ == "__main__":
    dataset_dir = Path('dataset')
    root_file = "farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root"
    root_path = dataset_dir / root_file
    graphs_dir = dataset_dir/"graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    dataset = "dataset_100000_with_noise_2e5.csv"

    df = pd.read_csv(dataset_dir / dataset)

    # for entry_idx in tqdm(df.entry.unique()):
    #     entry_df = df[df.entry == entry_idx]

    indexes = df.entry.unique()
    # items = [(df[df.entry == entry_idx], entry_idx)
    #         for entry_idx in indexes]
    print("Df loaded")
    def func(entry_idx):
        df_entry=df[df.entry == entry_idx]
        graphs = split_df_to_graphs(df_entry)

        for graph_idx, graph in enumerate(graphs):
            filename = graphs_dir/f"{entry_idx}_{graph_idx}_graph.pt"
            torch.save(graph, filename)
    with Pool(13) as p:
        list(tqdm(p.imap(func, indexes), total=len(indexes)))

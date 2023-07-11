from collections import defaultdict
from pathlib import Path
# from reader import show_uproot_tree, readInfoFromRoot, genChunkFromRoot
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from src.dataloaders import RootDataLoader, PointOnlyDataloader, NoiseDataLoader
from src.solutions import EllipseDataFitting

dataset_dir = Path('dataset')
root_file = "farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root"
root_path = dataset_dir / root_file

if __name__ == '__main__':
    import time

    total_acc = {}
    noise_freq_per_sqmm=2e4

    dl = NoiseDataLoader(root_path, verbose=False, noise_freq_per_sqmm=noise_freq_per_sqmm, only_hits=True)
    for df in dl.genChunkFromRoot(event_chunk_size=100):
        df['signal'] = df['signal'].astype(int)
        df.to_csv(dataset_dir / f'dataset_100_with_noise_{int(noise_freq_per_sqmm)}.csv')

        break
    # alg = EllipseDataFitting(save_graphics=True, eps_proj=1e-2, eps_ellipse=1e-4, minimal_points_in_plain=5)
    # for idx, df in enumerate(dl):
    #     if idx < 30:
    #         start = time.time()
    #         elipses = alg.run(df)
    #         top = max(elipses, key=lambda x: len(x.points))
    #         res = defaultdict(bool)
    #         for p in top.points:
    #             res[(p.x, p.y, p.t)] = True
    #         df['preds'] = df.apply(lambda x: res[(x.x_c, x.y_c, x.t_c)], axis=1)
    #         total_acc[idx] = (df[df.preds]['x_c'].count() / df[df.signal]['x_c'].count(), time.time() - start)
    #         assert len(elipses)

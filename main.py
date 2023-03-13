from pathlib import Path
# from reader import show_uproot_tree, readInfoFromRoot, genChunkFromRoot
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from tqdm import tqdm

from src.dataloaders import RootDataLoader, PointOnlyDataloader, NoiseDataLoader
from src.solutions import EllipseDataFitting

dataset_dir = Path('dataset')
root_file = "farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root"
root_path = dataset_dir / root_file

if __name__ == '__main__':
    dl = NoiseDataLoader(root_path, verbose=False, only_hits=True, noise_freq_per_sqmm=2e4)
    eps = 0.4
    alg = EllipseDataFitting(save_graphics=False, eps=eps, use_tqdm=True)
    for idx, df in tqdm(enumerate(dl)):
        if idx != 0:
            break
        center, points = alg.run(df)
        # print(len(points))
        ax = dl.plot_solution(df)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        t = [p[2] for p in points]
        ax.scatter3D(t, x, y, 'g', s=35, marker='^', label='Predict')
        plt.legend()
        plt.savefig(f'tmp/example_{idx}_{eps}.png')
        if idx > 10:
            break
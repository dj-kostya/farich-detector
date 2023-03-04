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
    dl = PointOnlyDataloader(root_path, verbose=False)
    alg = EllipseDataFitting(save_graphics=True)
    for idx, df in enumerate(dl):
        if idx == 5:
            centers, vectors = alg.run(df)
            ax = dl.plot_solution(df)
            ax.scatter3D(centers, centers, centers, 'g.', label='Center')
            plt.legend()
            plt.savefig(f'tmp/example_{idx}.png')
            break
            if idx > 10:
                break


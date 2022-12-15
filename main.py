from pathlib import Path
# from reader import show_uproot_tree, readInfoFromRoot, genChunkFromRoot
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from src.dataloaders import RootDataLoader, PointOnlyDataloader, NoiseDataLoader

dataset_dir = Path('dataset')
root_file = "farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root"
root_path = dataset_dir / root_file

if __name__ == '__main__':
    dl = NoiseDataLoader(root_path, verbose=False, noise_freq_per_sqmm=10e4, only_hits=True, noise_time_range=(0, 2))
    df = next(dl.__iter__())
    # print(df)
    sh = df['signal'].to_numpy()
    mpl_use('MacOSX')
    plt.figure(figsize=(8, 5), dpi=120)
    ax = plt.axes(projection='3d')
    ax.scatter3D(df['t_c'][sh], df['x_c'][sh], df['y_c'][sh], 'r.', label='Signal')
    ax.scatter3D(df['t_c'][~sh], df['x_c'][~sh], df['y_c'][~sh], 'k.', label='Noise')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title('data example')
    plt.legend()
    plt.savefig('example.png')

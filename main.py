from pathlib import Path
import pickle
from typing import Union
# from reader import show_uproot_tree, readInfoFromRoot, genChunkFromRoot
import matplotlib.pyplot as plt
# from matplotlib import use as mpl_use
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.expirements.data_filters import filter_empty_values
from src.expirements.models import Unet3D
from src.expirements.datasets import GraphFileDataset, MatrixDataset
from src.expirements.modules import BaseGNNModule, LightningGCN, Unet3DModule
from src.expirements.data_transforms import fix_datatypes
from src.expirements.loss import DiceLoss


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

dataset_dir = Path('dataset')
root_file = "farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root"
root_path = dataset_dir / root_file
graphs_dir = dataset_dir / 'graph_prew'
cache_dir = dataset_dir / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)
edge_feature_dims = [
    8,
    16,
    32,
    64,
    128,
    128,
    64,
    32
]

num_filters = [
    16,
    32,
    64,
    128,
    128,
    64,
    32,
    16
]


def to_pickle(obj, filename: Union[str, Path]):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

CONNECTION_STR="postgresql://farich:KfSZ5HQu3xH62b3cp5Cg652Hys6JpriwNjM48VK9N@192.168.1.133:32432/farich"

if __name__ == '__main__':
    train_dataset = MatrixDataset(CONNECTION_STR, "farich_sparse_matrix_train")
    val_dataset = MatrixDataset(CONNECTION_STR, "farich_sparse_matrix_val")
    test_dataset = MatrixDataset(CONNECTION_STR, "farich_sparse_matrix_test")

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)
    



    # train_dataset = GraphFileDataset(
    #     graphs_dir / 'train', pre_filter=filter_empty_values, pre_transform=fix_datatypes, use_tqdm=True)
    # val_dataset = GraphFileDataset(
    #     graphs_dir / 'val', pre_filter=filter_empty_values, pre_transform=fix_datatypes)
    # test_dataset = GraphFileDataset(
    #     graphs_dir / 'test', pre_filter=filter_empty_values, pre_transform=fix_datatypes)

    # batch_size = 100
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=12)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=12)
    # # test_loader = DataLoader(test_dataset)

    model = Unet3D(1, 1, model_depth=2)
    model.half()
    module = Unet3DModule(model=model, loss=DiceLoss(), batch_size=batch_size)
    logger = WandbLogger(project='FARICH', log_model='all', group="UNET3D_2", tags=['Unet3D'])
    trainer = pl.Trainer(max_epochs=25, logger=logger, callbacks=[
                         ModelCheckpoint(save_weights_only=True, mode="max", monitor="loss/val")])
    trainer.fit(model=module, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # model = PlainGCN(num_features=3, num_classes=1)
    # # print(model)
    # logger = WandbLogger(project='FARICH', log_model='all')
    # baseGNNModule = LightningGCN(model=model, batch_size=batch_size)
    # trainer = pl.Trainer(max_epochs=250, logger=logger, callbacks=[
    #                      ModelCheckpoint(save_weights_only=True, mode="max", monitor="loss/val")])
    # trainer.fit(model=baseGNNModule, train_dataloaders=train_loader,
    #             val_dataloaders=val_loader)

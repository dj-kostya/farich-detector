import os
import shutil
from tqdm import tqdm
from collections.abc import Sequence
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_folder_train_test_val(folder: Path, train_test_ratio: float = 0.6, test_val_ratio: float = 0.5):
    raw_path = folder / 'raw'
    test_folder = folder / 'test' / 'raw'
    train_folder = folder / 'train' / 'raw'
    val_folder = folder / 'val' / 'raw'
    files = [raw_path / f for f in os.listdir(raw_path) if os.path.isfile(raw_path / f)]
    train_files, test_val_files = train_test_split(
        files, train_size=train_test_ratio, random_state=42)
    test_files, val_files = train_test_split(test_val_files, train_size=test_val_ratio, random_state=42)
    _move_files_to_folder(train_files, train_folder)
    _move_files_to_folder(test_files, test_folder)
    _move_files_to_folder(val_files, val_folder)

def _move_files_to_folder(files: Sequence[Path], folder: Path)->None:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Process {folder} folder")
    for f in tqdm(files):
        shutil.copy(f, folder/(f.name))
    print("Done")

        


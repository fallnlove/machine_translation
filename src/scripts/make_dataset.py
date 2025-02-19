import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def main(config):
    path = Path(config["path"]).resolve().absolute()
    labels = pd.read_csv(str(path / "labels.csv"))

    train_path = path / "train"
    if not train_path.exists():
        train_path.mkdir()
    val_path = path / "val"
    if not val_path.exists():
        val_path.mkdir()

    for category in range(200):
        label = list(labels[labels["Category"] == category]["Id"])
        random.shuffle(label)
        for i, name in enumerate(label):
            if i < 50:
                shutil.move(str(path / "trainval" / name), str(val_path / name))
            else:
                shutil.move(str(path / "trainval" / name), str(train_path / name))


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--path",
        default=None,
        type=str,
        help="path to dataset",
    )
    config = parser.parse_args()
    main(vars(config))

import argparse

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CustomDataset
from src.dataset.collate import collate_fn
from src.model import ResNet
from src.trainer import Inferencer


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(type="resnet50", num_classes=200)
    model.load_state_dict(torch.load(config["modelpath"])["state_dict"])
    model = model.to(device)

    dataset = CustomDataset(config["path"], "test")

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )
    test_augs = transforms.Compose(
        [
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )

    trainer = Inferencer(
        model=model,
        device=device,
        dataloaders={
            "test": dataloader,
        },
        transforms={
            "test": transform,
        },
        test_augmentations=test_augs,
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--path",
        default="./bhw1",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "-modelpath",
        "--modelpath",
        default="./models/best_model.pth",
        type=str,
        help="path to pretrained model",
    )
    config = parser.parse_args()
    main(vars(config))

import argparse

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from src.dataset import CustomDataset
from src.dataset.collate import collate_fn
from src.model import TranslateTransformer
from src.trainer import Inferencer


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_train = CustomDataset("train", config["path"])
    dataset_val = CustomDataset("val", config["path"])
    dataset_test = CustomDataset("test", config["path"])

    vocabs = {}
    for language in dataset_train.get_languages():
        vocabs[language] = build_vocab_from_iterator(
            [dataset_train.get_texts()[language] + dataset_val.get_texts()[language]],
            min_freq=config["minfreq"],
            specials=["<pad>", "<unk>", "<bos>", "<eos>"],
            special_first=True,
        )

        vocabs[language].set_default_index(dataset_train.UNK)
    dataset_train.set_vocab(vocabs)
    dataset_val.set_vocab(vocabs)
    dataset_test.set_vocab(vocabs)

    model = TranslateTransformer(n_vocab_source=len(vocabs["de"]), n_vocab_dest=len(vocabs["en"]), pad_idx=dataset_train.PAD)
    model.load_state_dict(torch.load(config["modelpath"])["state_dict"])
    model = model.to(device)

    dataloader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    inferencer = Inferencer(
        model=model,
        device=device,
        dataloaders={
            "test": dataloader,
        },
        datasets={
            "test": dataset_test,
        },
    )

    inferencer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--path",
        default="./data",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "-modelpath",
        "--modelpath",
        default="./models/model_30.pth",
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "-minfreq",
        "--minfreq",
        default=5,
        type=int,
        help="Minimum frequency",
    )
    config = parser.parse_args()
    main(vars(config))

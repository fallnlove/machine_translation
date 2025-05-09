import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from src.dataset import CustomDataset
from src.dataset.collate import collate_fn
from src.loss import CrossEntropyLossWrapper
from src.metrics import Bleu
from src.model import TranslateTransformer
from src.scheduler import WarmupLR
from src.trainer import Trainer
from src.utils import set_random_seed
from src.writer import EmptyWriter, WanDBWriter


def main(config):
    set_random_seed(112)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "num_epochs": config["epochs"],
        "warmup_epochs": config["warmupepochs"],
        "lr": config["lr"],
        "batch_size": config["batchsize"],
        "model": "Transformer",
        "optimizer": "AdamW",
        "scheduler": "Const",
        "min_freq": config["minfreq"],
        "weight_sharing": config["weightsharing"],
        "max_tokens": config["maxtokens"],
        "dropout": config["dropout"],
    }

    dataset_train = CustomDataset("train", config["path"])
    dataset_val = CustomDataset("val", config["path"])

    vocabs = {}
    for language in dataset_train.get_languages():
        vocabs[language] = build_vocab_from_iterator(
            [dataset_train.get_texts()[language] + dataset_val.get_texts()[language]],
            min_freq=configs["min_freq"],
            specials=["<pad>", "<unk>", "<bos>", "<eos>"],
            special_first=True,
            max_tokens=configs["max_tokens"],
        )

        vocabs[language].set_default_index(dataset_train.UNK)
    dataset_train.set_vocab(vocabs)
    dataset_val.set_vocab(vocabs)

    model = TranslateTransformer(n_vocab_source=len(vocabs["de"]), n_vocab_dest=len(vocabs["en"]), pad_idx=dataset_train.PAD, weight_sharing=configs["weight_sharing"], dropout=configs["dropout"])
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params, len(vocabs["de"]), len(vocabs["en"]))

    train_loader = DataLoader(
        dataset_train,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    writer = WanDBWriter(project_name="DL-BHW-2", config=configs) if config["wandb"] else EmptyWriter()

    loss_fn = CrossEntropyLossWrapper(ignore_index=dataset_train.PAD).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)
    metrics = [Bleu()]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000000
    )

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        writer=writer,
        dataloaders={
            "train": train_loader,
            "eval": val_loader,
        },
        datasets={
            "train": dataset_train,
            "eval": dataset_val,
        },
        num_epochs=configs["num_epochs"],
        transforms=None,
    )
    trainer.run()


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
        "-wandb",
        "--wandb",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="log info to wandb",
    )
    parser.add_argument(
        "-epochs",
        "--epochs",
        default=71,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-warmupepochs",
        "--warmupepochs",
        default=1,
        type=int,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "-lr",
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate for training",
    )
    parser.add_argument(
        "-batchsize",
        "--batchsize",
        default=128,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-minfreq",
        "--minfreq",
        default=5,
        type=int,
        help="Minimum frequency",
    )
    parser.add_argument(
        "-weightsharing",
        "--weightsharing",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Use weight sharing in transformer",
    )
    parser.add_argument(
        "-maxtokens",
        "--maxtokens",
        default=None,
        type=int,
        help="Minimum frequency",
    )
    parser.add_argument(
        "-dropout",
        "--dropout",
        default=0.3,
        type=float,
        help="Dropout rate",
    )
    config = parser.parse_args()
    main(vars(config))

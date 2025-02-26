from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.metrics.tracker import MetricTracker


class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        dataloaders: dict,
        datasets: dict,
    ):
        self.model = model

        self.device = device
        self.dataloaders = dataloaders
        self.datasets = datasets

        self.save_path = Path("./submissions/")
        if not self.save_path.exists():
            self.save_path.mkdir()

    def run(self):
        """
        Run training process.
        """

        self.predictions = []
        try:
            self._eval_process()
        except KeyboardInterrupt as e:
            print("Keyboard interrupt.")

        with open(str(self.save_path / "test1.de-en.en"), 'w') as f:
            f.write("\n".join(self.predictions) + "\n")

    @torch.no_grad()
    def _eval_process(self):
        """
        Start inference.
        """
        self.model.eval()

        for index, batch in tqdm(
            enumerate(self.dataloaders["test"]), total=len(self.dataloaders["test"])
        ):
            batch = self._process_batch(batch)

    def _process_batch(self, batch):
        batch = self._move_to_device(batch)

        for i in range(batch["source"].shape[0]):
            output = self.model.translate(batch["source"][i], batch["length"][i], batch["length"][i] * 2 + 10)
            sentence = self.datasets["test"].dest_tokens2text(output.squeeze().cpu().numpy().tolist())
            self.predictions.append(sentence)

        return batch

    def _move_to_device(self, batch: dict):
        """
        Move batch to device. Batch should have a key "images" which is the input data.

        Input:
            batch (dict): batch of data
        Output:
            batch (dict): batch of data
        """
        batch["source"] = batch["source"].to(self.device)

        return batch

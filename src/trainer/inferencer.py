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
        beam_size: int,
        dataloaders: dict,
        datasets: dict,
    ):
        self.model = model

        self.device = device
        self.beam_size = beam_size
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
            unk = []
            for j in range(batch["source"].shape[1]):
                if batch["source"][i, j] == self.datasets["test"].UNK:
                    unk.append(self.datasets["test"].dest_tokens2text([batch["source"][i, j].item()]))
            output = self.model.translate(batch["source"][i], batch["length"][i], batch["length"][i] * 2 + 10, beam_size=self.beam_size)
            sentence = self.datasets["test"].dest_tokens2text(output.squeeze().cpu().numpy().tolist())
            tokens = sentence.split(' ')

            for j in range(len(tokens)):
                if tokens[j] == '<unk>' and len(unk) > 0:
                    tokens[j] = unk[0]
                    if len(unk) > 1:
                        unk = unk[1:]


            self.predictions.append(" ".join(tokens))

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

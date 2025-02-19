import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWrapper(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "CrossEntropyLoss"

    def forward(self, predictions, labels, **batch) -> Tensor:
        """
        Input:
            predictions (Tensor): predicted probabilities.
            labels (Tensor): labels.
        Output:
            loss (int): CE loss.
        """

        loss = super().forward(
            input=predictions,
            target=labels,
        )

        return {"loss": loss}

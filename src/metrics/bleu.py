from torch import Tensor

from torchtext.data.metrics import bleu_score


class Bleu:
    def __init__(self):
        self.name = "Bleu"
        self.metric_func = bleu_score

    def __call__(self, translated: Tensor, ground_truth: Tensor, **batch):
        """
        Input:
            predictions (Tensor): predicted probabilities.
            labels (Tensor): labels.
        Output:
            output (int): bleu metric.
        """

        return self.metric_func(translated, ground_truth)

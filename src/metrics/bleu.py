import subprocess
import os
from torch import Tensor


class Bleu:
    def __init__(self):
        self.name = "Bleu"

    def __call__(self, translated: Tensor, ground_truth: Tensor, **batch):
        """
        Input:
            predictions (Tensor): predicted probabilities.
            labels (Tensor): labels.
        Output:
            output (int): bleu metric.
        """

        with open('translated.en', 'w') as f:
            f.write("\n".join(translated) + "\n")
        with open('SECRET_FILE.en', 'w') as f:
            f.write("\n".join(ground_truth) + "\n")

        command = "cat translated.en | sacrebleu SECRET_FILE.en --tokenize none --width 2 -b"

        result = subprocess.run(command, shell=True, text=True, capture_output=True)

        os.remove('translated.en')
        os.remove('SECRET_FILE.en')

        return float(result.stdout)

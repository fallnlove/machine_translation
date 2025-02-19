from collections import defaultdict
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dataset.dataset import CustomDataset


def collate_fn(dataset_items: list[dict]) -> dict[Union[torch.Tensor, list]]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, list]]): dict, containing batch-version
            of the tensors.
    """

    result_batch = defaultdict()

    result_batch["source"] = pad_sequence(
        [item["source"] for item in dataset_items],
        batch_first=True,
        padding_value=CustomDataset.PAD
    )
    result_batch["length"] = [item["length"] for item in dataset_items]
    result_batch["ground_truth"] = [item["ground_truth"] for item in dataset_items]

    if "dest" in dataset_items[0].keys():
        result_batch["dest"] = pad_sequence(
            [item["dest"] for item in dataset_items],
            batch_first=True,
            padding_value=CustomDataset.PAD
        )

    return result_batch

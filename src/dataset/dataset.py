import random
import shutil
import os
from pathlib import Path

import gdown

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from tqdm import tqdm


class CustomDataset(Dataset):
    URL = "https://drive.google.com/uc?id=1hvU16vYvncpg4OSeveDxWbKSOgqrcVU4"
    ACHIEVE_PATH = "bhw2-data.zip"

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(
        self,
        part: str,
        path: str = "data/",
        shuffle_index: bool = False,
    ):
        """
        Input:
            path (str): path to dataset.
            part (str): part of dataset (train, val or test).
            min_freq (int): min frequency of word.
            shuffle_index (bool): shuffle dataset.
            instance_transforms (Callable): augmentations.
        """
        assert part in ["train", "val", "test"]

        self.dataset_path = Path(path).absolute().resolve()
        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True)
            self._download_dataset()

        self.part = part
        self.shuffle_index = shuffle_index

        self.languages = ["de"]
        if self.part != "test":
            self.languages = ["de", "en"]

        self.texts: dict = self._get_texts()

        self.tokenizers = {}
        for language in self.languages:
            self.tokenizers[language] = get_tokenizer(None, language)

        if shuffle_index:
            random.seed(42)
            for language in self.languages:
                random.shuffle(self.texts[language])

    def __len__(self):
        return len(self.texts[self.languages[0]])

    def __getitem__(self, index):
        data = {}
        for language, part in zip(self.languages, ["source", "dest"]):
            tokens = self.tokenizers[language](self.texts[language][index])
            data[part] = torch.LongTensor([self.BOS] + self.vocabs[language](tokens) + [self.EOS])
        data["length"] = len(data["source"])
        data["ground_truth"] = self.texts[language][index]

        return data
    
    def source_tokens2text(self, tokens):
        return " ".join(self.vocabs[self.languages[0]].lookup_tokens(tokens)).replace("<eos>", "").replace("<bos>", "").strip()
    
    def dest_tokens2text(self, tokens):
        return " ".join(self.vocabs[self.languages[1]].lookup_tokens(tokens)).strip()
    
    def set_vocab(self, vocab):
        self.vocabs = vocab

    def get_languages(self):
        return self.languages

    def get_texts(self):
        output = {}

        for language in self.languages:
            output[language] = []
            for text in self.texts[language]:
                output[language].extend(self.tokenizers[language](text))
        return output

    def _get_texts(self):
        if self.part == "test":
            paths = [self.dataset_path / f"{self.part}1.de-en.de"]
        else:
            paths = [self.dataset_path / f"{self.part}.de-en.de", self.dataset_path / f"{self.part}.de-en.en"]

        texts = {}
        for language, path in zip(self.languages, paths):
            with open(str(path), 'r') as f:
                text = list(map(lambda x: x.rstrip(), f.readlines()))
            texts[language] = text

        return texts
    
    def _download_dataset(self):
        gdown.download(self.URL, self.ACHIEVE_PATH)

        shutil.unpack_archive(self.ACHIEVE_PATH, self.dataset_path)

        for fpath in (self.dataset_path / "data").iterdir():
            shutil.move(str(fpath), str(self.dataset_path / fpath.name))
        
        os.remove(self.ACHIEVE_PATH)
        shutil.rmtree(str(self.dataset_path / "data"))

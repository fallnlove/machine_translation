# Image Classification

## Training model

To split the dataset into train and validation parts, run the command:

```bash
python src/scripts/make_dataset.py --path PATH_TO_DATASET
```

To start the training and inference process, run the command:

```bash
python main.py
```

You can specify path to dataset using the `--path` option.

## Inference pre-trained model

To inference the pre-trained model, download the weights with the command:

```bash
pip install gdown
python src/scripts/download_model.py
```

Start the inference process via the command:

```bash
python inference.py
```

You can specify the path to the dataset using the `--path` option. Predictions will
be saved in the `submissions` directory.

import argparse
import os


def main(config):
    os.system("pip install -r ./requirements.txt")

    os.system("python train.py --path " + config["path"])

    os.system("python inference.py --path " + config["path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--path",
        default="./bhw1",
        type=str,
        help="path to dataset",
    )
    config = parser.parse_args()
    main(vars(config))

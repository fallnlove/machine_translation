from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1CwpLhdwnzQkSGcqB5-VEjeC_xIM_lvW-": "models/best_model.pth",
}


def main():
    path_gzip = Path("models/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)


if __name__ == "__main__":
    main()

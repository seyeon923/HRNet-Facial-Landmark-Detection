import os
import logging
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--dataset-info", "-d", required=True)
    parser.add_argument("--image-root", "-i", required=True)
    parser.add_argument("--out-dir", "-o", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    preds_path = args.predictions
    dataset_info_path = args.dataset_info
    img_root = args.image_root
    out_dir = args.out_dir

    dataset_info = pd.read_csv(dataset_info_path)
    preds = torch.load(preds_path, weights_only=True)

    num_imgs = len(dataset_info)
    assert num_imgs == len(preds)

    for img_name, pred in tqdm(zip(dataset_info["image_name"], preds), total=num_imgs):
        fig = None
        try:
            img_path = os.path.join(img_root, img_name)
            img = np.array(Image.open(img_path))

            fig, ax = plt.subplots()

            ax.axis("off")
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.scatter(pred[:, 0], pred[:, 1], c="green", marker=".")

            save_path = os.path.join(
                out_dir, f"{img_name}-prediction.png")
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            assert os.path.isdir(save_dir)

            fig.savefig(save_path, bbox_inches="tight")
        except Exception as ex:
            logging.error(
                F"Failed to visualize predictions of image '{img_path}' - {ex}")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()

import os
import logging
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Rectangle


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

    for i in tqdm(range(num_imgs)):
        fig = None

        img_name = dataset_info.loc[i, "image_name"]
        scale = dataset_info.loc[i, "scale"]
        cx = dataset_info.loc[i, "center_w"]
        cy = dataset_info.loc[i, "center_h"]

        try:
            img_path = os.path.join(img_root, img_name)
            img = np.array(Image.open(img_path))

            pred = preds[i, :]

            fig, ax = plt.subplots()

            ax.axis("off")
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.scatter(pred[:, 0], pred[:, 1], c="green", marker=".", s=10)

            w = 200 * scale
            h = 200 * scale
            ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h,
                         edgecolor="red", facecolor="none"))

            ax.scatter(cx, cy, c="red", marker="*")

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
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    main()

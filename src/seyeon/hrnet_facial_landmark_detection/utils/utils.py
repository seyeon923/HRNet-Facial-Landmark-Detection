# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Kim Se-yeon(tpdussla93@gmail.com)
# ------------------------------------------------------------------------------

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print(f"Creating root output directory '{root_output_dir}'")
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.splitext(os.path.basename(cfg_name))[0]

    final_output_dir: Path = root_output_dir / dataset / cfg_name

    print(f"Creating final output directory '{final_output_dir}'")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] %(message)s"
    logging.basicConfig(format=fmt)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    time_str = time.strftime("%Y%m%d%H%M%S")
    log_file = f"{cfg_name}_{time_str}_{phase}.log"
    final_log_file = final_output_dir / log_file
    file_handler = logging.FileHandler(final_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(file_handler)

    tensorboard_log_dir: Path = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    logger.info(f"Creating tensorboard log directory '{tensorboard_log_dir}'")
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    ckpt_path = os.path.join(output_dir, filename)
    torch.save(states, ckpt_path)
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.abspath(ckpt_path), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'], os.path.join(
            output_dir, 'model_best.pth'))

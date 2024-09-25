# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Kim Se-yeon(tpdussla93@gmail.com)
# ------------------------------------------------------------------------------

import time
import logging

import torch
import numpy as np

from tqdm import tqdm

from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, critertion, optimizer, epoch, writer_dict):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    with tqdm(train_loader) as pbar:
        for inp, target, meta in pbar:
            # measure data time
            data_time.update(time.time()-end)

            # compute the output
            output = model(inp)
            target = target.cuda(non_blocking=True)

            loss = critertion(output, target)

            # NME
            score_map = output.data.cpu()
            preds = decode_preds(
                score_map, meta['center'], meta['scale'], target.shape[2:])

            nme_batch = compute_nme(preds, meta)
            nme_batch_sum += np.sum(nme_batch)
            nme_count += preds.size(0)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inp.size(0))

            pbar.set_postfix({
                "Epoch": epoch,
                "Data": f"{data_time.val:.4f}s ({data_time.avg:.4f}s)",
                "Loss": f"{losses.val:4f} ({losses.avg:.4f})",
                "NME": f"{np.mean(nme_batch):.4f} ({nme_batch_sum / nme_count:.4f})"
            })

            batch_time.update(time.time() - end)
            end = time.time()

    if writer_dict:
        writer = writer_dict['writer']
        writer.add_scalar('train_loss', losses.val, epoch)

    nme = nme_batch_sum / nme_count
    logger.info(
        f"Train Epoch {epoch}, batch time: {batch_time.avg:.4f}, data time: {data_time.avg:.4f}, loss: {losses.avg:.4f}, nme: {nme:.4f}")


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for inp, target, meta in tqdm(val_loader, total=len(val_loader), desc="Validate"):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(
                score_map, meta['center'], meta['scale'], target.shape[2:])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for inp, target, meta in tqdm(data_loader, desc="Inference", total=len(data_loader)):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(
                score_map, meta['center'], meta['scale'], target.shape[2:])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions

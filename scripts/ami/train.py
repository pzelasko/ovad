#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
#                2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import logging
import math
import numpy as np
import os
import sys
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.dataset import SingleCutSampler
from lhotse.utils import fix_random_seed
from snowfall.common import describe
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.models import AcousticModel

from ovad.dataset import K2VadDataset
from ovad.models import TdnnLstm1a


def get_objf(
    batch: Dict,
    model: AcousticModel,
    device: torch.device,
    training: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
):
    feature = batch["inputs"]  # (N, T, C)
    supervisions = batch["supervisions"]["is_voice"].unsqueeze(-1).long()  # (N, T, 1)

    feature = feature.to(device)
    supervisions = supervisions.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    if training:
        nnet_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output = model(feature)

    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

    # Compute cross-entropy loss
    xent_loss = torch.nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
    tot_score = xent_loss(
        nnet_output.contiguous().view(-1, 2), supervisions.contiguous().view(-1)
    )

    if training:
        optimizer.zero_grad()
        tot_score.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step(),

    ans = (
        tot_score.detach().cpu().item(),  # total objective function value
        supervisions.numel(),  # number of frames
    )
    return ans


def get_validation_objf(
    dataloader: torch.utils.data.DataLoader,
    model: AcousticModel,
    device: torch.device,
):
    total_objf = 0.0
    total_frames = 0.0  # for display only

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        objf, frames = get_objf(batch, model, device, False)
        total_objf += objf
        total_frames += frames

    return total_objf, total_frames


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    model: AcousticModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    tb_writer: SummaryWriter,
    num_epochs: int,
    global_batch_idx_train: int,
):
    total_objf, total_frames = 0.0, 0.0
    valid_average_objf = float("inf")
    time_waiting_for_batch = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()

        curr_batch_objf, curr_batch_frames = get_objf(
            batch, model, device, True, optimizer
        )

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames

        if batch_idx % 10 == 0:
            logging.info(
                "batch {}, epoch {}/{} "
                "global average objf: {:.6f} over {} "
                "frames, current batch average objf: {:.6f} over {} frames "
                "avg time waiting for batch {:.3f}s".format(
                    batch_idx,
                    current_epoch,
                    num_epochs,
                    total_objf / total_frames,
                    total_frames,
                    curr_batch_objf / (curr_batch_frames + 0.001),
                    curr_batch_frames,
                    time_waiting_for_batch / max(1, batch_idx),
                )
            )

            tb_writer.add_scalar(
                "train/global_average_objf",
                total_objf / total_frames,
                global_batch_idx_train,
            )

            tb_writer.add_scalar(
                "train/current_batch_average_objf",
                curr_batch_objf / (curr_batch_frames + 0.001),
                global_batch_idx_train,
            )

        if batch_idx > 0 and batch_idx % 200 == 0:
            (total_valid_objf, total_valid_frames,) = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
            )
            valid_average_objf = total_valid_objf / total_valid_frames
            model.train()
            logging.info(
                "Validation average objf: {:.6f} over {} frames".format(
                    valid_average_objf,
                    total_valid_frames,
                )
            )

            tb_writer.add_scalar(
                "train/global_valid_average_objf",
                valid_average_objf,
                global_batch_idx_train,
            )
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


def main():
    fix_random_seed(42)

    if not torch.cuda.is_available():
        logging.error("No GPU detected!")
        sys.exit(-1)
    device_id = 0
    device = torch.device("cuda", device_id)

    # Reserve the GPU with a dummy variable
    reserve_variable = torch.ones(1).to(device)

    start_epoch = 0
    num_epochs = 100

    exp_dir = "exp-tl1a-adam-xent"
    setup_logger("{}/log/log-train".format(exp_dir))
    tb_writer = SummaryWriter(log_dir=f"{exp_dir}/tensorboard")

    # load dataset
    feature_dir = Path("exp/data")
    logging.info("About to get train cuts")
    cuts_train = CutSet.from_json(feature_dir / "cuts_train.json.gz")
    logging.info("About to get dev cuts")
    cuts_dev = CutSet.from_json(feature_dir / "cuts_dev.json.gz")

    logging.info("About to create train dataset")
    train = K2VadDataset(cuts_train)
    train_sampler = SingleCutSampler(
        cuts_train,
        max_frames=90000,
        shuffle=True,
    )
    logging.info("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(
        train, sampler=train_sampler, batch_size=None, num_workers=4
    )
    logging.info("About to create dev dataset")
    validate = K2VadDataset(cuts_dev)
    valid_sampler = SingleCutSampler(cuts_dev, max_frames=90000)
    logging.info("About to create dev dataloader")
    valid_dl = torch.utils.data.DataLoader(
        validate, sampler=valid_sampler, batch_size=None, num_workers=1
    )

    logging.info("About to create model")
    model = TdnnLstm1a(
        num_features=80,
        num_classes=2,  # speech/silence
        subsampling_factor=1,
    )

    model.to(device)
    describe(model)

    learning_rate = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    best_epoch_info_filename = os.path.join(exp_dir, "best-epoch-info")
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, "epoch-{}.pt".format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer)
        best_objf = ckpt["objf"]
        best_valid_objf = ckpt["valid_objf"]
        global_batch_idx_train = ckpt["global_batch_idx_train"]
        logging.info(
            f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}"
        )

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        curr_learning_rate = learning_rate
        tb_writer.add_scalar("learning_rate", curr_learning_rate, epoch)

        logging.info("epoch {}, learning rate {}".format(epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            device=device,
            optimizer=optimizer,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
        )
        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(
                filename=best_model_path,
                model=model,
                epoch=epoch,
                optimizer=None,
                scheduler=None,
                learning_rate=curr_learning_rate,
                objf=objf,
                valid_objf=valid_objf,
                global_batch_idx_train=global_batch_idx_train,
            )
            save_training_info(
                filename=best_epoch_info_filename,
                model_path=best_model_path,
                current_epoch=epoch,
                learning_rate=curr_learning_rate,
                objf=best_objf,
                best_objf=best_objf,
                valid_objf=valid_objf,
                best_valid_objf=best_valid_objf,
                best_epoch=best_epoch,
            )

        # we always save the model for every epoch
        model_path = os.path.join(exp_dir, "epoch-{}.pt".format(epoch))
        save_checkpoint(
            filename=model_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
            learning_rate=curr_learning_rate,
            objf=objf,
            valid_objf=valid_objf,
            global_batch_idx_train=global_batch_idx_train,
        )
        epoch_info_filename = os.path.join(exp_dir, "epoch-{}-info".format(epoch))
        save_training_info(
            filename=epoch_info_filename,
            model_path=model_path,
            current_epoch=epoch,
            learning_rate=curr_learning_rate,
            objf=objf,
            best_objf=best_objf,
            valid_objf=valid_objf,
            best_valid_objf=best_valid_objf,
            best_epoch=best_epoch,
        )

    logging.warning("Done")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
#                2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import logging
import os
import sys

from pathlib import Path
from typing import Union

import torch
from sklearn.metrics import classification_report

import k2
from k2 import Fsa, SymbolTable
from kaldialign import edit_distance
from lhotse import CutSet
from lhotse.dataset import SingleCutSampler

from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.decoding.graph import compile_HLG
from snowfall.models import AcousticModel

from ovad.graph import prepare_decoding_graph
from ovad.dataset import K2VadDataset
from ovad.models import TdnnLstm1a

from ovad.utils import create_and_write_segments


def decode(
    dataloader: torch.utils.data.DataLoader,
    model: AcousticModel,
    device: Union[str, torch.device],
    HCLG: Fsa,
):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = []  # a list of pair [ref_labels, hyp_labels]
    for batch_idx, batch in enumerate(dataloader):
        feature = batch["inputs"]  # (N, T, C)
        supervisions = batch["supervisions"]

        feature = feature.to(device)

        # Since we are decoding with a k2 graph here, we need to create appropriate
        # supervisions. The segments need to be ordered in decreasing order of
        # length (although in our case all segments are of same length)
        supervision_segments = torch.stack(
            (
                supervisions["sequence_idx"],
                torch.floor_divide(
                    supervisions["start_frame"], model.subsampling_factor
                ),
                torch.floor_divide(supervisions["duration"], model.subsampling_factor),
            ),
            1,
        ).to(torch.int32)
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        with torch.no_grad():
            nnet_output = model(feature)

        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
        # assert HLG.is_cuda()
        assert (
            HCLG.device == nnet_output.device
        ), f"Check failed: HCLG.device ({HCLG.device}) == nnet_output.device ({nnet_output.device})"

        lattices = k2.intersect_dense_pruned(HCLG, dense_fsa_vec, 20.0, 7.0, 30, 10000)
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        assert best_paths.shape[0] == supervisions["is_voice"].shape[0]

        # best_paths is an FsaVec, and each of its FSAs is a linear FSA
        references = supervisions["is_voice"][indices]
        for i in range(references.shape[0]):
            ref = references[i, :]
            hyp = k2.arc_sort(best_paths[i]).arcs_as_tensor()[:-1, 2].detach().cpu()
            assert (
                ref.shape[0] == hyp.shape[0]
            ), "reference and hypothesis have unequal number of frames, {} vs. {}".format(
                ref.shape[0], hyp.shape[0]
            )
            results.append((supervisions["cut"][indices[i]], ref, hyp))

        if batch_idx % 10 == 0:
            logging.info(
                "batch {}, cuts processed until now is {}/{} ({:.6f}%)".format(
                    batch_idx,
                    num_cuts,
                    tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100,
                )
            )

        num_cuts += supervisions["is_voice"].shape[0]

    return results


def main():
    if not torch.cuda.is_available():
        logging.error("No GPU detected!")
        sys.exit(-1)
    device_id = 0
    device = torch.device("cuda", device_id)
    # Reserve the GPU with a dummy variable
    reserve_variable = torch.ones(1).to(device)

    exp_dir = Path("exp-tl1a-adam-xent")
    setup_logger("{}/log/log-decode".format(exp_dir), log_level="debug")

    if not os.path.exists(exp_dir / "HCLG.pt"):
        logging.info("Preparing decoding graph")
        # sym_str = """
        #     <eps> 0
        #     silence 1
        #     speech 2
        # """
        # symbol_table = k2.SymbolTable.from_str(sym_str)

        HCLG = prepare_decoding_graph(
            min_silence_duration=0.03,
            min_speech_duration=0.3,
            max_speech_duration=10.0,
        )

        # Arc sort the HCLG since it is needed for intersect
        logging.info("Sorting decoding graph by outgoing arcs")
        HCLG = k2.arc_sort(HCLG)

        # HCLG.symbols = symbol_table
        torch.save(HCLG.as_dict(), exp_dir / "HCLG.pt")
    else:
        logging.info("Loading pre-compiled decoding graph")
        d = torch.load(exp_dir / "HCLG.pt")
        HCLG = k2.Fsa.from_dict(d)

    # load dataset
    feature_dir = Path("exp/data")
    logging.info("About to get test cuts")
    cuts_test = CutSet.from_json(feature_dir / "cuts_test.json.gz")

    logging.info("About to create test dataset")
    test = K2VadDataset(cuts_test, return_cuts=True)
    sampler = SingleCutSampler(cuts_test, max_frames=100000)
    logging.info("About to create test dataloader")
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=None, sampler=sampler, num_workers=1
    )

    logging.info("About to load model")
    model = TdnnLstm1a(
        num_features=80,
        num_classes=2,  # speech/silence
        subsampling_factor=1,
    )

    checkpoint = os.path.join(exp_dir, "best_model.pt")
    load_checkpoint(checkpoint, model)
    model.to(device)
    model.eval()

    logging.info("convert decoding graph to device")
    HCLG = HCLG.to(device)
    HCLG.requires_grad_(False)
    logging.info("About to decode")
    results = decode(dataloader=test_dl, model=model, device=device, HCLG=HCLG)

    # Compute frame-level accuracy and precision-recall metrics
    y_true = []
    y_pred = []
    for result in results:
        cut, ref, hyp = result
        y_true.append(ref)
        y_pred.append(hyp)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    logging.info(
        "Results: \n{}".format(
            classification_report(y_true, y_pred, target_names=["silence", "speech"])
        )
    )
    # Create output segments per recording
    create_and_write_segments(
        [result[0] for result in results],  # cuts
        [result[2] for result in results],  # outputs
        exp_dir / "segments",  # segments file
    )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()

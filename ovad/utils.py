#!/usr/bin/env python

# Copyright 2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import argparse
import logging
import math

from typing import Optional, List
from dataclasses import dataclass
from itertools import groupby
from collections import defaultdict
from pathlib import Path

import torch
from k2 import Fsa
from lhotse import Cut


@dataclass
class Segment:
    """Stores information about a segment"""

    reco_id: str
    seg_id: str
    start: float
    duration: float

    def __init__(self, reco_id: str, start: float, end: float) -> None:
        self.reco_id = reco_id
        self.seg_id = f"{reco_id}-{int(start*100):06d}-{int(end*100):06d}"
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"{self.seg_id} {self.reco_id} {self.start} {self.end}"


def _merge_consecutive_segments(
    segments: List[Segment], tol: Optional[float] = 0.01
) -> List[Segment]:
    segments = sorted(segments, key=lambda x: x.start)
    merged_segs = [segments[0]]
    for seg in segments[1:]:
        if seg.start <= merged_segs[-1].end + tol:
            merged_segs[-1].end = seg.end
        else:
            merged_segs.append(seg)


def create_and_write_segments(
    cuts: List[Cut], outputs: List[torch.Tensor], file: Path
) -> List[Segment]:
    # Create segments from the frame-wise binary outputs
    segments = []
    for cut, output in zip(cuts, outputs):
        speech_frames = output.nonzero()
        while len(speech_frames) > 0:
            i = 0
            while (
                i < len(speech_frames)
                and i + 1 < len(speech_frames)
                and speech_frames[i + 1] == speech_frames[i] + 1
            ):
                i += 1
            start_frame = speech_frames[0].item()
            end_frame = speech_frames[i].item()
            segments.append(
                Segment(
                    cut.recording_id,
                    cut.start + 0.01 * start_frame,
                    cut.start + 0.01 * end_frame,
                )
            )
            speech_frames = speech_frames[i + 1 :]
    segments = sorted(segments, key=lambda x: x.reco_id)
    reco2segs = defaultdict(
        list, {reco_id: g for reco_id, g in groupby(segments, key=lambda x: x.reco_id)}
    )

    # Merge consecutive segments for each recording and write to file
    with open(file, "w") as f:
        for reco_id in reco2segs:
            merged_segs = _merge_consecutive_segments(reco2segs[reco_id])
            for seg in merged_segs:
                f.write(f"{str(seg)}\n")

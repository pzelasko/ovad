#!/usr/bin/env python

# Copyright 2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import argparse
import logging
import math

from typing import Optional, List
from dataclasses import dataclass

import torch
from k2 import Fsa


@dataclass
class Transition:
    """Defines a transition in the graph"""

    src: int
    dest: int = None
    label: str = None
    score: float = None

    def __str__(self) -> str:
        if self.dest is not None:
            return f"{self.src} {self.dest} {self.label} {self.score:.3f}"
        else:
            return f"{self.src}"


def prepare_decoding_graph(
    min_silence_duration: Optional[float] = 0.03,
    min_speech_duration: Optional[float] = 0.3,
    max_speech_duration: Optional[float] = 10,
    transition_scale: Optional[float] = 1.0,
    loop_scale: Optional[float] = 0.1,
    frame_shift: Optional[float] = 0.01,
    edge_silence_probability: Optional[float] = 0.5,
    transition_probability: Optional[float] = 0.1,
) -> Fsa:
    """
    Prepares a graph with a simple HMM topology for segmentation with minimum
    and maximum speech duration constraints and minimum silence duration constraint.

    :param min_silence_duration: float (default = 0.03), minimum duration for silence
    :param min_speech_duration: float (default = 0.3), minimum duration for speech
    :param max_speech_duration: float (default = 10), maximum duration for speech
    :param transition_scale: float (default = 1.0), scale on transition probabilities relative
        to LM weights
    :param loop_scale: float (default = 0.1), scale on self-loop log-probabilities
        relative to LM weights
    :param frame_shift: float (default = 0.01), frame shift in seconds
    :param edge_silence_probability (default = 0.5), probability of silence at the edges
    :param transition_probability (default = 0.1), transition probability for silence to
        speech, or vice versa
    :return: a k2.Fsa type with the encoded constraints
    """

    min_states_silence = int(min_silence_duration / frame_shift + 0.5)
    min_states_speech = int(min_speech_duration / frame_shift + 0.5)
    max_states_speech = int(max_speech_duration / frame_shift + 0.5)

    silence = "0"
    speech = "1"

    transitions = []

    # Initial transition to silence
    transitions.append(
        Transition(
            src=0, dest=1, label=silence, score=-math.log(edge_silence_probability)
        )
    )
    silence_start_state = 1

    # Silence min duration transitions
    # 1->2, 2->3 and so on until
    # (1 + min_states_silence - 2) -> (1 + min_states_silence - 1)  ...
    for state in range(
        silence_start_state, silence_start_state + min_states_silence - 1
    ):
        transitions.append(
            Transition(src=state, dest=state + 1, label=silence, score=0.0)
        )
    silence_last_state = silence_start_state + min_states_silence - 1

    # Silence self-loop
    transitions.append(
        Transition(
            src=silence_last_state, dest=silence_last_state, label=silence, score=0.0
        )
    )

    speech_start_state = silence_last_state + 1

    # Initial transition to speech
    transitions.append(
        Transition(
            src=0,
            dest=speech_start_state,
            label=speech,
            score=-math.log(1.0 - edge_silence_probability),
        )
    )

    # Silence to speech transition
    transitions.append(
        Transition(
            src=silence_last_state,
            dest=speech_start_state,
            label=speech,
            score=-math.log(transition_probability),
        )
    )

    # Speech min duration
    for state in range(speech_start_state, speech_start_state + min_states_speech - 1):
        transitions.append(
            Transition(src=state, dest=state + 1, label=speech, score=0.0)
        )

    # Speech max duration
    for state in range(
        speech_start_state + min_states_speech - 1,
        speech_start_state + max_states_speech - 1,
    ):
        transitions.append(
            Transition(src=state, dest=state + 1, label=speech, score=0.0)
        )
        transitions.append(
            Transition(
                src=state,
                dest=silence_start_state,
                label=silence,
                score=-math.log(transition_probability),
            )
        )

    speech_last_state = speech_start_state + max_states_speech - 1
    final_state = speech_last_state + 1

    # Transition to silence after max duration of speech
    transitions.append(
        Transition(
            src=speech_last_state, dest=silence_start_state, label=silence, score=0.0
        )
    )

    for state in range(1, speech_start_state):
        transitions.append(
            Transition(src=state, dest=final_state, label="-1", score=0.0)
        )

    for state in range(speech_start_state, speech_last_state + 1):
        transitions.append(
            Transition(src=state, dest=final_state, label="-1", score=0.0)
        )

    # Add final state
    transitions.append(Transition(src=final_state))

    # Sort FSA transitions by src state because k2 requires it
    transitions = "\n".join([str(t) for t in sorted(transitions, key=lambda x: x.src)])

    graph = Fsa.from_str(transitions)

    return graph

"""Torch-free serving path for the next-video transformer: onnxruntime + numpy.

The ONNX graph maps a left-padded history of video ids to a score for every video;
this module does the padding, removes already-watched videos and picks the top k.
"""

from __future__ import annotations

import pathlib

import numpy as np


class NextVideoRecommender:
    def __init__(self, onnx_path: pathlib.Path) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        history_input = self._session.get_inputs()[0]
        self._max_len = int(history_input.shape[1])
        self._num_videos = int(self._session.get_outputs()[0].shape[1]) - 1

    def scores(self, histories: list[list[int]]) -> np.ndarray:
        """Score per video id (column 0 is padding) after each history: [len(histories), num_videos + 1]."""
        padded = np.zeros((len(histories), self._max_len), dtype=np.int64)
        for row, history in enumerate(histories):
            # Ids the model was not trained with have no embedding; skip them rather than fail.
            known = [video for video in history if 0 < video <= self._num_videos][-self._max_len :]
            if known:
                padded[row, self._max_len - len(known) :] = known
        return self._session.run(None, {"history": padded})[0]

    def recommend(self, histories: list[list[int]], k: int) -> list[list[int]]:
        """Top-k unwatched video ids after each history, best first."""
        scores = self.scores(histories)
        scores[:, 0] = -np.inf
        out = []
        for row, history in enumerate(histories):
            watched = [video for video in history if 0 < video <= self._num_videos]
            scores[row, watched] = -np.inf
            top = np.argpartition(-scores[row], k)[:k]
            out.append([int(video) for video in top[np.argsort(-scores[row, top], kind="stable")]])
        return out

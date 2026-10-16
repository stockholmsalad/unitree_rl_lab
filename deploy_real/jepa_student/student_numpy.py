"""Float32 depth-only GRU inference; requires only NumPy and the Python standard library."""
import json
from pathlib import Path

import numpy as np


def elu(x):
    result = np.asarray(x, dtype=np.float32).copy()
    negative = result <= 0
    result[negative] = np.expm1(result[negative])
    return result


def linear(x, weight, bias):
    return np.matmul(x, weight.T) + bias


def conv_valid(x, weight, bias, stride):
    batch, channels, height, width = x.shape
    outputs, _, kh, kw = weight.shape
    oh, ow = (height - kh) // stride + 1, (width - kw) // stride + 1
    shape = (batch, channels, oh, ow, kh, kw)
    sb, sc, sh, sw = x.strides
    windows = np.lib.stride_tricks.as_strided(x, shape=shape,
                    strides=(sb, sc, sh * stride, sw * stride, sh, sw))
    matrix = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, channels * kh * kw)
    return (np.matmul(matrix, weight.reshape(outputs, -1).T) + bias).reshape(batch, oh, ow, outputs).transpose(0, 3, 1, 2)


class StudentNumpy:
    def __init__(self, prefix):
        prefix = str(prefix)
        self.meta = json.loads(Path(prefix + ".json").read_text())
        cfg = self.meta["student_cfg"]
        if cfg["method"] != "gru" or cfg.get("backbone", "gru") != "gru" or cfg.get("proprio_input", False):
            raise ValueError("depth-only GRU만 지원한다")
        with np.load(prefix + ".npz") as archive:
            self.w = {key: archive[key].astype(np.float32) for key in archive.files}
        self.h = np.zeros((1, cfg["context_dim"]), dtype=np.float32)
        self.strides = cfg.get("cnn_strides", [2, 2, 2])
        self.last_z = np.zeros((1, cfg["terrain_dim"]), dtype=np.float32)

    def reset(self):
        self.h.fill(0)

    def _linear(self, x, key):
        return linear(x, self.w[key + ".weight"], self.w[key + ".bias"])

    def _cnn(self, depth):
        x = np.asarray(depth, dtype=np.float32).reshape((1, 2) + tuple(self.meta["student_cfg"]["image_hw"]))
        for i, layer in enumerate((0, 2, 4)):
            key = "student.cnn.conv.{}.".format(layer)
            x = elu(conv_valid(x, self.w[key + "weight"], self.w[key + "bias"], self.strides[i]))
        return elu(self._linear(x.reshape(1, -1), "student.cnn.head.1"))

    def _gru(self, x):
        prefix = "student.backbone.cell."
        gi = linear(x, self.w[prefix + "weight_ih"], self.w[prefix + "bias_ih"])
        gh = linear(self.h, self.w[prefix + "weight_hh"], self.w[prefix + "bias_hh"])
        ir, iz, inn = np.split(gi, 3, axis=-1)
        hr, hz, hn = np.split(gh, 3, axis=-1)
        r = 1.0 / (1.0 + np.exp(-(ir + hr)))
        z = 1.0 / (1.0 + np.exp(-(iz + hz)))
        n = np.tanh(inn + r * hn)
        self.h = ((1.0 - z) * n + z * self.h).astype(np.float32)

    def step(self, depth, proprio, fresh):
        if fresh:
            frame = self._cnn(depth)
            if "student.frame_projection.weight" in self.w:
                frame = self._linear(frame, "student.frame_projection")
            self._gru(frame)
        z = elu(self._linear(self.h, "student.terrain_head.0"))
        self.last_z = self._linear(z, "student.terrain_head.2")
        p = np.asarray(proprio, dtype=np.float32).reshape(1, 45)
        p = (p - self.w["teacher.obs_normalizer._mean"]) / (
            self.w["teacher.obs_normalizer._std"] + np.float32(self.meta["normalizer_eps"]))
        x = np.concatenate((p, self.last_z), axis=-1)
        for layer in (0, 2, 4):
            x = elu(self._linear(x, "teacher.mlp.{}".format(layer)))
        return self._linear(x, "teacher.mlp.6")[0].astype(np.float32)

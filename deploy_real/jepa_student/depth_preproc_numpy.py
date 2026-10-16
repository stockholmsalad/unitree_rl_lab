"""NumPy 1.17-compatible equivalent of depth_proc.downsample_real(nearest)."""
import numpy as np


def downsample_nearest(depth_m, out_hw=(64, 112), near=0.28):
    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape
    out_h, out_w = out_hw
    row_edges = np.arange(out_h + 1) * height // out_h
    col_edges = np.arange(out_w + 1) * width // out_w
    kr = int(np.max(np.diff(row_edges)))
    kc = int(np.max(np.diff(col_edges)))
    rows = row_edges[:-1, None] + np.arange(kr)[None, :]
    cols = col_edges[:-1, None] + np.arange(kc)[None, :]
    row_inside = rows < row_edges[1:, None]
    col_inside = cols < col_edges[1:, None]
    rows = np.minimum(rows, height - 1)
    cols = np.minimum(cols, width - 1)
    patch = depth[rows[:, :, None, None], cols[None, None, :, :]].transpose(0, 2, 1, 3)
    inside = (row_inside[:, None, :, None] & col_inside[None, :, None, :]).reshape(out_h, out_w, -1)
    patch = patch.reshape(out_h, out_w, -1)
    valid = inside & np.isfinite(patch) & (patch >= near)
    center_r = (np.arange(out_h) + 0.5) * height / out_h - 0.5
    center_c = (np.arange(out_w) + 0.5) * width / out_w - 0.5
    dr = rows.astype(np.float32) - center_r[:, None]
    dc = cols.astype(np.float32) - center_c[:, None]
    distance = (dr[:, None, :, None] ** 2 + dc[None, :, None, :] ** 2).reshape(out_h, out_w, -1)
    choice = np.argmin(np.where(valid, distance, np.inf), axis=-1)
    result = np.take_along_axis(patch, choice[..., None], axis=-1)[..., 0]
    return np.where(np.any(valid, axis=-1), result, 0.0).astype(np.float32)


def to_two_channel(depth_m, near=0.28, far=2.0):
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= near)
    normalized = np.where(valid, (np.minimum(np.where(valid, depth, 0.0), far) - near) / (far - near), 0.0)
    return np.stack((normalized.astype(np.float32), valid.astype(np.float32)))


def preprocess_mm(image_mm, out_hw=(64, 112), near=0.28, far=2.0):
    depth_m = np.asarray(image_mm, dtype=np.float32) * np.float32(0.001)
    return to_two_channel(downsample_nearest(depth_m, out_hw, near), near, far)

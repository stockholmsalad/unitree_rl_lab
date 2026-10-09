"""D435i 높이맵 가시성 기하·가림 테스트."""

import numpy as np

from unitree_rl_lab.jepa_loco.eval.blind_spot import (
    height_patch_xy, project_base_points, visible_in_depth,
)
from unitree_rl_lab.jepa_loco.sensors.d435i_cfg import D435iParams


def test_patch_shape_and_flat_ground_blind_spot():
    camera = D435iParams()
    xy = height_patch_xy((1.6, 1.0), 0.1)
    assert xy.shape == (187, 2)
    base_z_m = 0.279
    points = np.array([[0.30, 0, -base_z_m], [0.60, 0, -base_z_m],
                       [1.00, 0, -base_z_m], [2.00, 0, -base_z_m]])
    _, _, _, valid = project_base_points(
        points, camera.depth_origin_in_base, camera.pitch_down_deg,
        camera.intrinsics(camera.render_wh), camera.render_wh,
        camera.min_range, camera.clip_far,
    )
    assert not valid[0]
    assert valid[1:3].all()
    assert valid.shape == (4,)


def test_depth_occlusion_and_shape():
    camera = D435iParams()
    point = np.array([[0.8, 0.0, -0.279]])
    u, v, axial, valid = project_base_points(
        point, camera.depth_origin_in_base, camera.pitch_down_deg,
        camera.intrinsics(camera.render_wh), camera.render_wh,
        camera.min_range, camera.clip_far,
    )
    assert valid[0]
    depth = np.full((camera.render_wh[1], camera.render_wh[0]), axial[0])
    geometric, visible = visible_in_depth(
        point, depth, camera.depth_origin_in_base, camera.pitch_down_deg,
        camera.intrinsics(camera.render_wh), camera.min_range, camera.clip_far, 0.05,
    )
    assert geometric.tolist() == visible.tolist() == [True]
    depth[int(round(v[0])), int(round(u[0]))] -= 0.2
    _, visible = visible_in_depth(
        point, depth, camera.depth_origin_in_base, camera.pitch_down_deg,
        camera.intrinsics(camera.render_wh), camera.min_range, camera.clip_far, 0.05,
    )
    assert visible.tolist() == [False]

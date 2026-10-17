"""연속 코스 설계와 성공 판정의 회귀 테스트."""

import math

import numpy as np
import pytest

from unitree_rl_lab.jepa_loco.eval.course import (
    CourseEvalCfg, CourseTerrainCfg, course_outcome, course_terrain, designed_height,
    heading_yaw_rate, section_edges, section_index,
)


def test_course_edges_and_heights():
    c = CourseEvalCfg()
    e = section_edges(c)
    assert np.allclose(e, (2, 4, 5.5, 6.7, 8.2, 9.4, 10.9, 13.9, 15.4))
    h = 0.12
    ramp = 2 * math.tan(math.radians(10))
    assert np.isclose(designed_height(0.5, h, c), 0)
    assert np.isclose(designed_height(3.0, h, c), ramp / 2)
    assert np.isclose(designed_height(4.5, h, c), ramp)
    assert np.isclose(designed_height(5.5, h, c), ramp + h)
    assert np.isclose(designed_height(6.4, h, c), ramp + 4 * h)
    assert np.isclose(designed_height(8.2, h, c), ramp + 3 * h)
    assert np.isclose(designed_height(9.1, h, c), ramp)
    assert np.isclose(designed_height(14.5, h, c), ramp)


def test_section_and_outcomes():
    c = CourseEvalCfg()
    assert [section_index(x, c) for x in (0.0, 2.0, 5.5, 13.9, 15.0)] == [0, 1, 3, 8, 8]
    assert course_outcome(True, False, False, False, False) == "success"
    assert course_outcome(False, True, False, False, False) == "off_course"
    assert course_outcome(False, False, True, False, False) == "fall"
    assert course_outcome(False, False, True, True, False) == "stall"
    assert course_outcome(False, False, False, False, True) == "stall"


def test_heading_wrap_and_limit():
    c = CourseEvalCfg()
    assert heading_yaw_rate(0.1, c) == pytest.approx(-0.1)
    assert heading_yaw_rate(2 * math.pi - 0.1, c) == pytest.approx(0.1)
    assert heading_yaw_rate(math.pi / 2, c) == pytest.approx(-0.3)
    c.heading_hold = False
    assert heading_yaw_rate(1.0, c) == 0.0


def test_mesh_vertical_steps_and_ramp():
    c = CourseEvalCfg()
    terrain_cfg = CourseTerrainCfg(course=c, step_height_m=0.12, size=(16.0, 4.0))
    assert terrain_cfg.function is course_terrain
    meshes, origin = terrain_cfg.function(0.5, terrain_cfg)
    assert len(meshes) == 1
    assert np.allclose(origin, (8.0, 2.0, 0.0))
    verts = meshes[0].vertices
    # 계단 수직면은 같은 x의 다른 z vertex를 가지며, 경사면의 x는 유지된다.
    up = verts[(verts[:, 0] > 5.4) & (verts[:, 0] < 6.7)]
    assert len(np.unique(up[:, 0])) < 13
    ramp = verts[(verts[:, 0] > 2.2) & (verts[:, 0] < 3.8)]
    assert len(np.unique(ramp[:, 0])) >= 15
    # 학습 HfRandomUniformTerrainCfg의 border 0.25m는 격자에서 int(0.25/0.1)+1=3셀이다.
    grid = verts[:, 2].reshape(161, 41)
    rough_start = round(section_edges(c)[6] / c.horizontal_scale_m)
    baseline = designed_height(section_edges(c)[6], 0.12, c)
    assert np.all(np.abs(grid[rough_start + 2, :] - baseline) < 0.006)
    assert grid[rough_start + 3, 4] > baseline

"""Oracle PPO용 에피소드 XY 경로 길이 누적 및 지형 종류별 레벨 로깅."""

from __future__ import annotations

import torch

from .diagnostic_env import JepaDiagnosticEnv
from .path_curriculum import terrain_column_type_ids, terrain_level_means_by_type, xy_path_increment


class OracleCurriculumEnv(JepaDiagnosticEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # train.py가 cfg.seed를 설정한 뒤 환경을 생성한다. 세 oracle에 동일한 seed 지형을 고정한다.
        if cfg.seed is not None:
            cfg.scene.terrain.terrain_generator.seed = cfg.seed
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.episode_path_length_m = torch.zeros(self.num_envs, device=self.device)
        self._path_previous_xy = self.scene["robot"].data.root_pos_w[:, :2].clone()
        self._path_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        generator = self.cfg.scene.terrain.terrain_generator
        self._terrain_type_names = list(generator.sub_terrains)
        self._terrain_column_type_ids = terrain_column_type_ids(
            generator.num_cols, [term.proportion for term in generator.sub_terrains.values()],
            self.cfg.terrain_column_assignment_eps,
        ).to(self.device)

    def _reset_idx(self, env_ids):
        # IsaacLab은 이 호출 직후 curriculum을 계산한다. 종료를 일으킨 마지막 스텝을 먼저 더한다.
        if hasattr(self, "episode_path_length_m"):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            xy = self.scene["robot"].data.root_pos_w[ids, :2]
            self.episode_path_length_m[ids] += xy_path_increment(xy, self._path_previous_xy[ids], self._path_valid[ids])
        super()._reset_idx(env_ids)
        if hasattr(self, "episode_path_length_m"):
            self.episode_path_length_m[ids] = 0.0
            self._path_previous_xy[ids] = self.scene["robot"].data.root_pos_w[ids, :2]
            self._path_valid[ids] = True

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        active = ~(terminated | truncated)
        xy = self.scene["robot"].data.root_pos_w[:, :2]
        self.episode_path_length_m += xy_path_increment(xy, self._path_previous_xy, active & self._path_valid)
        self._path_previous_xy[:] = xy
        self._path_valid[:] = True
        terrain = self.scene.terrain
        by_type = terrain_level_means_by_type(
            terrain.terrain_levels, terrain.terrain_types,
            self._terrain_type_names, self._terrain_column_type_ids,
        )
        log = extras.setdefault("log", {})
        log["Curriculum/path_length_m_mean"] = self.episode_path_length_m.mean()
        for name, value in by_type.items():
            log[f"Curriculum/terrain_level/{name}"] = value
        return obs, reward, terminated, truncated, extras

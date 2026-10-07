"""Oracle PPO용 에피소드 XY 경로 길이 누적 및 지형 종류별 레벨 로깅."""

from __future__ import annotations

import torch

from .diagnostic_env import JepaDiagnosticEnv
from .path_curriculum import (
    obstacle_clearance_edges, obstacle_clearance_step, stair_first_step_edges,
    terrain_column_type_ids, terrain_level_means_by_type, xy_path_increment,
)


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
        progress = cfg.progress
        if (progress.stair_approach_margin_m < 0 or progress.first_step_progress_margin_m < 0
                or progress.stall_speed_mps < 0 or progress.forward_attempt_speed_mps < 0
                or progress.stall_hold_steps < 1):
            raise ValueError("진단 여유 거리와 정지 속도는 음수가 될 수 없다")
        self._obstacle_clearance_edges = obstacle_clearance_edges(
            generator.sub_terrains, generator.size, progress.clearance_margin_m,
        ).to(self.device)
        self._terrain_type_per_env = self._terrain_column_type_ids[self.scene.terrain.terrain_types.long()]
        self._clearance_edge_per_env = self._obstacle_clearance_edges[self._terrain_type_per_env]
        self.episode_obstacle_cleared = self._clearance_edge_per_env <= 0
        self._clearance_consecutive = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._max_axis_excursion_m = torch.zeros(self.num_envs, device=self.device)
        self._near_step_stall_consecutive = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._near_step_max_outward_speed_mps = torch.zeros(self.num_envs, device=self.device)
        self._swing_foot_z_max_m = torch.full((self.num_envs,), -float("inf"), device=self.device)
        self._stairs_up_type_ids = torch.tensor(
            [i for i, name in enumerate(self._terrain_type_names) if name.startswith("stairs_up")],
            device=self.device, dtype=torch.long,
        )
        stair_cfg = next((term for name, term in generator.sub_terrains.items() if name.startswith("stairs_up")), None)
        if stair_cfg is not None:
            self._first_riser_m, self._first_step_far_edge_m = stair_first_step_edges(stair_cfg, generator.size)
        else:
            self._first_riser_m = float("inf")
            self._first_step_far_edge_m = float("inf")
        robot = self.scene["robot"]
        foot_ids, foot_names = robot.find_bodies(".*_foot")
        contact_ids, contact_names = self.scene.sensors["contact_forces"].find_bodies(".*_foot")
        contact_by_name = dict(zip(contact_names, contact_ids))
        self._robot_foot_ids = foot_ids
        self._contact_foot_ids = [contact_by_name[name] for name in foot_names]

    def _update_progress(self, ids):
        if ids.numel() == 0:
            return
        robot = self.scene["robot"]
        origin = self.scene.env_origins[ids]
        root_xy = robot.data.root_pos_w[ids, :2]
        local_xy = root_xy - origin[:, :2]
        axis_excursion = local_xy.abs().amax(dim=1)
        self._max_axis_excursion_m[ids] = torch.maximum(self._max_axis_excursion_m[ids], axis_excursion)
        consecutive, cleared = obstacle_clearance_step(
            root_xy, origin[:, :2], self._clearance_edge_per_env[ids],
            self._clearance_consecutive[ids], self.episode_obstacle_cleared[ids],
            self.cfg.progress.clearance_hold_steps,
        )
        self._clearance_consecutive[ids] = consecutive
        self.episode_obstacle_cleared[ids] = cleared
        near_stair = (
            torch.isin(self._terrain_type_per_env[ids], self._stairs_up_type_ids)
            & (axis_excursion >= self._first_riser_m - self.cfg.progress.stair_approach_margin_m)
            & (axis_excursion <= self._first_step_far_edge_m)
        )
        velocity_xy = robot.data.root_lin_vel_w[ids, :2]
        dominant_axis = local_xy.abs().argmax(dim=1, keepdim=True)
        outward_speed = velocity_xy.gather(1, dominant_axis).squeeze(1) * local_xy.gather(1, dominant_axis).squeeze(1).sign()
        self._near_step_max_outward_speed_mps[ids] = torch.maximum(
            self._near_step_max_outward_speed_mps[ids], torch.where(near_stair, outward_speed, 0.0),
        )
        speed = torch.linalg.vector_norm(velocity_xy, dim=1)
        self._near_step_stall_consecutive[ids] = torch.where(
            near_stair & (speed <= self.cfg.progress.stall_speed_mps),
            self._near_step_stall_consecutive[ids] + 1, 0,
        )
        feet_z = robot.data.body_pos_w[ids][:, self._robot_foot_ids, 2] - origin[:, None, 2]
        air_time = self.scene.sensors["contact_forces"].data.current_air_time[ids][:, self._contact_foot_ids]
        swing = air_time > 0
        swing_z = torch.where(swing & near_stair[:, None], feet_z, -float("inf")).amax(dim=1)
        self._swing_foot_z_max_m[ids] = torch.maximum(self._swing_foot_z_max_m[ids], swing_z)

    def _reset_idx(self, env_ids):
        # IsaacLab은 이 호출 직후 curriculum을 계산한다. 종료를 일으킨 마지막 스텝을 먼저 더한다.
        if hasattr(self, "episode_path_length_m"):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            xy = self.scene["robot"].data.root_pos_w[ids, :2]
            self.episode_path_length_m[ids] += xy_path_increment(xy, self._path_previous_xy[ids], self._path_valid[ids])
            self._update_progress(ids)
            names = self._terrain_type_names
            type_ids = self._terrain_type_per_env[ids]
            stair = torch.isin(type_ids, self._stairs_up_type_ids)
            approached = self._max_axis_excursion_m[ids] >= self._first_riser_m - self.cfg.progress.stair_approach_margin_m
            advanced = self._max_axis_excursion_m[ids] >= self._first_step_far_edge_m + self.cfg.progress.first_step_progress_margin_m
            stalled = stair & approached & ~advanced & (
                self._near_step_stall_consecutive[ids] >= self.cfg.progress.stall_hold_steps
            )
            advanced &= stair
            attempted = stair & ~stalled & (
                advanced | (self._near_step_max_outward_speed_mps[ids] >= self.cfg.progress.forward_attempt_speed_mps)
            )
            reward_sums = {
                name: values[ids].clone() for name, values in self.reward_manager._episode_sums.items()
            }
            if hasattr(self, "_swing_clearance_sum"):
                clearance_sum = self._swing_clearance_sum[ids].clone()
                clearance_count = self._swing_clearance_count[ids].clone()
                clearance_max = self._swing_clearance_max[ids].clone()
            else:
                clearance_sum = clearance_count = clearance_max = None
            cleared = self.episode_obstacle_cleared[ids].clone()
            curriculum_term = self.cfg.curriculum.terrain_levels
            path_qualified = (
                self.episode_path_length_m[ids] > (
                    self.cfg.scene.terrain.terrain_generator.size[0] * curriculum_term.params["up_fraction"]
                ) if curriculum_term is not None else torch.zeros_like(cleared)
            )
            swing_height = self._swing_foot_z_max_m[ids].clone()
        super()._reset_idx(env_ids)
        if hasattr(self, "episode_path_length_m"):
            log = self.extras.setdefault("log", {})
            for type_id, name in enumerate(names):
                mask = type_ids == type_id
                if mask.any() and self._obstacle_clearance_edges[type_id] > 0:
                    log[f"Curriculum/obstacle_clear_rate/{name}"] = cleared[mask].float().mean()
                    log[f"Curriculum/path_only_false_promotion_rate/{name}"] = (
                        path_qualified[mask] & ~cleared[mask]
                    ).float().mean()
                if mask.any() and name in ("flat", "stairs_up") and clearance_sum is not None:
                    count = clearance_count[mask].sum()
                    if count > 0:
                        log[f"Diagnosis/swing_clearance_mean_m/{name}"] = clearance_sum[mask].sum() / count
                        log[f"Diagnosis/swing_clearance_max_m/{name}"] = clearance_max[mask].amax()
            for label, mask in (("approach_stall", stalled), ("forward_attempt", attempted)):
                if mask.any():
                    for name, values in reward_sums.items():
                        log[f"Diagnosis/stairs_up/{label}/episode_reward_sum/{name}"] = values[mask].mean()
                    log[f"Diagnosis/stairs_up/{label}/episode_count"] = mask.sum()
                    finite = torch.isfinite(swing_height[mask])
                    if finite.any():
                        log[f"Diagnosis/stairs_up/{label}/swing_foot_z_max_m"] = swing_height[mask][finite].mean()
            if stair.any():
                log["Diagnosis/stairs_up/first_step_progress_rate"] = advanced[stair].float().mean()
            self.episode_path_length_m[ids] = 0.0
            self._path_previous_xy[ids] = self.scene["robot"].data.root_pos_w[ids, :2]
            self._path_valid[ids] = True
            self._terrain_type_per_env[ids] = self._terrain_column_type_ids[self.scene.terrain.terrain_types[ids].long()]
            self._clearance_edge_per_env[ids] = self._obstacle_clearance_edges[self._terrain_type_per_env[ids]]
            self.episode_obstacle_cleared[ids] = self._clearance_edge_per_env[ids] <= 0
            self._clearance_consecutive[ids] = 0
            self._max_axis_excursion_m[ids] = 0.0
            self._near_step_stall_consecutive[ids] = 0
            self._near_step_max_outward_speed_mps[ids] = 0.0
            self._swing_foot_z_max_m[ids] = -float("inf")
            if clearance_sum is not None:
                self._swing_clearance_sum[ids] = 0.0
                self._swing_clearance_count[ids] = 0.0
                self._swing_clearance_max[ids] = -float("inf")

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        active = ~(terminated | truncated)
        xy = self.scene["robot"].data.root_pos_w[:, :2]
        self.episode_path_length_m += xy_path_increment(xy, self._path_previous_xy, active & self._path_valid)
        self._update_progress(torch.nonzero(active, as_tuple=True)[0])
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

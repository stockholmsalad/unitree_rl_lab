"""10 Hz depth 관측 항: 렌더링 → 노이즈 → 지연 → 2채널 (CLAUDE.md §3, §4, §6).

타이밍:
- 렌더링은 ``sim.render_interval`` (= decimation × depth_period_steps 물리 스텝)마다 한 번.
  카메라는 ``update_period=0`` 으로 매 물리 스텝 최신 렌더 결과를 읽는다(env reset 으로 갱신 위상이
  렌더링과 어긋나 프레임이 낡는 것을 막는다).
- 새 렌더가 오면 env 별로 추가 지연 d_add ~ U[d_min − d0, d_max − d0] 제어 스텝 뒤에 공개한다.
  d0 = IsaacLab 렌더링 고유 지연(측정값, 기본 0). 실제 지연 = d0 + d_add ∈ [d_min, d_max].
- 공개된 스텝에 ``fresh`` = 1. 정책 backbone 은 이 플래그로만 갱신한다.
- env reset 이전에 찍힌 렌더(순간이동 전 장면)는 버린다. reset 뒤 첫 프레임 전까지 입력은 전부 결측(mask 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from ..sensors.depth_noise import depth_noise
from ..sensors.depth_proc import to_two_channel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DepthFrame(ManagerTermBase):
    """params: sensor_cfg, fx, near, far, delay_range, inherent_delay, noise (depth_noise kwargs | None)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.cam = env.scene[p["sensor_cfg"].name]
        n, dev = env.num_envs, env.device
        h, w = self.cam.cfg.height, self.cam.cfg.width
        self.current = torch.zeros(n, h, w, device=dev)
        self.pending = torch.zeros(n, h, w, device=dev)
        self.release_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.blocked_until_render = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.fresh = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_render_id = -1
        period = env.cfg.depth_period_steps
        d_lo = p["delay_range"][0] - p["inherent_delay"]
        d_hi = p["delay_range"][1] - p["inherent_delay"]
        if d_lo < 0 or d_hi >= period:
            raise ValueError(f"추가 지연 [{d_lo}, {d_hi}] 는 [0, {period}) 안이어야 한다")
        self.d_add = (d_lo, d_hi)
        env.jepa_depth_term = self  # depth_fresh 관측이 읽는다

    def _render_id(self) -> int:
        return self._env._sim_step_counter // self._env.cfg.sim.render_interval

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.current[ids] = 0.0
        self.release_at[ids] = -1
        self.fresh[ids] = False
        self.blocked_until_render[ids] = self._render_id()  # 이미 찍힌 렌더는 reset 전 장면

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, fx: float, near: float, far: float,
                 delay_range: tuple[int, int], inherent_delay: int, noise: dict | None) -> torch.Tensor:
        step = env.common_step_counter
        rid = self._render_id()
        if rid != self.last_render_id:
            self.last_render_id = rid
            raw = self.cam.data.output["distance_to_image_plane"][..., 0]
            frame = depth_noise(raw, fx, **noise) if noise is not None else raw
            take = self.blocked_until_render < rid
            d = torch.randint(self.d_add[0], self.d_add[1] + 1, (env.num_envs,), device=env.device)
            self.pending[take] = frame[take]
            self.release_at[take] = step + d[take]
        self.fresh = self.release_at == step
        self.current[self.fresh] = self.pending[self.fresh]
        # fp16: rollout 저장·trajectory 패딩 메모리 절반 ([0,1] 범위라 정밀도 충분). 모델이 float32 로 올린다.
        return to_two_channel(self.current, near, far).half()


def depth_fresh(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[N, 1] — 이번 스텝에 새 depth 프레임이 공개되었으면 1."""
    return env.jepa_depth_term.fresh.float().unsqueeze(-1)

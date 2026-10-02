"""학습 시작·재개 시 설정 일관성 검사와 전역 스텝 복원. Isaac 앱 없이 import 가능."""

from __future__ import annotations


def check_command_schedule(command_cfg, num_steps_per_env: int) -> None:
    """명령 스케줄의 steps_per_iteration 이 PPO rollout 길이와 다르면 오류.

    스케줄은 전역 제어 스텝 / steps_per_iteration 으로 학습 iteration 을 환산하므로, 두 값이 어긋나면
    명령 범위 확장 속도가 조용히 바뀐다. 스케줄이 없는 명령(steps_per_iteration 없음)은 통과.
    """
    steps = getattr(command_cfg, "steps_per_iteration", None)
    if steps is not None and steps != num_steps_per_env:
        raise ValueError(
            f"명령 스케줄 steps_per_iteration={steps} 가 PPO num_steps_per_env={num_steps_per_env} 와 다르다."
            " 두 값을 맞춰라."
        )


def restore_global_step(env, iteration: int, steps_per_iteration: int) -> int:
    """resume 시 env.common_step_counter 를 checkpoint iteration 에 맞춰 복원한다.

    rsl_rl 은 학습 iteration 마다 env.step 을 정확히 num_steps_per_env 번 호출하므로
    iteration k 끝의 전역 스텝은 k × num_steps_per_env 다. 복원하지 않으면 0 으로 돌아가
    iteration 기반 스케줄(명령 범위 등)이 처음부터 다시 시작한다.
    """
    if iteration < 0 or steps_per_iteration <= 0:
        raise ValueError(f"iteration={iteration}, steps_per_iteration={steps_per_iteration}")
    env.common_step_counter = iteration * steps_per_iteration
    return env.common_step_counter

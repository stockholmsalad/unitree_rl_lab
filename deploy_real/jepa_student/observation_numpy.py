"""Go2 LowState to the 45 policy observations used by OracleCurrent."""
import numpy as np


def projected_gravity(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float32)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    # World down vector rotated into body coordinates (R(q)^T @ [0,0,-1]).
    return np.array([2 * (w * y - x * z), -2 * (w * x + y * z),
                     -(1 - 2 * (x * x + y * y))], dtype=np.float32)


def sdk_to_isaac(values, joint_ids_map):
    return np.asarray(values, dtype=np.float32)[np.asarray(joint_ids_map, dtype=np.int64)]


def isaac_to_sdk(values, joint_ids_map):
    result = np.zeros(12, dtype=np.float32)
    result[np.asarray(joint_ids_map, dtype=np.int64)] = values
    return result


def assemble_observation(gyro, quat_wxyz, command, q_sdk, dq_sdk, last_action, cfg):
    q = sdk_to_isaac(q_sdk, cfg["joint_ids_map"])
    dq = sdk_to_isaac(dq_sdk, cfg["joint_ids_map"])
    p = np.concatenate((np.asarray(gyro, dtype=np.float32) * cfg["ang_vel_scale"],
                        projected_gravity(quat_wxyz), np.asarray(command, dtype=np.float32),
                        q - np.asarray(cfg["default_joint_pos"], dtype=np.float32),
                        dq * cfg["joint_vel_scale"], np.asarray(last_action, dtype=np.float32)))
    if p.shape != (45,) or not np.isfinite(p).all():
        raise ValueError("정책 관측은 유한한 45차원이어야 한다")
    return np.clip(p, -cfg["observation_clip"], cfg["observation_clip"]).astype(np.float32)

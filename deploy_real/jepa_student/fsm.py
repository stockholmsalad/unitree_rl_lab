"""Testable Go2 student state and automatic Passive transitions."""
import math

import numpy as np


class SafetyFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "Passive"
        self.stand_started = None
        self.stand_start_q = None
        self.prev_target = None
        self.last_keys = 0

    def update(self, now, keys, gravity_z, depth_age, lowstate_age, q_isaac, target=None):
        buttons = self.cfg["buttons"]
        rising = keys & ~self.last_keys
        self.last_keys = keys
        stop = self.cfg["safety"]
        limits = np.asarray(self.cfg["joint_limits"], dtype=np.float32)
        margin = stop["joint_limit_margin_rad"]
        joint_unsafe = np.any(q_isaac <= limits[:, 0] + margin) or np.any(q_isaac >= limits[:, 1] - margin)
        target_unsafe = (target is not None and
                         (not np.isfinite(target).all() or np.any(target <= limits[:, 0] + margin) or
                          np.any(target >= limits[:, 1] - margin)))
        jump = target is not None and self.prev_target is not None and np.max(np.abs(target - self.prev_target)) > stop["max_target_delta_rad"]
        unsafe = (gravity_z > -math.cos(math.radians(stop["max_tilt_deg"])) or
                  depth_age >= stop["depth_timeout_s"] or lowstate_age >= stop["lowstate_timeout_s"] or
                  joint_unsafe or target_unsafe or jump)
        if rising & (1 << buttons["passive_bit"]) or unsafe:
            self.state = "Passive"
            self.prev_target = None
            return self.state
        if self.state == "Passive" and rising & (1 << buttons["stand_bit"]):
            self.state = "FixStand"
            self.stand_started = now
            self.stand_start_q = q_isaac.copy()
        elif self.state == "FixStand" and now - self.stand_started >= self.cfg["stand_duration_s"] and rising & (1 << buttons["policy_bit"]):
            self.state = "Policy"
            self.prev_target = None
        if self.state == "Policy" and target is not None:
            self.prev_target = target.copy()
        return self.state


def should_publish(live, state):
    return bool(live and state in ("Passive", "FixStand", "Policy"))

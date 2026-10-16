"""Compare a recorded IsaacLab student rollout against the NumPy policy."""
import argparse

import numpy as np

try:
    from .student_numpy import StudentNumpy
except ImportError:
    from student_numpy import StudentNumpy


def replay(prefix, recording):
    policy = StudentNumpy(prefix)
    with np.load(recording) as data:
        count = min(data["policy"].shape[1], 4)
        errors = []
        for env_id in range(count):
            policy.reset()
            for step in range(data["policy"].shape[0]):
                action = policy.step(data["depth"][step, env_id], data["policy"][step, env_id],
                                     bool(data["depth_fresh"][step, env_id, 0] > .5))
                errors.append(np.max(np.abs(action - data["action"][step, env_id])))
                if data["reset"][step, env_id]:
                    policy.reset()
    maximum = float(np.max(errors))
    print("ROLL_OUT_PARITY max_action_abs_error={:.8g} steps={}".format(maximum, len(errors)))
    if maximum > 1e-4:
        raise AssertionError("NumPy rollout error exceeds 1e-4")
    return maximum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--recording", required=True)
    args = parser.parse_args()
    replay(args.model_prefix, args.recording)

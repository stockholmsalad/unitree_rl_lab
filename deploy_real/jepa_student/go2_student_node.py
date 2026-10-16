"""Go2 depth student ROS2 node. Dry-run is the default; --live publishes LowCmd."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .depth_preproc_numpy import preprocess_mm
    from .fsm import SafetyFSM
    from .motor_crc import motor_crc
    from .observation_numpy import assemble_observation, isaac_to_sdk, sdk_to_isaac
    from .student_numpy import StudentNumpy
except ImportError:  # direct script invocation on Orin
    from depth_preproc_numpy import preprocess_mm
    from fsm import SafetyFSM
    from motor_crc import motor_crc
    from observation_numpy import assemble_observation, isaac_to_sdk, sdk_to_isaac
    from student_numpy import StudentNumpy


def joystick_command(msg, cfg):
    limits = cfg["command"]
    vx = np.clip(max(float(msg.ly), 0.0) * limits["vx_max"], limits["vx_min"], limits["vx_max"])
    yaw = np.clip(float(msg.rx) * limits["yaw_max"], limits["yaw_min"], limits["yaw_max"])
    return np.array((vx, limits["vy"], yaw), dtype=np.float32)


class StudentController:
    """ROS-independent control calculation; unit tests use synthetic messages."""
    def __init__(self, prefix, cfg, live=False):
        self.cfg, self.live = cfg, live
        self.policy = StudentNumpy(prefix)
        self.fsm = SafetyFSM(cfg)
        self.tick = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_depth_stamp = None
        self.depth_image = None
        self.depth_frame_id = 0
        self.consumed_frame_id = -1
        self.latest_lowstate = None
        self.last_lowstate_stamp = None
        self.keys = 0
        self.command = np.zeros(3, dtype=np.float32)
        self.camera_info_ok = False

    def receive_depth(self, image, now):
        expected = self.cfg["camera"]
        if tuple(np.shape(image)) != (expected["height"], expected["width"]):
            raise ValueError("D435i depth 크기가 설정과 다르다: {}".format(np.shape(image)))
        self.depth_image = np.asarray(image, dtype=np.uint16).copy()
        self.last_depth_stamp = now
        self.depth_frame_id += 1

    def receive_lowstate(self, msg, now):
        self.latest_lowstate = msg
        self.last_lowstate_stamp = now

    def receive_joystick(self, msg):
        self.keys = int(msg.keys)
        self.command = joystick_command(msg, self.cfg)

    def receive_camera_info(self, msg):
        expected = self.cfg["camera"]
        measured = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        requested = (expected["fx"], expected["fy"], expected["cx"], expected["cy"])
        self.camera_info_ok = (msg.width == expected["width"] and msg.height == expected["height"] and
                               all(abs(a - b) <= expected["intrinsics_tolerance_px"]
                                   for a, b in zip(measured, requested)))
        return self.camera_info_ok

    def step(self, now):
        cfg = self.cfg
        start = time.monotonic()
        if self.latest_lowstate is None:
            return None
        low = self.latest_lowstate
        q_sdk = np.array([low.motor_state[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([low.motor_state[i].dq for i in range(12)], dtype=np.float32)
        q_isaac = sdk_to_isaac(q_sdk, cfg["joint_ids_map"])
        depth_age = float("inf") if self.last_depth_stamp is None else now - self.last_depth_stamp
        if self.live and not self.camera_info_ok:
            depth_age = float("inf")
        low_age = now - self.last_lowstate_stamp
        quat = np.asarray(low.imu_state.quaternion, dtype=np.float32)
        # Unitree quaternion order in SDK examples is [w,x,y,z].
        from_observation = assemble_observation(low.imu_state.gyroscope, quat, self.command,
                                                q_sdk, dq_sdk, self.last_action, cfg)
        old_state = self.fsm.state
        target = None
        if old_state == "Policy":
            fresh = (self.tick % cfg["depth_period_steps"] == 0 and
                     self.depth_frame_id != self.consumed_frame_id and self.depth_image is not None)
            depth = (preprocess_mm(self.depth_image, cfg["depth_hw"], cfg["depth_near_m"], cfg["depth_far_m"])
                     if fresh else None)
            if fresh:
                self.consumed_frame_id = self.depth_frame_id
            action = self.policy.step(depth, from_observation, fresh)
            target = np.asarray(cfg["default_joint_pos"], dtype=np.float32) + cfg["action_scale"] * action
        else:
            fresh = False
            action = np.zeros(12, dtype=np.float32)
        gravity_z = from_observation[5]
        state = self.fsm.update(now, self.keys, gravity_z, depth_age, low_age, q_isaac, target)
        if state != old_state:
            self.policy.reset()
            self.last_action.fill(0)
            if state == "Policy":
                self.tick = -1
                self.consumed_frame_id = -1
            if state == "Passive":
                action = np.zeros(12, dtype=np.float32)
                target = None
        if state == "Policy" and target is not None:
            self.last_action = action.copy()
        self.tick += 1
        return {"state": state, "obs": from_observation, "latent": self.policy.last_z[0].copy(),
                "action": action, "target": target, "command": self.command.copy(),
                "depth_stamp": self.last_depth_stamp if self.last_depth_stamp is not None else np.nan,
                "fresh": fresh, "latency_s": time.monotonic() - start, "q_sdk": q_sdk,
                "q_isaac": q_isaac}

    def motor_targets(self, result, now):
        state = result["state"]
        cfg = self.cfg
        if state == "Passive":
            return None, 0.0, cfg["passive_kd"]
        if state == "FixStand":
            fraction = np.clip((now - self.fsm.stand_started) / cfg["stand_duration_s"], 0.0, 1.0)
            middle = sdk_to_isaac(cfg["stand_middle_q_sdk"], cfg["joint_ids_map"])
            final = sdk_to_isaac(cfg["stand_final_q_sdk"], cfg["joint_ids_map"])
            if fraction < 0.5:
                q = (1 - 2 * fraction) * self.fsm.stand_start_q + 2 * fraction * middle
            else:
                q = (2 - 2 * fraction) * middle + (2 * fraction - 1) * final
            return isaac_to_sdk(q, cfg["joint_ids_map"]), cfg["stand_kp_sdk"], cfg["stand_kd_sdk"]
        if result["target"] is None:
            return None, 0.0, cfg["passive_kd"]
        return isaac_to_sdk(result["target"], cfg["joint_ids_map"]), cfg["policy_kp"], cfg["policy_kd"]


def build_lowcmd(result, controller, now, LowCmd):
    msg = LowCmd()
    msg.head = [0xfe, 0xef]
    msg.level_flag = 0xff
    positions, kp, kd = controller.motor_targets(result, now)
    for i in range(len(msg.motor_cmd)):
        motor = msg.motor_cmd[i]
        motor.mode = 0x01
        active_joint = i < 12 and positions is not None
        motor.q = float(positions[i]) if active_joint else 2.146e9
        motor.dq = 0.0 if active_joint else 16000.0
        motor.tau = 0.0
        motor.kp = float(kp[i] if isinstance(kp, list) else kp) if active_joint else 0.0
        motor.kd = float(kd[i] if isinstance(kd, list) else kd) if i < 12 else 0.0
    msg.crc = motor_crc(msg)
    return msg


def write_log(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("obs", "latent", "action", "command", "state", "depth_stamp", "fresh", "latency_s")
    np.savez(str(path), **{key: np.asarray([row[key] for row in rows]) for key in keys})


def offline_replay(controller, directory, output, steps=150):
    """Z790-only smoke using recorded PNGs and synthetic LowState, without ROS or commands."""
    from types import SimpleNamespace
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("오프라인 PNG 재생에는 Z790의 Pillow가 필요하다; Orin 노드에는 필요 없다") from exc
    paths = sorted(Path(directory).glob("*.png"))
    if not paths:
        raise FileNotFoundError("depth PNG가 없다: {}".format(directory))
    cfg = controller.cfg
    default_sdk = isaac_to_sdk(np.asarray(cfg["default_joint_pos"], dtype=np.float32), cfg["joint_ids_map"])
    motors = [SimpleNamespace(q=float(q), dq=0.0) for q in default_sdk]
    low = SimpleNamespace(motor_state=motors, imu_state=SimpleNamespace(
        gyroscope=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]))
    joy = SimpleNamespace(ly=0.0, rx=0.0, keys=0)
    rows = []
    start = time.monotonic()
    for i in range(steps):
        now = start + i / cfg["control_hz"]
        if i % cfg["depth_period_steps"] == 0:
            frame = np.asarray(Image.open(str(paths[(i // cfg["depth_period_steps"]) % len(paths)])), dtype=np.uint16)
            controller.receive_depth(frame, now)
        controller.receive_lowstate(low, now)
        joy.keys = ((1 << cfg["buttons"]["stand_bit"]) if i == 1 else
                    (1 << cfg["buttons"]["policy_bit"]) if i == int(cfg["stand_duration_s"] * cfg["control_hz"]) + 2 else 0)
        controller.receive_joystick(joy)
        result = controller.step(now)
        rows.append(result)
        deadline = start + (i + 1) / cfg["control_hz"]
        time.sleep(max(0.0, deadline - time.monotonic()))
    write_log(output, rows)
    latencies = np.asarray([row["latency_s"] for row in rows])
    print("OFFLINE_DRY_RUN steps={} mean={:.4f}s p99={:.4f}s overruns={}".format(
        steps, float(np.mean(latencies)), float(np.percentile(latencies, 99)),
        int(np.count_nonzero(latencies > 1.0 / cfg["control_hz"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--log", default="student_dry_run.npz")
    parser.add_argument("--dry_run", action="store_true", help="기본 동작. /lowcmd를 발행하지 않는다")
    parser.add_argument("--live", action="store_true", help="/lowcmd 발행; 매단 로봇에서만 사용")
    parser.add_argument("--sport_mode_released", action="store_true", help="sport mode 해제를 수동 확인했음을 명시")
    parser.add_argument("--offline_replay_dir", help="Z790 녹화 PNG + 가짜 LowState의 dry-run 재생")
    parser.add_argument("--offline_steps", type=int, default=150)
    args = parser.parse_args()
    if args.live and args.dry_run:
        parser.error("--dry_run과 --live를 함께 사용할 수 없다")
    if args.live and not args.sport_mode_released:
        parser.error("--live에는 확인된 sport mode 해제 후 --sport_mode_released가 필요하다")
    cfg = json.loads(Path(args.config).read_text())  # JSON syntax is valid YAML; no PyYAML on Orin.
    controller = StudentController(args.model_prefix, cfg, live=args.live)
    if args.offline_replay_dir:
        if args.live:
            parser.error("오프라인 재생은 --live를 지원하지 않는다")
        offline_replay(controller, args.offline_replay_dir, args.log, args.offline_steps)
        return
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from unitree_go.msg import LowCmd, LowState, WirelessController
    except ImportError as exc:
        raise RuntimeError("ROS2 Foxy 또는 unitree_go Python msg import 실패. unitree_ros2/cyclonedds_ws를 source했는지 확인") from exc
    rclpy.init()

    class NodeImpl(Node):
        def __init__(self):
            super().__init__("jepa_student")
            self.controller = controller
            self.rows = []
            self.log_chunk = 0
            self.latencies = []
            self.publisher = self.create_publisher(LowCmd, "/lowcmd", 10) if args.live else None
            self.create_subscription(LowState, "/lowstate", self.low_cb, 10)
            self.create_subscription(WirelessController, "/wirelesscontroller", self.joy_cb, 10)
            self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_cb, 2)
            self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.camera_info_cb, 2)
            self.create_timer(1.0 / cfg["control_hz"], self.tick)

        def low_cb(self, msg):
            controller.receive_lowstate(msg, time.monotonic())

        def joy_cb(self, msg):
            controller.receive_joystick(msg)

        def depth_cb(self, msg):
            if msg.encoding != "16UC1" or msg.is_bigendian:
                self.get_logger().error("depth는 little-endian 16UC1이어야 한다")
                return
            pixels = np.frombuffer(msg.data, dtype="<u2").reshape(msg.height, msg.step // 2)[:, :msg.width]
            controller.receive_depth(pixels, time.monotonic())

        def camera_info_cb(self, msg):
            if not controller.receive_camera_info(msg):
                self.get_logger().error("camera_info intrinsic 또는 해상도가 config.yaml과 다르다; live는 Passive 유지")

        def tick(self):
            now = time.monotonic()
            result = controller.step(now)
            if result is None:
                return
            if self.publisher is not None:
                self.publisher.publish(build_lowcmd(result, controller, now, LowCmd))
            self.rows.append(result)
            self.latencies.append(result["latency_s"])
            if len(self.rows) >= cfg["log_flush_steps"]:
                self.flush_log()
                p99 = np.percentile(self.latencies, 99)
                overruns = int(np.count_nonzero(np.asarray(self.latencies) > 1.0 / cfg["control_hz"]))
                self.get_logger().info("loop mean={:.4f}s p99={:.4f}s overrun={} / {}".format(
                    float(np.mean(self.latencies)), p99, overruns, len(self.latencies)))

        def flush_log(self):
            if not self.rows:
                return
            path = Path(args.log)
            chunk_path = path.with_name("{}_{:06d}{}".format(path.stem, self.log_chunk, path.suffix))
            write_log(chunk_path, self.rows)
            self.rows.clear()
            self.log_chunk += 1

    node = NodeImpl()
    try:
        rclpy.spin(node)
    finally:
        node.flush_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

import ast
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deploy_real.jepa_student.depth_preproc_numpy import downsample_nearest, preprocess_mm, to_two_channel
from deploy_real.jepa_student.fsm import SafetyFSM, should_publish
from deploy_real.jepa_student.go2_student_node import StudentController
from deploy_real.jepa_student.motor_crc import motor_crc, pack_lowcmd_for_crc
from deploy_real.jepa_student.observation_numpy import assemble_observation, isaac_to_sdk, sdk_to_isaac
from deploy_real.jepa_student.student_numpy import StudentNumpy


ROOT = Path(__file__).resolve().parents[3]
CFG = json.loads((ROOT / "deploy_real/jepa_student/config.yaml").read_text())
CHECKPOINT = ROOT / "logs/student/gru_t17000_s43/model_3000.pt"


class PureTests(unittest.TestCase):
    def test_joint_roundtrip_and_observation(self):
        q = np.arange(12, dtype=np.float32) * 0.01
        sdk = isaac_to_sdk(q, CFG["joint_ids_map"])
        np.testing.assert_array_equal(sdk_to_isaac(sdk, CFG["joint_ids_map"]), q)
        obs = assemble_observation([1, 2, 3], [1, 0, 0, 0], [0.6, 0, 0], sdk,
                                   np.zeros(12), np.zeros(12), CFG)
        self.assertEqual(obs.shape, (45,))
        np.testing.assert_allclose(obs[:9], [0.2, 0.4, 0.6, 0, 0, -1, 0.6, 0, 0], atol=1e-6)
        np.testing.assert_allclose(obs[9:21], q - CFG["default_joint_pos"], atol=1e-6)

    def test_teacher_deploy_parameters(self):
        path = ROOT / "logs/rsl_rl/jepa_loco_oracle_current/2026-10-07_17-57-46_oracle_current_easystart_s42/params/deploy.yaml"
        if not path.exists():
            self.skipTest("teacher deploy.yaml unavailable")
        lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "joint_ids_map: {}".format(CFG["joint_ids_map"]))
        self.assertEqual(lines[4], "default_joint_pos: {}".format(CFG["default_joint_pos"]))
        self.assertIn("step_dt: 0.02", lines)

    def test_depth_small(self):
        depth = np.array([[0, .5, 1, 1], [0, .6, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], np.float32)
        out = downsample_nearest(depth, (2, 2))
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.isfinite(to_two_channel(out)).all())
        self.assertEqual(to_two_channel(np.array([[3.0, 0.0]], np.float32))[0, 0, 0], 1.0)

    def test_fsm_stops_and_dry_run(self):
        q = np.asarray(CFG["default_joint_pos"], np.float32)
        fsm = SafetyFSM(CFG)
        self.assertEqual(fsm.update(0, 1 << CFG["buttons"]["stand_bit"], -1, 0, 0, q), "FixStand")
        self.assertEqual(fsm.update(2.1, 1 << CFG["buttons"]["policy_bit"], -1, 0, 0, q), "Policy")
        self.assertEqual(fsm.update(2.2, 1 << CFG["buttons"]["passive_bit"], -1, 0, 0, q), "Passive")
        for key, value in (("gravity_z", 0.0), ("depth_age", .21), ("lowstate_age", .11)):
            fsm.state = "Policy"
            kwargs = dict(gravity_z=-1, depth_age=0, lowstate_age=0)
            kwargs[key] = value
            self.assertEqual(fsm.update(3, 0, q_isaac=q, **kwargs), "Passive")
        fsm.state = "Policy"
        self.assertEqual(fsm.update(3, 0, -1, 0, 0, np.full(12, 99)), "Passive")
        fsm.state = "Policy"
        fsm.prev_target = np.zeros(12)
        self.assertEqual(fsm.update(3, 0, -1, 0, 0, q, np.ones(12)), "Passive")
        fsm.state = "Policy"
        fsm.prev_target = None
        self.assertEqual(fsm.update(3, 0, -1, 0, 0, q, np.full(12, 99)), "Passive")
        self.assertFalse(should_publish(False, "Policy"))

    def test_python_38_syntax(self):
        for path in (ROOT / "deploy_real/jepa_student").glob("*.py"):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))

    def test_crc_layout(self):
        motor = SimpleNamespace(mode=1, q=0.1, dq=0., tau=0., kp=25., kd=.5, reserve=[0, 0, 0])
        msg = SimpleNamespace(head=[0xfe, 0xef], level_flag=0xff, frame_reserve=0,
                              sn=[0, 0], version=[0, 0], bandwidth=0,
                              motor_cmd=[motor] * 20, bms_cmd=SimpleNamespace(off=0, reserve=[0, 0, 0]),
                              wireless_remote=[0] * 40, led=[0] * 12, fan=[0] * 2, gpio=0, reserve=0)
        self.assertEqual(len(pack_lowcmd_for_crc(msg)), 808)
        # Same 812-byte C struct and crc32_core reference from unitree_ros2.
        self.assertEqual(motor_crc(msg), 0xb4a9d931)

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler unavailable")
    def test_crc_against_compiled_unitree_layout(self):
        motor = SimpleNamespace(mode=1, q=0.1, dq=0., tau=0., kp=25., kd=.5, reserve=[0, 0, 0])
        msg = SimpleNamespace(head=[0xfe, 0xef], level_flag=0xff, frame_reserve=0,
                              sn=[0, 0], version=[0, 0], bandwidth=0, motor_cmd=[motor] * 20,
                              bms_cmd=SimpleNamespace(off=0, reserve=[0, 0, 0]),
                              wireless_remote=[0] * 40, led=[0] * 12, fan=[0] * 2, gpio=0, reserve=0)
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "crc_reference"
            raw = Path(directory) / "lowcmd.bin"
            source = Path(__file__).with_name("crc_reference.cpp")
            subprocess.check_call(["g++", "-std=c++11", str(source), "-o", str(binary)])
            size, crc = subprocess.check_output([str(binary), str(raw)], text=True).split()
            self.assertEqual(int(size), 812)
            self.assertEqual(raw.read_bytes(), pack_lowcmd_for_crc(msg))
            self.assertEqual(int(crc, 16), motor_crc(msg))


@unittest.skipUnless(CHECKPOINT.exists(), "Z790 student checkpoint unavailable")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from deploy_real.jepa_student.export_student import export
        from unitree_rl_lab.jepa_loco.models.student_depth import StudentInferencePolicy
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prefix = str(Path(cls.tmp.name) / "model")
        export(CHECKPOINT, cls.prefix)
        cls.numpy_policy = StudentNumpy(cls.prefix)
        cls.torch_policy = StudentInferencePolicy(str(CHECKPOINT), torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_205_step_parity_with_resets(self):
        from tensordict import TensorDict
        rng = np.random.RandomState(3)
        max_error = 0.0
        for i in range(205):
            fresh = i % 5 == 0
            depth = rng.rand(2, 64, 112).astype(np.float32)
            proprio = (rng.randn(45) * .1).astype(np.float32)
            n = self.numpy_policy.step(depth, proprio, fresh)
            obs = TensorDict({"depth": self.torch.from_numpy(depth[None]),
                              "policy": self.torch.from_numpy(proprio[None]),
                              "depth_fresh": self.torch.tensor([[float(fresh)]])}, batch_size=[1])
            with self.torch.no_grad():
                t = self.torch_policy(obs).numpy()[0]
            max_error = max(max_error, float(np.max(np.abs(n - t))))
            if i in (63, 157):
                self.numpy_policy.reset()
                self.torch_policy.reset(self.torch.tensor([True]))
        self.assertLessEqual(max_error, 1e-4)

    def test_dry_run_controller(self):
        controller = StudentController(self.prefix, CFG, live=False)
        sdk = isaac_to_sdk(np.asarray(CFG["default_joint_pos"], np.float32), CFG["joint_ids_map"])
        low = SimpleNamespace(motor_state=[SimpleNamespace(q=float(q), dq=0.) for q in sdk],
                              imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]))
        controller.receive_lowstate(low, 0.)
        controller.receive_depth(np.full((480, 848), 1000, np.uint16), 0.)
        result = controller.step(0.)
        self.assertEqual(result["state"], "Passive")
        self.assertEqual(result["obs"].shape, (45,))
        info = CFG["camera"]
        camera = SimpleNamespace(width=info["width"], height=info["height"],
                                 k=[info["fx"], 0, info["cx"], 0, info["fy"], info["cy"], 0, 0, 1])
        self.assertTrue(controller.receive_camera_info(camera))
        camera.k[0] += 10
        self.assertFalse(controller.receive_camera_info(camera))

    def test_recorded_depth_preprocessing_matches_torch(self):
        from PIL import Image
        from unitree_rl_lab.jepa_loco.sensors.depth_proc import downsample_real, to_two_channel as torch_two
        files = sorted((ROOT / "results/jepa_loco/real_depth/flat_stand").glob("*.png"))[:3]
        if not files:
            self.skipTest("flat_stand recording unavailable")
        for path in files:
            mm = np.asarray(Image.open(str(path)), dtype=np.uint16)
            numpy_result = preprocess_mm(mm)
            depth = self.torch.from_numpy(mm.astype(np.float32) * .001)
            torch_result = torch_two(downsample_real(depth, (64, 112), mode="nearest", near=.28)).numpy()
            np.testing.assert_allclose(numpy_result, torch_result, atol=1e-6, rtol=0)

    def test_npz_replay_contract(self):
        from tensordict import TensorDict
        from unitree_rl_lab.jepa_loco.models.student_depth import StudentInferencePolicy
        from deploy_real.jepa_student.replay_rollout import replay
        model = StudentInferencePolicy(str(CHECKPOINT), self.torch.device("cpu"))
        rng = np.random.RandomState(9)
        rows = {key: [] for key in ("policy", "depth", "depth_fresh", "reset", "action")}
        for step in range(12):
            depth = rng.rand(1, 2, 64, 112).astype(np.float32)
            proprio = (rng.randn(1, 45) * .1).astype(np.float32)
            fresh = np.asarray([[float(step % 5 == 0)]], dtype=np.float32)
            obs = TensorDict({"depth": self.torch.from_numpy(depth),
                              "policy": self.torch.from_numpy(proprio),
                              "depth_fresh": self.torch.from_numpy(fresh)}, batch_size=[1])
            with self.torch.no_grad():
                action = model(obs).numpy()
            done = np.asarray([step == 6])
            for key, value in (("policy", proprio), ("depth", depth), ("depth_fresh", fresh),
                               ("reset", done), ("action", action)):
                rows[key].append(value)
            model.reset(self.torch.from_numpy(done))
        path = Path(self.tmp.name) / "replay.npz"
        np.savez(str(path), **{key: np.stack(value) for key, value in rows.items()})
        self.assertLessEqual(replay(self.prefix, path), 1e-4)


if __name__ == "__main__":
    unittest.main()

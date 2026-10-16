# Go2 Orin NX depth student

첫 실기 시험은 **평지와 9~12 cm 계단**으로 제한한다. 14·15 cm는 실측 카메라 위치를 적용한 시뮬 평가에서 낙상 4~8%가 관찰되어 제외한다. [카메라 민감도 기록](../../results/jepa_loco/camera_mount/eval/summary.md)을 먼저 확인한다. 실측 roll 3.3°와 장착 위치 차이는 이 정책의 학습에 반영되지 않았다. 이번 배포는 재학습 없이 모델을 시험한다.

Orin 환경은 JetPack 5.1.1, Python 3.8.10, NumPy 1.17.4를 기준으로 한다. 추론 파일은 torch와 onnxruntime을 import하지 않는다. `config.yaml`은 JSON 문법의 YAML이어서 Python 표준 `json`만으로 읽는다. 카메라 depth는 16UC1 mm, 848×480 입력이다. `depth_proc.downsample_real(..., out_hw=(64,112), mode="nearest", near=0.28)`와 같은 footprint 중심의 유효 픽셀 선택을 쓴 다음 `to_two_channel(near=0.28, far=2.0)`을 적용한다. 시뮬 결측 띠와 노이즈는 실기에 더하지 않는다.

## 명령

Z790에서 checkpoint를 export한다.

```bash
# ★ Z790
conda activate env_test && python deploy_real/jepa_student/export_student.py --checkpoint logs/student/gru_t17000_s43/model_3000.pt --output_prefix deploy_real/jepa_student/gru_t17000_s43
```

Orin으로 이 디렉터리의 Python/config 파일과 export 두 파일을 복사한다. 다음 명령의 `go2` 호스트명은 실제 SSH 주소로 바꾼다.

```bash
# ★ Z790
scp -r deploy_real/jepa_student go2:~/jepa_student
```

Orin에서 ROS2 Foxy와 `~/unitree_ros2`, `~/cyclonedds_ws`를 source한 셸에서 먼저 메시 import와 dry-run을 확인한다. `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`를 사용한다. `unitree_go` 메시가 import되지 않으면 노드는 명확한 오류로 종료한다. `/camera/depth/camera_info`의 해상도와 intrinsic이 `config.yaml`의 실측값과 다르면 live는 Passive에 머문다. `unitree_api`는 자동 sport mode 전환을 구현하지 않았으므로 사용하지 않는다.

```bash
# ★ Go2 Orin NX
source /opt/ros/foxy/setup.bash && source ~/cyclonedds_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && python3 ~/jepa_student/go2_student_node.py --model_prefix ~/jepa_student/gru_t17000_s43 --log ~/jepa_student/dry_run.npz
```

ROS2 sport mode 자동 해제 서비스의 요청 스키마는 설치 환경에서 확인되지 않았다. `--live` 전에 Unitree의 기존 도구에서 sport mode를 끄고 low-level 제어권이 해제된 것을 확인한다. 로봇을 매단 상태에서 `--sport_mode_released`를 명시해 FixStand와 정책을 시험한다. 확인 없이 플래그를 넣지 않는다.

```bash
# ★ Go2 Orin NX
source /opt/ros/foxy/setup.bash && source ~/cyclonedds_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && python3 ~/jepa_student/go2_student_node.py --model_prefix ~/jepa_student/gru_t17000_s43 --log ~/jepa_student/suspended_live.npz --live --sport_mode_released
```

그다음 같은 명령으로 평지 저속을 확인하고, 마지막으로 9~12 cm 계단을 시험한다. 조이스틱 왼쪽 세로축은 전진 0~0.6 m/s, 오른쪽 가로축은 yaw ±0.5 rad/s다. B 버튼 하나로 어느 상태에서든 Passive, A로 FixStand, 2초 뒤 Start로 Policy에 들어간다. Passive에서 Policy 직접 전환은 없다. 모든 버튼·gain·안전 한계는 `config.yaml`에 있다.

매 제어 스텝의 관측, latent, action, 명령, FSM, depth 수신 시각, 루프 지연이 `.npz`에 저장된다. `log_flush_steps`마다 `_000000.npz` 형식의 청크를 저장하고 종료 시 남은 청크를 저장한다. 루프 지연 평균·p99·20 ms 초과 횟수를 출력한다. `/lowcmd`는 `--live`가 없으면 publisher 자체를 만들지 않는다.

## 사전 점검과 재생

```bash
# ★ Z790
conda activate env_test && python -m unittest deploy_real.jepa_student.tests.test_deployment -v
```

```bash
# ★ Z790
conda activate env_test && python deploy_real/jepa_student/go2_student_node.py --model_prefix deploy_real/jepa_student/gru_t17000_s43 --offline_replay_dir results/jepa_loco/real_depth/flat_stand --offline_steps 150 --log /tmp/jepa_student_offline.npz
```

시뮬 관측 기록은 `eval_student_fixed.py --record_rollout /tmp/rollout.npz`로 만든다. 기록을 `replay_rollout.py`로 재생하면 NumPy와 PyTorch 행동의 최대 오차를 확인한다.

```bash
# ★ Z790
conda activate env_test && python deploy_real/jepa_student/replay_rollout.py --model_prefix deploy_real/jepa_student/gru_t17000_s43 --recording /tmp/rollout.npz
```

CRC는 [unitree_ros2 원본 구조체](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/include/common/motor_crc.h)와 [motor_crc.cpp](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/common/motor_crc.cpp)를 기준으로 한다. `unitree_go` Python 메시의 실제 import 가능 여부와 sport mode 해제 절차는 Orin에서 다시 확인해야 한다.

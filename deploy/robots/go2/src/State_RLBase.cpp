#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <unordered_map>

namespace isaaclab
{

REGISTER_OBSERVATION(hybrid_velocity_commands)
{
    std::vector<float> obs = {0.0f, 0.0f, 0.0f};

    auto& joystick = env->robot->data.joystick;
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    const float min_vx = cfg["lin_vel_x"][0].as<float>();
    const float max_vx = cfg["lin_vel_x"][1].as<float>();

    const float min_vy = cfg["lin_vel_y"][0].as<float>();
    const float max_vy = cfg["lin_vel_y"][1].as<float>();

    const float min_wz = cfg["ang_vel_z"][0].as<float>();
    const float max_wz = cfg["ang_vel_z"][1].as<float>();

    // ---------------------------------------------------------
    // Keyboard input
    // ---------------------------------------------------------
    std::string key;

    if (FSMState::keyboard)
    {
        key = FSMState::keyboard->key();
    }

    static const std::unordered_map<std::string, std::vector<float>>
        keyboard_commands = {
            {"w", { 1.0f,  0.0f,  0.0f}},
            {"s", {-1.0f,  0.0f,  0.0f}},
            {"a", { 0.0f,  1.0f,  0.0f}},
            {"d", { 0.0f, -1.0f,  0.0f}},
            {"q", { 0.0f,  0.0f,  1.0f}},
            {"e", { 0.0f,  0.0f, -1.0f}},
            {" ", { 0.0f,  0.0f,  0.0f}},
        };

    auto keyboard_it = keyboard_commands.find(key);

    // 키보드 이동키가 현재 입력 중이면 키보드 우선
    if (keyboard_it != keyboard_commands.end())
    {
        const auto& cmd = keyboard_it->second;

        obs[0] = (cmd[0] >= 0.0f) ? cmd[0] * max_vx
                                  : -cmd[0] * min_vx;

        obs[1] = (cmd[1] >= 0.0f) ? cmd[1] * max_vy
                                  : -cmd[1] * min_vy;

        obs[2] = (cmd[2] >= 0.0f) ? cmd[2] * max_wz
                                  : -cmd[2] * min_wz;
    }
    else
    {
        // -----------------------------------------------------
        // No keyboard motion input: use joystick
        // -----------------------------------------------------
        obs[0] = std::clamp(
            joystick->ly(),
            min_vx,
            max_vx
        );

        obs[1] = std::clamp(
            -joystick->lx(),
            min_vy,
            max_vy
        );

        obs[2] = std::clamp(
            -joystick->rx(),
            min_wz,
            max_wz
        );
    }

    return obs;
}

} // namespace isaaclab


State_RLBase::State_RLBase(
    int state_mode,
    std::string state_string
)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir =
        param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<
            unitree::BaseArticulation<LowState_t::SharedPtr>
        >(FSMState::lowstate)
    );

    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            {
                return isaaclab::mdp::bad_orientation(
                    env.get(),
                    1.0
                );
            },
            FSMStringMap.right.at("Passive")
        )
    );
}


void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();

    for (
        int i = 0;
        i < env->robot->data.joint_ids_map.size();
        i++
    )
    {
        lowcmd->msg_
            .motor_cmd()[env->robot->data.joint_ids_map[i]]
            .q() = action[i];
    }
}
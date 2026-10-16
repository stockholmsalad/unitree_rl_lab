// Standalone layout and CRC check derived from unitree_ros2 motor_crc.h/.cpp.
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>

struct MotorCmd {
    uint8_t mode;
    float q, dq, tau, Kp, Kd;
    std::array<uint32_t, 3> reserve;
};
struct BmsCmd { uint8_t off; std::array<uint8_t, 3> reserve; };
struct LowCmd {
    std::array<uint8_t, 2> head;
    uint8_t levelFlag, frameReserve;
    std::array<uint32_t, 2> SN, version;
    uint16_t bandWidth;
    std::array<MotorCmd, 20> motorCmd;
    BmsCmd bms;
    std::array<uint8_t, 40> wirelessRemote;
    std::array<uint8_t, 12> led;
    std::array<uint8_t, 2> fan;
    uint8_t gpio;
    uint32_t reserve, crc;
};

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    uint32_t crc = 0xffffffff;
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t data = ptr[i], xbit = 1u << 31;
        for (int bit = 0; bit < 32; ++bit) {
            if (crc & 0x80000000) { crc <<= 1; crc ^= 0x04c11db7; }
            else crc <<= 1;
            if (data & xbit) crc ^= 0x04c11db7;
            xbit >>= 1;
        }
    }
    return crc;
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    LowCmd command{};
    command.head = {0xfe, 0xef};
    command.levelFlag = 0xff;
    for (auto& motor : command.motorCmd) {
        motor.mode = 1; motor.q = 0.1f; motor.Kp = 25.f; motor.Kd = 0.5f;
    }
    FILE* file = std::fopen(argv[1], "wb");
    if (!file) return 2;
    std::fwrite(&command, 1, sizeof(LowCmd) - 4, file);
    std::fclose(file);
    std::cout << sizeof(LowCmd) << " " << std::hex
              << crc32_core(reinterpret_cast<uint32_t*>(&command), (sizeof(LowCmd) >> 2) - 1);
}

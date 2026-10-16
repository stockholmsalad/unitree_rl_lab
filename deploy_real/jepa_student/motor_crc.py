"""Go2 LowCmd CRC matching unitree_ros2 example common/motor_crc.cpp."""
import struct


def crc32_words(data):
    if len(data) % 4:
        raise ValueError("CRC 입력은 4바이트 단위여야 한다")
    crc = 0xffffffff
    for offset in range(0, len(data), 4):
        word = struct.unpack_from("<I", data, offset)[0]
        for bit in range(32):
            high = crc & 0x80000000
            crc = (crc << 1) & 0xffffffff
            if bool(high) ^ bool(word & (0x80000000 >> bit)):
                crc ^= 0x04c11db7
    return crc


def pack_lowcmd_for_crc(msg):
    """Pack padded C LowCmd through reserve, excluding the final CRC word."""
    b = bytearray()
    b += struct.pack("<2B2B", *msg.head, msg.level_flag, msg.frame_reserve)
    b += struct.pack("<2I2IH", *msg.sn, *msg.version, msg.bandwidth)
    b += b"\x00\x00"  # C alignment before MotorCmd[20]
    for motor in msg.motor_cmd:
        b += struct.pack("<B3x5f3I", motor.mode, motor.q, motor.dq, motor.tau,
                         motor.kp, motor.kd, *motor.reserve)
    b += struct.pack("<4B", msg.bms_cmd.off, *msg.bms_cmd.reserve)
    b += struct.pack("<40B", *msg.wireless_remote)
    b += struct.pack("<12B", *msg.led)
    b += struct.pack("<2B", *msg.fan)
    b += struct.pack("<B1xI", msg.gpio, msg.reserve)
    return bytes(b)


def motor_crc(msg):
    return crc32_words(pack_lowcmd_for_crc(msg))

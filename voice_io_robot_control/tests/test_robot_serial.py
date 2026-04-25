"""Manual robot serial test. Run this before running main.py."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ROBOT_PORT, ROBOT_BAUD, SERIAL_TIMEOUT
from robot_action_reference import RobotSerialController


def main():
    robot = RobotSerialController(ROBOT_PORT, ROBOT_BAUD, SERIAL_TIMEOUT)
    robot.set_volume_min()

    print("Type action ID or action name. Examples: 8, forward, 6, handshake")
    print("Type q to quit.")

    try:
        while True:
            cmd = input("Command ID or word > ").strip().lower()
            if cmd in ["q", "quit", "exit"]:
                break
            robot.send_action(cmd, wait=True)
            robot.read_available()
    finally:
        robot.close()


if __name__ == "__main__":
    main()

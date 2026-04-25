"""
Robot action reference table.

The robot firmware accepts a number followed by newline through serial, for example:
    8\n
This module keeps all action IDs, names, aliases, and wait times in one place.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import time
import serial


@dataclass(frozen=True)
class RobotAction:
    action_id: int
    name: str
    description: str
    firmware_command: str
    estimated_wait_s: float
    aliases: List[str]


ACTIONS: Dict[int, RobotAction] = {
    0: RobotAction(0, "ready", "Robot replies that it is ready / here.", "@ZAI#$", 1.5, ["ready", "wake", "hello"]),
    1: RobotAction(1, "stand", "Stand at attention.", "@LIZHENG#$", 1.5, ["stand", "attention", "stand up"]),
    2: RobotAction(2, "down", "Get down / lie down.", "@PAXIA#$", 3.0, ["down", "lie down", "get down"]),
    3: RobotAction(3, "butt_up", "Raise buttocks.", "@JUEPIGU#$", 3.0, ["butt up", "raise butt"]),
    4: RobotAction(4, "squat", "Squat down.", "@DUNXIA#$", 2.5, ["squat", "sit"]),
    5: RobotAction(5, "plank", "Do plank pose.", "@PINGBANZHICHENG#$", 3.5, ["plank"]),
    6: RobotAction(6, "handshake", "Shake hand.", "@WOSHOU#$", 3.5, ["handshake", "shake hand"]),
    7: RobotAction(7, "sleep", "Sleep action.", "@SHUIJIAO#$", 2.5, ["sleep", "go to sleep"]),
    8: RobotAction(8, "forward", "Move forward.", "@QIANJIN#$", 4.0, ["forward", "move forward", "go forward"]),
    9: RobotAction(9, "backward", "Move backward.", "@HOUTUI#$", 4.0, ["backward", "retreat", "move back"]),
    10: RobotAction(10, "turn_right", "Turn right.", "@YOUZHUAN#$", 3.0, ["right", "turn right"]),
    11: RobotAction(11, "turn_left", "Turn left.", "@ZUOZHUAN#$", 3.0, ["left", "turn left"]),
    12: RobotAction(12, "swing", "Swing / sway body.", "@YAOBAI#$", 3.0, ["swing", "sway"]),
    13: RobotAction(13, "dig", "Dig/rake soil motion.", "@BATU#$", 3.0, ["dig", "rake"]),
    14: RobotAction(14, "cute", "Cute/coquetry behavior.", "@SAJIAO#$", 3.0, ["cute", "act cute"]),
    15: RobotAction(15, "kick", "Kick legs.", "@GOUDENGTUI#$", 3.5, ["kick", "kick legs"]),
    16: RobotAction(16, "bow", "Bow/kowtow.", "@GUIBAI#$", 3.5, ["bow", "kowtow"]),
    17: RobotAction(17, "push_up", "Do push-ups.", "@ZUOFUWOCHENG#$", 4.0, ["push up", "push-up", "pushup"]),
    18: RobotAction(18, "lazy", "Lazy/slack off action.", "@BAILAN#$", 3.0, ["lazy", "slack off"]),
    19: RobotAction(19, "face_change", "Change facial expression.", "@BIANLIAN#$", 2.0, ["face", "face change", "change face"]),
    20: RobotAction(20, "light_on", "Turn light on.", "@KAIDENG#$", 1.0, ["light on", "turn on light"]),
    21: RobotAction(21, "light_off", "Turn light off.", "@GUANDENG#$", 1.0, ["light off", "turn off light"]),
    36: RobotAction(36, "circle", "Spin/circle motion.", "@ZHUANQUANQUAN#$", 3.5, ["circle", "spin", "turn around"]),
    43: RobotAction(43, "butt_shake", "Butt-shake motion.", "@DOUTUN#$", 3.0, ["butt shake", "shake butt"]),
    44: RobotAction(44, "waist", "Waist movement.", "@YAOBUYUNDONG#$", 3.0, ["waist", "waist movement"]),
    45: RobotAction(45, "twist", "Twist body.", "@NIUYINIU#$", 3.0, ["twist"]),
    46: RobotAction(46, "arch", "Arch body.", "@GONGYIGONG#$", 3.0, ["arch"]),
    47: RobotAction(47, "wave", "Wave motion.", "@BOLANGYUNDONG#$", 3.0, ["wave", "wave motion"]),
    48: RobotAction(48, "slow_twist", "Slow twist.", "@MANMANNIU#$", 3.0, ["slow twist"]),
    49: RobotAction(49, "spin_kick", "Spin kick.", "@XUANFENGTUI#$", 3.5, ["spin kick"]),
    50: RobotAction(50, "shake_left_leg", "Shake left leg.", "@DOUYOUTUI#$", 3.0, ["shake left leg"]),
    51: RobotAction(51, "shake_right_leg", "Shake right leg.", "@DOUZUOTUI#$", 3.0, ["shake right leg"]),
    52: RobotAction(52, "side_movement", "Side body movement.", "@CESHENYUNDONG#$", 3.0, ["side movement"]),
    53: RobotAction(53, "shake", "Shake body.", "@YAOYIYAO#$", 3.0, ["shake"]),
    67: RobotAction(67, "continuous_forward", "Continuous forward movement.", "@LIANXUXIANGQIAN#$", 4.0, ["continuous forward"]),
    68: RobotAction(68, "continuous_backward", "Continuous backward movement.", "@LIANXUXIANGHOU#$", 4.0, ["continuous backward"]),
    69: RobotAction(69, "continuous_right", "Continuous right turn.", "@LIANXUXIANGYOU#$", 4.0, ["continuous right"]),
    70: RobotAction(70, "continuous_left", "Continuous left turn.", "@LIANXUXIANGZUO#$", 4.0, ["continuous left"]),
    71: RobotAction(71, "continuous_swing", "Continuous swing.", "@LIANXUYAO#$", 4.0, ["continuous swing"]),
    72: RobotAction(72, "tail", "Wag tail.", "@YAOWEIBA#$", 2.5, ["tail", "wag tail"]),
    73: RobotAction(73, "follow", "Follow mode.", "@GENSUI#$", 4.0, ["follow", "follow me"]),
    74: RobotAction(74, "patrol", "Patrol mode.", "@XUNLUO#$", 4.0, ["patrol"]),
    79: RobotAction(79, "servo_debug", "Servo angle debug / neutral angle packet.", "servo angle packet", 2.0, ["servo", "servo debug"]),
}

ALIASES: Dict[str, int] = {}
for action_id, action in ACTIONS.items():
    ALIASES[action.name] = action_id
    for alias in action.aliases:
        ALIASES[alias.lower()] = action_id


def get_action(action: Union[int, str, None]) -> Optional[RobotAction]:
    if action is None:
        return None
    if isinstance(action, int):
        return ACTIONS.get(action)
    text = str(action).strip().lower()
    if text.isdigit():
        return ACTIONS.get(int(text))
    action_id = ALIASES.get(text)
    if action_id is None:
        return None
    return ACTIONS.get(action_id)


def build_model_action_prompt() -> str:
    lines = ["Supported robot actions:"]
    for action_id, action in sorted(ACTIONS.items()):
        lines.append(f"- {action_id}: {action.name} = {action.description}")
    return "\n".join(lines)


class RobotSerialController:
    def __init__(self, port: str = "COM3", baud: int = 115200, timeout: float = 0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def set_volume_min(self):
        self.ser.write(b"VOL_MIN\n")
        self.ser.flush()
        time.sleep(0.2)
        return self.read_available()

    def send_action(self, action: Union[int, str], wait: bool = True) -> Optional[RobotAction]:
        robot_action = get_action(action)
        if robot_action is None:
            print(f"Unknown robot action: {action}")
            return None

        self.ser.write((str(robot_action.action_id) + "\n").encode("utf-8"))
        self.ser.flush()
        print(f"sent ID: {robot_action.action_id} ({robot_action.name})")

        if wait:
            time.sleep(robot_action.estimated_wait_s)

        return robot_action

    def read_available(self):
        messages = []
        time.sleep(0.1)
        while self.ser.in_waiting:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                messages.append(line)
                print("robot:", line)
        return messages

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

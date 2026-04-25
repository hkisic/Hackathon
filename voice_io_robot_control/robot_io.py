"""Robot I/O wrapper used by the final main program."""

from robot_action_reference import RobotSerialController, get_action


class RobotIO:
    def __init__(self, port="COM3", baud=115200, timeout=0.5, set_volume_min=True):
        self.controller = RobotSerialController(port=port, baud=baud, timeout=timeout)
        if set_volume_min:
            self.controller.set_volume_min()

    def send_action_id(self, action_id, wait=True):
        return self.controller.send_action(action_id, wait=wait)

    def send_action_name(self, action_name, wait=True):
        return self.controller.send_action(action_name, wait=wait)

    def read_available(self):
        return self.controller.read_available()

    def close(self):
        self.controller.close()

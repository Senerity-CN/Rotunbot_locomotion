from .tester_type_base import TesterTypeBase
from engineai_rl_lib.command_filter import convert_to_visible_commands


class TesterBackwardCommands(TesterTypeBase):
    def set_goals(self) -> None:
        convert_to_visible_commands(self.env.vel_commands)
        self.env.vel_commands[:, 0] = -abs(self.env.vel_commands[:, 0])
        self.env.vel_commands[:, 1:] = 0
        return self.env.goal_dict

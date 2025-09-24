class Goals:
    def __init__(self, env):
        self.env = env

    def commands(self):
        return self.env.vel_commands * self.env.vel_commands_scales

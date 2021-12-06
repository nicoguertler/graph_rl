class EnvInfo:
    def __init__(self, reward = None, new_obs = None, done = None, info = None):
        self.reward = reward
        self.new_obs = new_obs
        self.done = done
        self.info = info

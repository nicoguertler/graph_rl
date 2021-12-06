class SubtaskTransition():
    def __init__(self, obs = None, action = None, ind_child = None, reward = None, 
            new_obs = None, ended = None, info = None):
        self.obs = obs
        self.action = action
        self.ind_child = ind_child
        self.reward = reward
        self.new_obs = new_obs
        self.ended = ended
        self.info = info

    def __str__(self):
        return "obs: {}\n".format(self.obs) + \
        "action: {}\n".format(self.action) + \
        "ind_child: {}\n".format(self.ind_child) + \
        "reward: {}\n".format(self.reward) + \
        "new_obs: {}\n".format(self.new_obs) + \
        "ended: {}\n".format(self.ended) + \
        "info: {}\n".format(self.info)


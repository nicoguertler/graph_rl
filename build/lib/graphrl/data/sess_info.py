class SessInfo:
    def __init__(self, ep_step = None, total_step = None, learn = True, 
            testing = False):
        self.ep_step = ep_step
        self.total_step = total_step
        self.learn = learn
        self.testing = testing

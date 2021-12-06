from gym.envs.registration import register

from .session.session import Session

# register test envs with gym
register(
        id = "Obstacle-v1",
        entry_point = "graph_rl.envs:ObstacleEnv"
        )


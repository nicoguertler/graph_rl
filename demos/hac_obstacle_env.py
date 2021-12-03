import argparse
from math import ceil

import gym
import torch

from graphrl.graphs import HACGraph
from graphrl.models import SACModel
from graphrl.subtasks import BoxSPSubtaskSpec, EnvSPSubtaskSpec
from graphrl import Session


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train HAC on Obstacle-v1.")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of nodes/layers in graph/hierarchy.")
    parser.add_argument("--max_n_actions", type=int, default=17, help="Action budget when pursuing a subgoal.")
    parser.add_argument("--render", default=False, action="store_true", help="Render while training.")
    parser.add_argument("--hidden_layers", default=2, type=int)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    args = parser.parse_args()

    # create environment
    env = gym.make("Obstacle-v1")

    # NOTE: Subtask specifications, model and graph can all be customized but are 
    # chosen in a generic fashion in this example.

    # define subtask specifications (i.e. specify objective in each node/layer)
    subtask_specs = []
    for _ in range(0, args.n_layers - 1):
        # lower layers see full observation and subgoal space is equal to observation space
        s_spec = BoxSPSubtaskSpec(
            max_n_actions=args.max_n_actions,
            goal_achievement_threshold=0.033,
            env=env
        )
        subtask_specs.append(s_spec)
    # highest layer pursues environment goal
    highest_s_spec = EnvSPSubtaskSpec(
        max_n_actions=ceil(float(env.max_episode_length)/args.max_n_actions**(args.n_layers - 1)),
        env=env,
        map_to_env_goal=env.map_to_env_goal # map from observation to goal
    )
    subtask_specs.append(highest_s_spec)

    # specify model (i.e. actor and critic) for SAC to optimize
    # NOTE: Input and output size are determined automatically
    HAC_kwargs = []
    for _ in range(args.n_layers):
        model = SACModel(
            hidden_layers_actor=[16]*args.hidden_layers,
            hidden_layers_critics=[16]*args.hidden_layers,
            activation_fns_actor=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            activation_fns_critics=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            learning_rate_actor=args.learning_rate,
            learning_rate_critics=args.learning_rate,
            device="cpu",
            force_negative=True
        )
        HAC_kwargs.append({"model": model})

    # create graph via Graph class
    graph = HACGraph(
        name="HAC_graph", 
        n_layers=args.n_layers,
        env=env,
        subtask_specs=subtask_specs,
        HAC_kwargs=HAC_kwargs,
        update_sgs_rendering=env.update_subgoals
    )

    # create and run session (training)
    sess = Session(graph, env)
    sess.run(
        n_steps=100000,
        learn=True,
        render=args.render,
        success_reward=0.0
    )

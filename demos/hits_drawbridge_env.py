
import argparse
from math import ceil

import gym
import torch
import dyn_rl_benchmarks

from graph_rl.graphs import HiTSGraph
from graph_rl.models import SACModel
from graph_rl.subtasks import DictInfoHidingTGSubtaskSpec, EnvSPSubtaskSpec
from graph_rl import Session


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train HAC on Obstacle-v1.")
    parser.add_argument(
        "--max_n_timed_subgoals", type=int, default=5, help="Timed subgoal budget."
    )
    parser.add_argument(
        "--render", default=False, action="store_true", help="Render while training."
    )
    parser.add_argument("--hidden_layers", default=2, type=int)
    parser.add_argument("--learning_rate", default=1.0e-4, type=float)
    args = parser.parse_args()

    # create environment
    env = gym.make("Drawbridge-v1")

    ###########################################################################
    # define subtask specifications (i.e. specify objective in each node/layer)
    ###########################################################################

    # Layer 0 (lower layer):
    # Objective is to reach a timed goal (TG).
    # The observation space of the environment is a dict space so keys can be 
    # used to select what the layer is to see and what the subgoal space should 
    # contain.
    partial_obs_keys = {"ship_pos", "ship_vel", "sails_unfurled", "bridge_phase"}
    goal_keys = {"ship_pos", "ship_vel", "sails_unfurled"}
    goal_achievement_threshold = {
        "ship_pos": 0.05,
        "ship_vel": 0.2,
        "sails_unfurled": 0.1
    }
    s_spec_0 = DictInfoHidingTGSubtaskSpec(
        goal_achievement_threshold=goal_achievement_threshold, 
        partial_obs_keys=partial_obs_keys, 
        goal_keys=goal_keys, 
        env=env, 
        delta_t_max=env.max_episode_length/args.max_n_timed_subgoals, 
    )

    # Layer 1 (higher layer):
    # Higher layer pursues a conventional environment goal.
    s_spec_1 = EnvSPSubtaskSpec(
        max_n_actions=args.max_n_timed_subgoals,
        env=env,
        map_to_env_goal=lambda partial_obs: partial_obs["ship_pos"]
    )
    subtask_specs = [s_spec_0, s_spec_1]

    ###########################################################
    # specify model (i.e. actor and critic) for SAC to optimize
    ###########################################################

    # NOTE: Input and output sizes are determined automatically
    algo_kwargs = []
    # entropy coefficient of SAC
    flat_algo_kwargs = {"alpha": 0.02}
    buffer_sizes = [400000, 10000]
    for i in range(2):
        model = SACModel(
            hidden_layers_actor=[16]*args.hidden_layers,
            hidden_layers_critics=[16]*args.hidden_layers,
            activation_fns_actor=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            activation_fns_critics=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            learning_rate_actor=args.learning_rate,
            learning_rate_critics=args.learning_rate,
            device="cpu",
            # q function of higher level should be restricted to < 0 because
            # of shortest path objective (reward <= 0)
            force_negative=True if i == 1 else False 
        )
        algo_kwargs.append(
        {
            "model": model, 
            "flat_algo_name": "SAC",
            "flat_algo_kwargs": flat_algo_kwargs,
            "buffer_size": buffer_sizes[i]
        })


    ##############################
    # create graph via Graph class
    ##############################

    graph = HiTSGraph(
        name="HiTS_graph", 
        n_layers=2,
        env=env,
        subtask_specs=subtask_specs,
        HAC_kwargs=algo_kwargs[-1], # algorithm parameters of highest layer
        HiTS_kwargs=algo_kwargs[:-1], # algorithm parameters of all other layers
        update_tsgs_rendering=env.update_timed_subgoals,
        # highest level sees env reward plus penalty for emitting timed subgoals
        env_reward_weight=2.0 
    )

    ###################################
    # create and run session (training)
    ###################################

    sess = Session(graph, env)
    sess.run(
        n_steps=500000,
        learn=True,
        render=args.render,
        success_reward=0.0
    )

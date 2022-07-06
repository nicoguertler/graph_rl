import os 
import time
from datetime import datetime
import json

from gym import GoalEnv
import numpy as np
from tianshou.data.utils.converter import to_numpy
import torch
from torch.utils.tensorboard import SummaryWriter

from ..data import EnvInfo, SessInfo
from ..logging import CSVLogger, read_csv_log


class Session:

    def __init__(self, graph, env):
        self.graph = graph
        self.env = env

        # detect whether environment is goal-based
        if isinstance(env, GoalEnv):
            self._goal_based_env = True
        else:
            self._goal_based_env = False

    def _process_env_observation(self, obs):
        # convert lists and tuples into numpy arrays
        return to_numpy(obs)

    def step_env_and_graph(self, env_info, sess_info, render, render_frequency, episode, total_step_multiplier = 1.):
            # get atomic action from forward pass through graph
            action = self.graph.get_atomic_action(env_info.new_obs, sess_info, testing = sess_info.testing)

            # step environment
            new_raw_obs, env_rew, env_done, info = self.env.step(action)
            new_env_obs = self._process_env_observation(new_raw_obs)
            new_env_info = EnvInfo(
                    new_obs = new_env_obs, 
                    reward = env_rew,
                    done = env_done,
                    info = info)

            sess_info.total_step += 1*total_step_multiplier
            sess_info.ep_step += 1

            # render 
            if render and episode % render_frequency == 0:
                self.env.render()

            # register experience in two backward passes through graph
            graph_done = self.graph.register_experience(new_env_info, sess_info)

            return new_env_info, graph_done

    def run(self, n_steps = None, n_episodes = None, max_runtime = None, learn = True, render = False, 
            render_frequency = 10, test = False, test_frequency = 10, test_episodes = 1, 
            test_render = False, test_render_frequency = 1, verbose = 1, 
            tensorboard_logdir = None, csv_logdir = None, run_name = None, 
            append_run_name_to_log_paths = True, torch_num_threads = None, 
            cb_after_train_episode = None, total_step_init = 0, append_to_logfiles = False, 
            success_reward = None, train=True):
        """Run reinforcement learning session with policy induced by graph in env.
        Args:
            n_steps (int or None): Number of training timesteps to take. Can be None if 
                n_episodes is not None. In this case the argument is ignored. 
                One of the arguments n_steps and n_episodes has to be an int 
                and one None.
            n_episodes (int or None): Number of training episodes to run. Can be None if 
                n_steps is not None. In this case the argument is ignored. 
                One of the arguments n_steps and n_episodes has to be an int 
                and one None.
            success_reward (float or None): If this reward is encountered 
                during an episode, it is counted as a successful episode."""

        assert {type(n_steps), type(n_episodes)} == {int, type(None)}, \
                "Please provide either a number of time steps to take or \
                a number of episodes to run (in n_steps or n_episodes)."


        if torch_num_threads is not None:
            torch.set_num_threads(torch_num_threads)
            print(f"Using {torch_num_threads} threads in pytorch.")

        if verbose >= 1:
            print("-------------------------")
            print("Running session with")
            print("Env class: {}".format(type(self.env).__name__))
            print("Graph class: {}".format(type(self.graph).__name__))
            if n_steps is not None:
                print("Number of timesteps: {}".format(n_steps))
            if n_episodes is not None:
                print("Number of episodes: {}".format(n_episodes))
            print("Learn: {}".format(learn))
            print("-------------------------\n")

        if run_name is None:
            current_time = datetime.now()
            run_name = current_time.strftime("%Y_%m_%d_%H_%M")

        if tensorboard_logdir is not None:
            if append_run_name_to_log_paths:
                tb_logdir = os.path.join(tensorboard_logdir, run_name, "tensorboard")
            else:
                tb_logdir = tensorboard_logdir
            writer = SummaryWriter(tb_logdir)
            self.graph.set_tensorboard_writer(writer)
        else:
            writer = None

        if csv_logdir is not None:
            if append_run_name_to_log_paths:
                csv_logdir_run = os.path.join(csv_logdir, run_name, "log")
            else:
                csv_logdir_run = csv_logdir
            os.makedirs(csv_logdir_run, exist_ok = True)
            self.graph.create_logfiles(csv_logdir_run, append = append_to_logfiles)
            fieldnames_train = ["return", "length", "time"]
            fieldnames_test = ["return", "length", "step", "time"]
            if success_reward is not None:
                fieldnames_train.append("success")
                fieldnames_test.append("success")
                fieldnames_test.append("return_success")

            logger_train = CSVLogger(os.path.join(csv_logdir_run, "session_train.csv"), 
                    fieldnames_train, append = append_to_logfiles)
            logger_test = CSVLogger(os.path.join(csv_logdir_run, "session_test.csv"), 
                    fieldnames_test, append = append_to_logfiles)
        else:
            logger_train = None
            logger_test = None

        ep_return = 0.
        test_episode = 0
        episode = 0
        success = False

        env_info = EnvInfo(
                new_obs = self._process_env_observation(self.env.reset()), 
                done = True)
        sess_info = SessInfo(
                ep_step = 0,
                total_step = total_step_init,
                learn = learn,
                testing = False)
        self.graph.reset()

        t_start = time.process_time()

        while (n_steps is None or sess_info.total_step < n_steps) and \
                (n_episodes is None or episode < n_episodes) and \
                (max_runtime is None or time.process_time() - t_start < max_runtime):

            if verbose >= 1 and (env_info.done or graph_done):
                print("\nEpisode {}:".format(episode))

            if verbose >= 2:
                print("Step {}:".format(sess_info.total_step))

            if train:
                env_info, graph_done = self.step_env_and_graph(env_info, sess_info, 
                        render, render_frequency, episode)
                if success_reward is not None and env_info.reward == success_reward:
                    success = True
                    print("Successful episode!")
                ep_return += env_info.reward
            else:
                env_info = EnvInfo()
                graph_done = False

            # end of a training episode
            if not train or env_info.done or graph_done:
                if verbose >= 1:
                    print("Return: {}".format(ep_return))
                    print("env_done: {}".format(env_info.done))
                    print("graph_done: {}".format(graph_done))
                    print("")

                if writer is not None:
                    writer.add_scalar("env/return/train", ep_return, sess_info.total_step)
                    writer.add_scalar("env/ep_length/train", sess_info.ep_step, sess_info.total_step)
                    if success_reward is not None:
                        writer.add_scalar("env/success/train", int(success), sess_info.total_step)
                if logger_train is not None:
                    row_dict = {
                            "return": ep_return, 
                            "length": sess_info.ep_step, 
                            "time": logger_train.time_passed()
                            }
                    if success_reward is not None:
                        row_dict["success"] = int(success)
                    logger_train.log(row_dict)

                if cb_after_train_episode is not None:
                    cb_after_train_episode(self.graph, sess_info, ep_return, graph_done)

                # learn
                if sess_info.learn:
                    self.graph.learn(sess_info)

                episode += 1
                success = False
                ep_return = 0

                # testing
                if test and episode % test_frequency == 0:
                    if verbose >= 1:
                        print("Testing")
                    cumulative_env_return_test = 0
                    # return of successful episodes
                    cumulative_env_return_test_success = 0
                    cumulative_env_length_test = 0
                    cumulative_success_test = 0
                    sess_info_test = SessInfo(
                            ep_step = 0,
                            total_step = sess_info.total_step,
                            learn = False,
                            testing = True)
                    for test_episode in range(test_episodes):
                        env_info = EnvInfo(
                                new_obs = self._process_env_observation(self.env.reset()), 
                                done = False)
                        self.graph.reset()
                        env_return_test = 0
                        success_test = False
                        sess_info_test.ep_step = 0
                        while True:
                            env_info, graph_done = self.step_env_and_graph(env_info, sess_info_test, 
                                    test_render, test_render_frequency, test_episode, 1./test_episodes)

                            if (success_reward is not None and 
                                    env_info.reward == success_reward):
                                success_test = True
                            env_return_test += env_info.reward

                            if env_info.done or graph_done:
                                if verbose >= 1:
                                    print("Return: {}".format(env_return_test))
                                    print("env_done: {}".format(env_info.done))
                                    print("graph_done: {}\n".format(graph_done))
                                cumulative_env_return_test += env_return_test
                                cumulative_env_length_test += sess_info_test.ep_step
                                if success_test:
                                    cumulative_env_return_test_success += env_return_test
                                cumulative_success_test += float(success_test)
                                break

                    avg_return_test = cumulative_env_return_test/test_episodes
                    avg_length_test = cumulative_env_length_test/test_episodes
                    avg_success_test = cumulative_success_test/test_episodes
                    if cumulative_success_test > 0.:
                        avg_return_test_success = (cumulative_env_return_test_success/
                                cumulative_success_test)
                    else:
                        avg_return_test_success = np.nan
                    if writer is not None:
                        # fractional steps for testing to keep logging consistent
                        writer.add_scalar("env/return/test", avg_return_test, 
                                sess_info.total_step)
                        writer.add_scalar("env/ep_length/test", avg_length_test, 
                                sess_info.total_step)
                        if success_reward is not None:
                            writer.add_scalar("env/success/test", avg_success_test,
                                sess_info.total_step)
                    if logger_test is not None:
                        row_dict = {
                                "return": avg_return_test, 
                                "length": avg_length_test, 
                                "step": sess_info.total_step, 
                                "time": logger_test.time_passed()
                                }
                        if success_reward is not None:
                            row_dict["success"] = avg_success_test
                            row_dict["return_success"] = avg_return_test_success
                        logger_test.log(row_dict)

                # reset env_info and graph for next training episode
                env_info = EnvInfo(
                        new_obs = self._process_env_observation(self.env.reset()), 
                        done = True)
                sess_info.ep_step = 0
                self.graph.reset()

        t_elapsed = time.process_time() - t_start

        session_props = {}
        for arg_name in self.run.__code__.co_varnames[:self.run.__code__.co_argcount]:
            if arg_name != "self" and type(locals()[arg_name]) in [str, bool, int, float]:
                session_props[arg_name] = locals()[arg_name]
        session_props["t_elapsed"] = t_elapsed
        session_props["timed_out"] = max_runtime is not None and t_elapsed >= max_runtime
        session_props["total_step"] = sess_info.total_step
        # log session properties (all arguments to run, elapsed time)
        if csv_logdir is not None:
            session_file = open(os.path.join(csv_logdir_run, "session.json"), "w")
            json.dump(session_props, session_file, indent = 4)

        return session_props

    @classmethod
    def read_logfiles(cls, log_path):
        data = {
                "train": read_csv_log(os.path.join(log_path, "session_train.csv")), 
                "test": read_csv_log(os.path.join(log_path, "session_test.csv"))
                }
        # reconstruct step at end of training episode from episode length
        current_step = 0
        steps = []
        for l in data["train"]["length"]:
            current_step += l
            steps.append(current_step)
        data["train"]["step"] = steps
        return data

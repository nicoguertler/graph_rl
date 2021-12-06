from abc import ABC, abstractmethod
from copy import copy
import os

import tianshou as ts
import numpy as np
import torch

from . import Algorithm
from ..nodes import Node
from ..policies import TianshouPolicy

class OffPolicyAlgorithm(Algorithm):
    """Off-policy algorithm. Supports SAC and DDPG."""

    supported_flat_offpol_algos = {
            "SAC": ts.policy.SACPolicy,
            "DDPG": ts.policy.DDPGPolicy
            }

    class _Transition():
        def __init__(self, subtask_tr, env_info, child_feedback, algo_info):
            self.subtask_tr = subtask_tr
            self.env_info = env_info
            self.child_feedback = child_feedback
            self.algo_info = algo_info

        def __str__(self):
            return "subtask_tr: {}\n".format(self.subtask_tr) + \
            "child_feedback: {}\n".format(self.child_feedback) + \
            "algo_info: {}\n".format(self.algo_info)

    def __init__(self, name, flat_algo_name, model, fully_random_fraction, flat_algo_kwargs=None, 
            buffer_size=100000, batch_size=128, learning_starts=0, grad_steps_per_env_step=1.0):
        """
        Args:
            model (Model): Model instance which provides...
            for SAC:
            * actor: Pytorch module that returns ((mu, sigma), hidden_state) in forward method. mu and sigma refer to value in R^n before squashing.
            * two critics
            for DDPG:
            * actor: Pytorch module that returns (action, hidden_state) in forward method. action refers to value in R^n before squashing.
            * one critic

            learning_starts: Learnings starts after this many transitions have been collected.
        """
        super().__init__(name)

        if flat_algo_name not in self.supported_flat_offpol_algos:
            raise ValueError("Flat off policy RL algorithm \"{}\" not supported.".
                    format(flat_algo_name))

        assert model is not None

        self._flat_algo_name = flat_algo_name
        self._flat_algo_class = self.supported_flat_offpol_algos[flat_algo_name]
        self._flat_algorithm = None
        if flat_algo_kwargs is None:
            self._flat_algo_kwargs = {}
        else:
            self._flat_algo_kwargs = copy(flat_algo_kwargs)
        self._model = model
        self._fully_random_fraction = fully_random_fraction
        self._tb_low_level_writer = None
        self._actions_taken = 0
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._learning_starts = learning_starts
        self._grad_steps_per_env_step = grad_steps_per_env_step
        self._episode_transitions = []

        # if a target entropy is given and no optimizer for the entropy coefficient is 
        # specified use adam and interpret third entry of tuple as learning rate
        if flat_algo_name == "SAC" and "alpha" in self._flat_algo_kwargs: 
            if (isinstance(self._flat_algo_kwargs["alpha"], list) 
                and not isinstance(self._flat_algo_kwargs["alpha"][0], torch.optim.Optimizer)):
                target_entropy = self._flat_algo_kwargs["alpha"][0]
                log_alpha = torch.tensor([self._flat_algo_kwargs["alpha"][1]], dtype = torch.float, requires_grad = True)
                alpha_optim = torch.optim.Adam([log_alpha], lr = self._flat_algo_kwargs["alpha"][2])
                self._flat_algo_kwargs["alpha"] = (target_entropy, log_alpha, alpha_optim)


        # set action range to [-1., 1.] in tianshou and rescale in graph_rl if needed
        self._flat_algo_kwargs["action_range"] = [-1., 1.]

        self._last_learning_step = None

        self._replay_buffer = ts.data.ReplayBuffer(size = buffer_size)

    def _map_action_to_default_space(self, action):
        res = (action - self._action_offset)/self._action_scale
        res = np.clip(res, -1., 1.)
        return res

    def _add_to_flat_replay_buffer(self, f_trans):
        # map action back to [-1, 1]^n before adding transition to buffer
        action = self._action_space.flatten_value(f_trans.action)
        mapped_action = self._map_action_to_default_space(action)
        self._replay_buffer.add(
                obs = self._observation_space.flatten_value(f_trans.obs), 
                act = mapped_action, 
                rew = f_trans.reward, 
                done = f_trans.done,
                obs_next = self._observation_space.flatten_value(f_trans.new_obs))

    def get_policy(self, name, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        flat_action_space = action_space.get_flat_space()
        self._action_scale = 0.5*(flat_action_space.high - flat_action_space.low)
        self._action_offset = 0.5*(flat_action_space.high + flat_action_space.low)
        flat_obs_dim = observation_space.get_flat_dim()
        flat_action_dim = action_space.get_flat_dim()
        flat_gym_action_space = flat_action_space.get_gym_space()

        self._model.create(flat_obs_dim, flat_action_dim)
        self._learning_rate_critics = self._model._learning_rate_critics
        self._learning_rate_actor = self._model._learning_rate_actor

        # actor and critic modules
        nets = {
                "actor": self._model.actor,
                "actor_optim": self._model.actor_optim, 
                f"critic{'1' if self._flat_algo_name == 'SAC' else ''}": self._model.critics[0], 
                f"critic{'1' if self._flat_algo_name == 'SAC' else ''}_optim": self._model.critic_optims[0]
                }
        if self._flat_algo_name == "SAC":
            # SAC expects two critics while DDPG wants only one
            nets["critic2"] = self._model.critics[1]
            nets["critic2_optim"] = self._model.critic_optims[1]
        elif self._flat_algo_name == "DDPG":
            # Gaussian noise with n-dim. mean and variance
            class GaussianNoiseND(ts.exploration.random.BaseNoise):
                """Gaussian noise with n-dim. mean and variance."""

                def __init__(self, mu=[0.0], sigma=[1.0]):
                    super().__init__()
                    self._mu = mu
                    assert np.all(0 <= sigma), "Noise std should not be negative."
                    self._sigma = sigma

                def __call__(self, size):
                    # Note: size has to coincide with shape of mu and sigma
                    return np.random.normal(self._mu, self._sigma)

            # DDPG needs a noise object
            # flatten noise sigma object to vector using action space
            sigma = action_space.flatten_value(self._flat_algo_kwargs["noise_sigma"])
            exploration_noise = GaussianNoiseND(mu=np.zeros_like(sigma), sigma=sigma)
            self._flat_algo_kwargs["exploration_noise"] = exploration_noise
            del self._flat_algo_kwargs["noise_sigma"]

        self._flat_algorithm = self._flat_algo_class(**nets, **self._flat_algo_kwargs, 
                action_space=flat_gym_action_space)
        self._flat_algorithm.eval()

        policy = TianshouPolicy(
                name = name, 
                observation_space = observation_space, 
                action_space = action_space, 
                ts_policy = self._flat_algorithm, 
                hidden_net = self._model.actor,
                fully_random_fraction = self._fully_random_fraction)

        return policy

    @abstractmethod
    def _add_experience_to_flat_algo(self, parent_info, deterministic_episode, node_is_sink, sess_info):
        """Process transitions from ending episode and store them in flat replay buffer.

        At this point hindsight manipulations can be implemented.
        """

        pass

    def add_experience(self, env_obs, env_info, subtask_trans, parent_info, 
            child_feedback, algo_info, interruption, sess_info):

        # save transition to episode transitions buffer
        transition = self._Transition(
                subtask_tr = subtask_trans,
                env_info = env_info,
                child_feedback = child_feedback,
                algo_info = algo_info
                )
        self._episode_transitions.append(transition)
        if self._verbose:
            print(transition)
        self._actions_taken += 1

        # tensorboard logging
        if self._tb_writer is not None:
            obs = self._observation_space.flatten_value(subtask_trans.obs)
            if self._log_q_values and len(self._episode_transitions) == 1:
                act = self._action_space.flatten_value(subtask_trans.action)
                with torch.no_grad():
                    q_values = [critic(obs[None], act[None]).numpy() for critic  in self._model.critics]
                    q_value = min(q_values)
                mode = "test" if sess_info.testing else "train"
                self._tb_writer.add_scalar(f"{self.name}/{mode}/q_value", q_value, sess_info.total_step)
            if self._flat_algo_name == "SAC" and not sess_info.testing and parent_info is None:
                with torch.no_grad():
                    net_out, _ = self._model.actor(obs[None])
                    _, sigma = net_out
                    entropy = np.log(sigma.numpy()).sum() + (0.5 + np.log(np.sqrt(2.*np.pi)))*len(sigma)
                self._tb_writer.add_scalar(f"{self.name}/train/entropy", entropy, sess_info.total_step)

        # if the subtask has ended (i.e. control will be returned to active parent
        # node), process episode transitions and add them to replay buffer 
        # of flat algorithm
        if subtask_trans.ended or env_info.done:
            if parent_info:
                deterministic_episode = algo_info["is_deterministic"]
            else:
                deterministic_episode = False

            node_is_sink = child_feedback is Node.sink_identification_feedback
            self._add_experience_to_flat_algo(parent_info, deterministic_episode, 
                    node_is_sink, sess_info)

    def learn(self, current_step):
        if self._last_learning_step is None:
            self._last_learning_step = current_step - 1

        elapsed_steps = current_step - self._last_learning_step
        n_grad_steps = int(self._grad_steps_per_env_step*elapsed_steps)
        assert(elapsed_steps > 0)
        if len(self._replay_buffer) >= self._batch_size and current_step > self._learning_starts:
            for _ in range(int(n_grad_steps)):
                batch, indices = self._replay_buffer.sample(self._batch_size)
                batch = self._flat_algorithm.process_fn(batch, self._replay_buffer, indices)
                results = self._flat_algorithm.learn(batch)

                # tensorboard logging of losses
                if self._tb_writer is not None:
                    for key, item in results.items():
                        self._tb_writer.add_scalar(f"{self.name}/train/{key}", item, current_step)

        self._last_learning_step = current_step

    def get_parameters(self):
        """Returns dictionary containing all learned parameters."""
        params = {
                "flat_algo_name": self._flat_algo_name, 
                "model_params": self._model.get_parameters()
                }
        return params

    def get_state(self):
        """Returns dictionary containing state."""
        state = {
                "flat_algo_name": self._flat_algo_name, 
                "model_state": self._model.get_state(), 
                "model_params": self._model.get_parameters(), 
                "replay_buffer": self._replay_buffer
                }
        return state

    def save_state(self, node_id, dir_path):
        """Saves state and returns dictionary with file paths."""
        replay_buffer_file_name = f"{node_id}_algo_replay_buffer.hdf5"
        self._replay_buffer.save_hdf5(os.path.join(dir_path, replay_buffer_file_name))
        state = {
                "flat_algo_name": self._flat_algo_name, 
                "model_state": self._model.get_state(), 
                "model_params": self._model.get_parameters(), 
                "replay_buffer": replay_buffer_file_name
                }
        return state

    def _override_target_networks(self):
        """Override target networks with current networks."""
        tau = self._flat_algorithm._tau
        self._flat_algorithm._tau = 1.0
        self._flat_algorithm.sync_weight()
        self._flat_algorithm._tau = tau

    def load_parameters(self, params):
        """Load all learned parameters from dictionary."""
        assert self._flat_algo_name == params["flat_algo_name"]
        self._model.load_parameters(params["model_params"])
        self._override_target_networks()

    def load_state(self, state, dir_path = None):
        """Load state from dictionary."""
        assert self._flat_algo_name == state["flat_algo_name"]
        self._model.load_state(state["model_state"])
        self._model.load_parameters(state["model_params"])
        self._override_target_networks()
        if isinstance(state["replay_buffer"], ts.data.ReplayBuffer):
            self._replay_buffer = state["replay_buffer"]
        else:
            del self._replay_buffer
            self._replay_buffer = ts.data.ReplayBuffer.load_hdf5(
                    os.path.join(dir_path, state["replay_buffer"])
                    )

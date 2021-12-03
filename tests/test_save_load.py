import tempfile

import torch

from graphrl import Session
from graphrl.data import SessInfo
from graphrl.envs import ObstacleEnv
from graphrl.graphs import HACGraph
from graphrl.models import SACModel
from graphrl.subtasks import EnvSPSubtaskSpec, BoxSPSubtaskSpec


def compare_dicts_with_tensors(d1, d2):
    for k, v in d1.items():
        if k not in d2:
            return False
        else:
            if isinstance(v, dict):
                if not compare_dicts_with_tensors(v, d2[k]):
                    return False
            elif isinstance(v, torch.Tensor):
                if (v != d2[k]).byte().any():
                    return False
            elif isinstance(v, list):
                if not compare_lists_with_tensors(v, d2[k]):
                    return False
            else:
                if v != d2[k]:
                    return False
    return True


def compare_lists_with_tensors(l1, l2):
    if len(l1) != len(l2):
        return False
    for e1, e2 in zip(l1, l2):
        if isinstance(e1, dict):
            if not compare_dicts_with_tensors(e1, e2):
                return False
        elif isinstance(e1, torch.Tensor):
            if (e1 != e2).byte().any():
                return False
        elif isinstance(e1, list):
            if not compare_lists_with_tensors(e1, e2):
                return False
        else:
            if e1 != e2:
                return False

    return True


def test_save_load_state():
    n_layers = 2
    env = ObstacleEnv()

    def create_graph(n_layers):
        # create HAC graph for Obstacle environment
        subtask_specs = []
        HAC_args = []

        for i in range(n_layers):
            subtask_specs.append(
                EnvSPSubtaskSpec(10, env, env.map_to_env_goal)
                if i == n_layers - 1
                else BoxSPSubtaskSpec(10, 0.05, env)
            )
            model = SACModel(
                hidden_layers_actor=[1],
                hidden_layers_critics=[1],
                activation_fns_actor=[torch.nn.ReLU()],
                activation_fns_critics=[torch.nn.ReLU()],
                learning_rate_actor=1.0e-3,
                learning_rate_critics=1.0e-3,
                device="cpu"
            )
            # TODO: Set buffer size explicitly
            HAC_args.append({
                "flat_algo_name": "SAC",
                "flat_algo_kwargs": {},
                "model": model,
                "buffer_size": int(2e4)
            })
        graph = HACGraph(
            name="test_graph",
            n_layers=n_layers,
            env=env,
            subtask_specs=subtask_specs,
            HAC_kwargs=HAC_args
        )
        return graph

    # create graph
    graph1 = create_graph(n_layers)

    # run session to collect data
    sess = Session(graph1, env)
    sess.run(
        n_steps=5000,
        learn=False,
        render=False,
        test=False,
        torch_num_threads=1
    )

    # save state of first graph
    with tempfile.TemporaryDirectory(prefix="graphrl_test_") as dir_path:
        graph1.save_state(dir_path)

        # load state of first graph from file into second graph
        graph2 = create_graph(n_layers)
        graph2.load_state(dir_path=dir_path)

        graph1.reset()
        graph2.reset()

        def compare_nodes(node1, node2):
            # test equality of replay buffers
            assert (node1.algorithm._replay_buffer.obs[:]
                    == node2.algorithm._replay_buffer.obs[:]).all()
            assert (node1.algorithm._replay_buffer.act[:]
                    == node2.algorithm._replay_buffer.act[:]).all()
            assert (node1.algorithm._replay_buffer.rew[:]
                    == node2.algorithm._replay_buffer.rew[:]).all()
            assert (node1.algorithm._replay_buffer._index
                    == node2.algorithm._replay_buffer._index)
            # test equality of model states and parameters
            assert compare_dicts_with_tensors(
                node1.algorithm._model.get_state(),
                node2.algorithm._model.get_state()
            )
            assert compare_dicts_with_tensors(
                node1.algorithm._model.get_parameters(),
                node2.algorithm._model.get_parameters()
            )
            # make sure target networks are updated with saved parameters
            # (target network and network have same parameters initially)
            assert compare_dicts_with_tensors(
                node1.algorithm._flat_algorithm.critic1.state_dict(),
                node2.algorithm._flat_algorithm.critic1_old.state_dict(),
            )

        nodes1 = graph1.entry_node.get_descendants_recursively()
        nodes2 = graph2.entry_node.get_descendants_recursively()

        for node1, node2 in zip(nodes1, nodes2):
            compare_nodes(node1, node2)

        # test equality of emitted action
        obs = env.reset()
        sess_info = SessInfo()
        assert (
            graph1.get_atomic_action(obs, sess_info, testing=True)
            == graph2.get_atomic_action(obs, sess_info, testing=True)
                ).all()


if __name__ == "__main__":
    test_save_load_state()

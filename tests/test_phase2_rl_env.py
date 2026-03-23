import collections
import tempfile
import unittest
from pathlib import Path

from graphgen.ai_phase2.refine_graph import refine_assignments
from graphgen.ai_phase2.rl_env import GraphDirectionRefineEnv, build_candidate_edge_ids


class Phase2RLEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            GraphDirectionRefineEnv([], {}, {}, {})
        except ImportError:
            cls._gym_available = False
        else:
            cls._gym_available = True

    def _synthetic_junction_graph(self):
        j = (0.0, 0.0)
        west = (-1.0, 0.0)
        east = (1.0, 0.0)
        north = (1.0, 1.0)

        edge_list = [
            {"id": 0, "u": j, "v": west, "kind": "L", "geom": {"type": "LINE", "a": j, "b": west}},
            {"id": 1, "u": j, "v": east, "kind": "L", "geom": {"type": "LINE", "a": j, "b": east}},
            {
                "id": 2,
                "u": north,
                "v": j,
                "kind": "A",
                "geom": {"type": "ARC", "pts": [north, (0.6, 0.8), (0.3, 0.5), (0.1, 0.2), j]},
            },
        ]

        adj = collections.defaultdict(set)
        for e in edge_list:
            adj[e["u"]].add(e["v"])
            adj[e["v"]].add(e["u"])
        return edge_list, adj, {}

    def test_env_reset_step_smoke(self):
        if not self._gym_available:
            self.skipTest("gymnasium not installed")
        edge_list, adj, station_nodes = self._synthetic_junction_graph()
        env = GraphDirectionRefineEnv(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            initial_assign={0: 0, 1: 0, 2: 0},
        )
        obs, _ = env.reset(seed=123)
        self.assertEqual(obs.shape[0], 11)

        next_obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(next_obs.shape[0], 11)
        self.assertIsInstance(reward, float)
        self.assertIn("score", info)
        self.assertFalse(truncated)
        self.assertIsInstance(terminated, bool)

    def test_reward_positive_when_flip_improves_score(self):
        if not self._gym_available:
            self.skipTest("gymnasium not installed")
        edge_list, adj, station_nodes = self._synthetic_junction_graph()
        env = GraphDirectionRefineEnv(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            initial_assign={0: 0, 1: 0, 2: 0},
            candidate_edge_ids=[0],
            step_penalty=0.0,
        )
        env.reset(seed=7)
        _, reward, terminated, _, _ = env.step(1)
        self.assertGreater(reward, 0.0)
        self.assertTrue(terminated)

    def test_ppo_inference_smoke(self):
        if not self._gym_available:
            self.skipTest("gymnasium not installed")
        try:
            from stable_baselines3 import PPO
        except Exception:
            self.skipTest("stable-baselines3 not installed")

        edge_list, adj, station_nodes = self._synthetic_junction_graph()
        env = GraphDirectionRefineEnv(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            initial_assign={0: 0, 1: 0, 2: 0},
        )
        model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, seed=3, verbose=0)
        model.learn(total_timesteps=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "phase2_ppo_test_model"
            model.save(str(model_path))
            refined = refine_assignments(
                edge_list=edge_list,
                adj=adj,
                station_nodes=station_nodes,
                assign={0: 0, 1: 0, 2: 0},
                mode="ppo",
                model_path=model_path,
                seed=3,
            )
        self.assertEqual(set(refined.keys()), {0, 1, 2})

    def test_candidate_edge_filter(self):
        if not self._gym_available:
            self.skipTest("gymnasium not installed")
        edge_list, adj, station_nodes = self._synthetic_junction_graph()
        env = GraphDirectionRefineEnv(edge_list, adj, station_nodes, {0: 0, 1: 0, 2: 0})
        candidate_ids = build_candidate_edge_ids(env.features)
        self.assertTrue(len(candidate_ids) >= 1)


if __name__ == "__main__":
    unittest.main()

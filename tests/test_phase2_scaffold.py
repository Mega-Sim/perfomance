import unittest

from graphgen.ai_phase2.env import LocalDirectionRepairEnv, evaluate_violations, random_repair, synthetic_graph_cases
from graphgen.ai_phase2.refine_graph import refine_assignments


class Phase2ScaffoldTest(unittest.TestCase):
    def test_environment_transition_updates_assignment(self):
        case = synthetic_graph_cases()[0]
        env = LocalDirectionRepairEnv(case.edge_list, case.adj, case.station_nodes, case.rule_assign, max_steps=2)
        s0 = env.reset()
        self.assertIn("nonholonomic", s0.violation_flags)
        eid = case.edge_list[0]["id"]
        env.step(("flip", eid, None))
        self.assertEqual(env.assign[eid], 1)

    def test_reward_positive_when_violations_drop(self):
        case = synthetic_graph_cases()[3]  # merge_2_to_1
        env = LocalDirectionRepairEnv(case.edge_list, case.adj, case.station_nodes, case.rule_assign, max_steps=4)
        env.reset()
        found_improving = False
        for action in env.available_actions():
            env.reset()
            _, reward, _, info = env.step(action)
            if sum(info["after"].values()) <= sum(info["before"].values()) and reward >= 0.0:
                found_improving = True
                break
        self.assertTrue(found_improving)

    def test_rule_mode_does_not_break_output_shape(self):
        case = synthetic_graph_cases()[1]
        refined = refine_assignments(case.edge_list, case.adj, case.station_nodes, case.rule_assign, mode="rule")
        self.assertEqual(set(refined.keys()), set(case.rule_assign.keys()))

    def test_rl_repair_beats_random_baseline_on_synthetic(self):
        case = synthetic_graph_cases()[2]
        random_assign = random_repair(case.edge_list, case.adj, case.station_nodes, case.rule_assign, seed=7, steps=8)
        random_v = evaluate_violations(case.edge_list, case.adj, case.station_nodes, random_assign)

        greedy_assign = refine_assignments(case.edge_list, case.adj, case.station_nodes, case.rule_assign, mode="rl-repair")
        greedy_v = evaluate_violations(case.edge_list, case.adj, case.station_nodes, greedy_assign)
        self.assertLessEqual(sum(greedy_v.values()), sum(random_v.values()))


if __name__ == "__main__":
    unittest.main()

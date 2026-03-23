import collections
import unittest

from graphgen.ai_phase2.reward import score_assignment_quality
from src.build_graph import (
    count_nonholonomic_branch_violations,
    solve,
    violates_nonholonomic_branch_rule,
)


class NonholonomicBranchRegressionTest(unittest.TestCase):
    def _synthetic_junction_graph(self):
        j = (0.0, 0.0)
        west = (-1.0, 0.0)
        east = (1.0, 0.0)
        north = (1.0, 1.0)

        edge_list = [
            {
                "id": 0,
                "u": j,
                "v": west,
                "kind": "L",
                "geom": {"type": "LINE", "a": j, "b": west},
            },
            {
                "id": 1,
                "u": j,
                "v": east,
                "kind": "L",
                "geom": {"type": "LINE", "a": j, "b": east},
            },
            {
                "id": 2,
                "u": north,
                "v": j,
                "kind": "A",
                "geom": {
                    "type": "ARC",
                    "pts": [north, (0.6, 0.8), (0.3, 0.5), (0.1, 0.2), j],
                },
            },
        ]

        adj = collections.defaultdict(set)
        by_ends = collections.defaultdict(list)
        for e in edge_list:
            adj[e["u"]].add(e["v"])
            adj[e["v"]].add(e["u"])
            by_ends[(e["u"], e["v"])].append(e["id"])
            by_ends[(e["v"], e["u"])].append(e["id"])
        return edge_list, adj, by_ends, {}, j

    def test_solve_repairs_contradictory_branch_pattern(self):
        edge_list, adj, by_ends, station_nodes, junction = self._synthetic_junction_graph()

        # Baseline (all assign=0) has one incoming arc + two outgoing straight branches,
        # which violates smooth pass-through at the trunk pair.
        invalid_assign = {0: 0, 1: 0, 2: 0}
        self.assertEqual(count_nonholonomic_branch_violations(edge_list, adj, invalid_assign), 1)

        _, _, assign = solve(edge_list, adj, by_ends, station_nodes)

        self.assertEqual(count_nonholonomic_branch_violations(edge_list, adj, assign), 0)

        inc = [eid for eid in (0, 1, 2) if ((edge_list[eid]["u"], edge_list[eid]["v"]) if assign[eid] == 0 else (edge_list[eid]["v"], edge_list[eid]["u"]))[1] == junction]
        out = [eid for eid in (0, 1, 2) if ((edge_list[eid]["u"], edge_list[eid]["v"]) if assign[eid] == 0 else (edge_list[eid]["v"], edge_list[eid]["u"]))[0] == junction]
        self.assertTrue(len(inc) == 1 or len(out) == 1)
        self.assertFalse(violates_nonholonomic_branch_rule(junction, [0, 1, 2], edge_list, assign))

    def test_reward_penalizes_nonholonomic_violation_strongly(self):
        edge_list, adj, _, station_nodes, _ = self._synthetic_junction_graph()
        valid_assign = {0: 1, 1: 0, 2: 0}
        invalid_assign = {0: 0, 1: 0, 2: 0}

        valid_score = score_assignment_quality(edge_list, adj, station_nodes, valid_assign)
        invalid_score = score_assignment_quality(edge_list, adj, station_nodes, invalid_assign)

        self.assertLess(invalid_score, valid_score)
        self.assertGreaterEqual(valid_score - invalid_score, 20.0)


if __name__ == "__main__":
    unittest.main()

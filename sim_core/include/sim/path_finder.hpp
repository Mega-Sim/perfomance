#pragma once

#include "world.hpp"

#include <functional>
#include <vector>

namespace sim {

struct PathOptions {
  // edge_id -> allowed?
  std::function<bool(int)> edge_allowed;
  // edge_id -> traversal cost (default: edge.length)
  std::function<double(int)> edge_cost;
};

// Path result is ALWAYS a sequence of ORIGINAL edge_id (for Traffic/Execution).
// Internally, it uses an overlay graph (decision nodes) + A*.
class PathFinder {
public:
  explicit PathFinder(const World& w);

  // Find shortest path from start_node to goal_node.
  // start/goal can be any original node (station or not). Original graph is NOT modified.
  std::vector<int> find_path_edges_astar_overlay(int start_node, int goal_node,
                                                 const PathOptions& opt = {}) const;

private:
  struct OverlayEdge {
    int to = -1;
    double cost = 0.0;
    std::vector<int> orig_edges; // ORIGINAL edge_id chain
  };

  struct Pred {
    int prev_node = -1;
    int prev_oe_idx = -1; // index into overlay_out_[prev_node]
  };

  double heuristic_euclid(int node_id, int goal_id) const;

  void build_out_edges();
  void build_degrees();

  void build_overlay(int start_node, int goal_node,
                     const std::vector<unsigned char>& is_decision,
                     const PathOptions& opt,
                     std::vector<std::vector<OverlayEdge>>& overlay_out) const;

private:
  const World& w_;
  std::vector<std::vector<int>> out_edges_; // node_id -> outgoing edge_id list
  std::vector<int> in_deg_;
  std::vector<int> out_deg_;
};

} // namespace sim

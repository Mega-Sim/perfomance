#include "sim/path_finder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>

namespace sim {

static bool default_edge_allowed(int /*edge_id*/) { return true; }

PathFinder::PathFinder(const World& w) : w_(w) {
  build_out_edges();
  build_degrees();
}

void PathFinder::build_out_edges() {
  out_edges_.assign(w_.nodes.size(), {});
  for (int eid = 0; eid < (int)w_.edges.size(); ++eid) {
    const auto& e = w_.edges[eid];
    if (e.tail >= 0 && e.tail < (int)w_.nodes.size()) {
      out_edges_[e.tail].push_back(eid);
    }
  }
}

void PathFinder::build_degrees() {
  in_deg_.assign(w_.nodes.size(), 0);
  out_deg_.assign(w_.nodes.size(), 0);
  for (const auto& e : w_.edges) {
    if (e.tail >= 0 && e.tail < (int)w_.nodes.size()) out_deg_[e.tail]++;
    if (e.head >= 0 && e.head < (int)w_.nodes.size()) in_deg_[e.head]++;
  }
}

double PathFinder::heuristic_euclid(int node_id, int goal_id) const {
  const auto& a = w_.nodes[node_id];
  const auto& b = w_.nodes[goal_id];
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

void PathFinder::build_overlay(int start_node, int goal_node,
                               const std::vector<unsigned char>& is_decision,
                               const PathOptions& opt,
                               std::vector<std::vector<OverlayEdge>>& overlay_out) const {
  overlay_out.assign(w_.nodes.size(), {});

  const auto& edge_allowed = opt.edge_allowed ? opt.edge_allowed : default_edge_allowed;
  const auto& edge_cost = opt.edge_cost;

  auto cost_of = [&](int eid) -> double {
    return edge_cost ? edge_cost(eid) : w_.edges[eid].length;
  };

  // For every decision node u, follow each outgoing edge until next decision node.
  for (int u = 0; u < (int)w_.nodes.size(); ++u) {
    if (!is_decision[u]) continue;

    for (int eid0 : out_edges_[u]) {
      if (eid0 < 0 || eid0 >= (int)w_.edges.size()) continue;
      if (!edge_allowed(eid0)) continue;

      std::vector<int> chain;
      chain.reserve(16);
      double total = 0.0;

      int cur_eid = eid0;
      int cur_node = u;
      int v = -1;
      bool invalid = false;

      // cycle guard within this expansion
      std::unordered_set<int> seen;
      seen.reserve(64);
      seen.insert(u);

      while (true) {
        const auto& e = w_.edges[cur_eid];

        // sanity: ensure chain direction is consistent
        if (e.tail != cur_node) {
          invalid = true;
          break;
        }

        chain.push_back(cur_eid);
        total += cost_of(cur_eid);

        const int next_node = e.head;
        if (next_node < 0 || next_node >= (int)w_.nodes.size()) {
          invalid = true;
          break;
        }

        // stop condition: next decision node OR explicit endpoints
        if (is_decision[next_node] || next_node == start_node || next_node == goal_node) {
          v = next_node;
          break;
        }

        // linear node: (in==1 && out==1) is compressible for overlay
        const int indeg = in_deg_[next_node];
        const int outdeg = out_deg_[next_node];
        if (indeg == 1 && outdeg == 1) {
          if (out_edges_[next_node].size() != 1) {
            // inconsistent data. treat as decision endpoint.
            v = next_node;
            break;
          }
          if (seen.find(next_node) != seen.end()) {
            // pure 1-in-1-out cycle. stop here.
            v = next_node;
            break;
          }
          seen.insert(next_node);

          int next_eid = out_edges_[next_node][0];
          if (next_eid < 0 || next_eid >= (int)w_.edges.size()) {
            invalid = true;
            break;
          }
          if (!edge_allowed(next_eid)) {
            invalid = true;
            break;
          }

          cur_node = next_node;
          cur_eid = next_eid;
          continue;
        }

        // not linear => should have been decision; stop here.
        v = next_node;
        break;
      }

      if (invalid || v < 0) continue;

      OverlayEdge oe;
      oe.to = v;
      oe.cost = total;
      oe.orig_edges = std::move(chain);
      overlay_out[u].push_back(std::move(oe));
    }
  }
}

std::vector<int> PathFinder::find_path_edges_astar_overlay(int start_node, int goal_node,
                                                            const PathOptions& opt) const {
  if (start_node < 0 || goal_node < 0 ||
      start_node >= (int)w_.nodes.size() || goal_node >= (int)w_.nodes.size()) {
    return {};
  }
  if (start_node == goal_node) {
    return {};
  }

  // decision nodes: branch/merge/endpoints + start/goal
  std::vector<unsigned char> is_decision(w_.nodes.size(), 0);
  for (int n = 0; n < (int)w_.nodes.size(); ++n) {
    if (in_deg_[n] != 1 || out_deg_[n] != 1) is_decision[n] = 1;
  }
  is_decision[start_node] = 1;
  is_decision[goal_node] = 1;

  std::vector<std::vector<OverlayEdge>> overlay_out;
  build_overlay(start_node, goal_node, is_decision, opt, overlay_out);

  // A* on overlay graph
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> g(w_.nodes.size(), INF);
  std::vector<Pred> pred(w_.nodes.size());

  struct Q {
    double f;
    int node;
  };
  struct QLess {
    bool operator()(const Q& a, const Q& b) const { return a.f > b.f; }
  };
  std::priority_queue<Q, std::vector<Q>, QLess> pq;

  g[start_node] = 0.0;
  pq.push({heuristic_euclid(start_node, goal_node), start_node});

  while (!pq.empty()) {
    const Q cur = pq.top();
    pq.pop();

    const int u = cur.node;
    if (u == goal_node) break;
    if (overlay_out[u].empty()) continue;

    for (int idx = 0; idx < (int)overlay_out[u].size(); ++idx) {
      const auto& oe = overlay_out[u][idx];
      const int v = oe.to;
      if (v < 0 || v >= (int)w_.nodes.size()) continue;

      const double ng = g[u] + oe.cost;
      if (ng < g[v]) {
        g[v] = ng;
        pred[v] = Pred{u, idx};
        const double f = ng + heuristic_euclid(v, goal_node);
        pq.push({f, v});
      }
    }
  }

  if (!std::isfinite(g[goal_node])) {
    return {};
  }

  // reconstruct overlay path: goal -> start
  std::vector<std::pair<int, int>> overlay_steps; // (from_node, overlay_edge_idx)
  for (int cur = goal_node; cur != start_node;) {
    const auto& p = pred[cur];
    if (p.prev_node < 0 || p.prev_oe_idx < 0) {
      overlay_steps.clear();
      break;
    }
    overlay_steps.push_back({p.prev_node, p.prev_oe_idx});
    cur = p.prev_node;
  }
  if (overlay_steps.empty()) return {};
  std::reverse(overlay_steps.begin(), overlay_steps.end());

  // expand to original edge chain
  std::vector<int> route_edges;
  for (const auto& st : overlay_steps) {
    const int u = st.first;
    const int idx = st.second;
    const auto& oe = overlay_out[u][idx];
    route_edges.insert(route_edges.end(), oe.orig_edges.begin(), oe.orig_edges.end());
  }
  return route_edges;
}

} // namespace sim

#pragma once
#include <unordered_map>
#include <vector>

namespace sim {

struct Node {
  int id = -1;
  double x = 0.0;
  double y = 0.0;
};

struct Edge {
  int id = -1;
  int tail = -1;
  int head = -1;
  double length = 0.0;
};

struct Oht {
  int id = -1;
  std::vector<int> route_edges;
  size_t route_idx = 0;
  double v_mps = 1.0;
};

struct World {
  std::unordered_map<int, Node> nodes;
  std::unordered_map<int, Edge> edges;
  std::unordered_map<int, Oht>  ohts;
};

} // namespace sim

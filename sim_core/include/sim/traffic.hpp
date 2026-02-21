#pragma once
#include <cstddef>
#include <deque>
#include <vector>

namespace sim {

struct EdgeResv {
  int owner = -1;
  std::deque<int> waiters;
};

class Traffic {
public:
  void init(size_t edge_count) {
    edges_.clear();
    edges_.resize(edge_count);
  }

  bool try_reserve(int edge_id, int oht_id) {
    auto& e = edges_[edge_id];
    if (e.owner == -1) {
      e.owner = oht_id;
      return true;
    }
    if (e.owner != oht_id) e.waiters.push_back(oht_id);
    return false;
  }

  int release_and_pick_next(int edge_id, int oht_id) {
    auto& e = edges_[edge_id];
    if (e.owner != oht_id) return -1;

    if (!e.waiters.empty()) {
      int next = e.waiters.front();
      e.waiters.pop_front();
      e.owner = next;
      return next;
    }

    e.owner = -1;
    return -1;
  }

private:
  std::vector<EdgeResv> edges_;
};

} // namespace sim

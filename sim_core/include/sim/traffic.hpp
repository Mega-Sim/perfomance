#pragma once
#include <deque>
#include <unordered_map>

namespace sim {

struct EdgeResv {
  int owner = -1;
  std::deque<int> waiters;
};

class Traffic {
public:
  bool try_reserve(int edge_id, int oht_id) {
    auto& r = edges_[edge_id];
    if (r.owner == -1) { r.owner = oht_id; return true; }
    if (r.owner != oht_id) r.waiters.push_back(oht_id);
    return false;
  }

  int release_and_pick_next(int edge_id, int oht_id) {
    auto& r = edges_[edge_id];
    if (r.owner == oht_id) r.owner = -1;
    if (!r.waiters.empty()) {
      int next = r.waiters.front();
      r.waiters.pop_front();
      r.owner = next;
      return next;
    }
    return -1;
  }

private:
  std::unordered_map<int, EdgeResv> edges_;
};

} // namespace sim

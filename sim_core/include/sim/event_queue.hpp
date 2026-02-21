#pragma once
#include "event.hpp"
#include <queue>
#include <vector>

namespace sim {

struct EventLess {
  bool operator()(const Event& a, const Event& b) const {
    if (a.time_ns != b.time_ns) return a.time_ns > b.time_ns;
    return a.seq > b.seq;
  }
};

class EventQueue {
public:
  void push(const Event& e) { q_.push(e); }
  bool empty() const { return q_.empty(); }
  Event pop() { auto e = q_.top(); q_.pop(); return e; }

private:
  std::priority_queue<Event, std::vector<Event>, EventLess> q_;
};

} // namespace sim

#include "sim/engine.hpp"

#include <cmath>
#include <sstream>

namespace sim {

void Engine::run(int64_t end_time_ns) {
  traffic_.init(world_.edges.size());

  while (!eq_.empty()) {
    Event ev = eq_.pop();
    if (ev.time_ns > end_time_ns) break;
    now_ns_ = ev.time_ns;

    switch (ev.type) {
      case EventType::SpawnOht:         on_spawn(ev); break;
      case EventType::RequestEnterEdge: on_request_enter(ev); break;
      case EventType::EnterEdge:        on_enter_edge(ev); break;
      case EventType::ExitEdge:         on_exit_edge(ev); break;
      case EventType::ArriveStation:    on_arrive(ev); break;
    }
  }
}

void Engine::on_spawn(const Event& ev) {
  auto& o = world_.ohts[ev.oht_id];
  if (o.route_edges.empty()) return;

  int e0 = o.route_edges[0];
  push({now_ns_, next_seq_++, EventType::RequestEnterEdge, o.id, e0});

  std::ostringstream ss;
  ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"spawn\",\"oht\":" << o.id << "}";
  log_line(ss.str());
}

void Engine::on_request_enter(const Event& ev) {
  if (traffic_.try_reserve(ev.edge_id, ev.oht_id)) {
    push({now_ns_, next_seq_++, EventType::EnterEdge, ev.oht_id, ev.edge_id});
  }
}

void Engine::on_enter_edge(const Event& ev) {
  const auto& o = world_.ohts[ev.oht_id];
  const auto& e = world_.edges[ev.edge_id];

  double dt_s = e.length / o.v_mps;
  int64_t dt_ns = (int64_t)std::llround(dt_s * 1'000'000'000.0);

  push({now_ns_ + dt_ns, next_seq_++, EventType::ExitEdge, ev.oht_id, ev.edge_id});

  std::ostringstream ss;
  ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"enter\",\"oht\":" << o.id
     << ",\"edge\":" << e.id << ",\"tail\":" << e.tail << ",\"head\":" << e.head << "}";
  log_line(ss.str());
}

void Engine::on_exit_edge(const Event& ev) {
  auto& o = world_.ohts[ev.oht_id];
  const auto& e = world_.edges[ev.edge_id];

  int next_oht = traffic_.release_and_pick_next(ev.edge_id, ev.oht_id);
  if (next_oht != -1) {
    push({now_ns_, next_seq_++, EventType::EnterEdge, next_oht, ev.edge_id});
  }

  {
    std::ostringstream ss;
    ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"exit\",\"oht\":" << o.id
       << ",\"edge\":" << e.id << "}";
    log_line(ss.str());
  }

  o.route_idx++;
  if (o.route_idx >= o.route_edges.size()) {
    push({now_ns_, next_seq_++, EventType::ArriveStation, o.id, -1});
    return;
  }
  int next_edge = o.route_edges[o.route_idx];
  push({now_ns_, next_seq_++, EventType::RequestEnterEdge, o.id, next_edge});
}

void Engine::on_arrive(const Event& ev) {
  std::ostringstream ss;
  ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"arrive\",\"oht\":" << ev.oht_id << "}";
  log_line(ss.str());
}

} // namespace sim

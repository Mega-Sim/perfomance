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
    stats_.events_popped++;

    switch (ev.type) {
      case EventType::SpawnOht:         on_spawn(ev); break;
      case EventType::RequestEnterEdge: on_request_enter(ev); break;
      case EventType::EnterEdge:        on_enter_edge(ev); break;
      case EventType::ExitEdge:         on_exit_edge(ev); break;
      case EventType::ArriveStation:    on_arrive(ev); break;
    }

    maybe_dump_stats();
  }

  // 종료 시점에 마지막 통계 1회
  if (stats_output_) {
    // now_ns_가 0일 수도 있으니, 최소 1번은 찍고 싶으면 여기서 강제 출력 가능
    // 현재는 "다음 dump 시점이 이미 지났으면"만 출력하도록 maybe_dump_stats()에 맡김.
    // 필요하면 여기서 final summary를 찍을 수도 있음.
  }
}

void Engine::maybe_dump_stats() {
  if (!stats_output_) return;

  // 이벤트 시간 점프가 있을 수 있으니 while로 처리
  while (now_ns_ >= next_stat_dump_ns_) {
    const int64_t t_sec = next_stat_dump_ns_ / kStatIntervalNs;

    const uint64_t d_events = stats_.events_popped - last_dump_stats_.events_popped;
    const uint64_t d_enter  = stats_.enter_count - last_dump_stats_.enter_count;
    const uint64_t d_exit   = stats_.exit_count - last_dump_stats_.exit_count;
    const uint64_t d_wait   = stats_.wait_fail_count - last_dump_stats_.wait_fail_count;
    const uint64_t d_done   = stats_.completed_oht - last_dump_stats_.completed_oht;

    std::ostringstream ss;
    ss << "{"
       << "\"t_sec\":" << t_sec
       << ",\"events\":" << stats_.events_popped
       << ",\"enter\":" << stats_.enter_count
       << ",\"exit\":" << stats_.exit_count
       << ",\"wait\":" << stats_.wait_fail_count
       << ",\"done\":" << stats_.completed_oht
       << ",\"maxq\":" << stats_.max_queue_len
       << ",\"d_events\":" << d_events
       << ",\"d_enter\":" << d_enter
       << ",\"d_exit\":" << d_exit
       << ",\"d_wait\":" << d_wait
       << ",\"d_done\":" << d_done
       << "}";

    log_line(ss.str());

    last_dump_stats_ = stats_;
    next_stat_dump_ns_ += kStatIntervalNs;
  }
}

void Engine::on_spawn(const Event& ev) {
  auto& o = world_.ohts[ev.oht_id];
  if (o.route_edges.empty()) return;

  const int e0 = o.route_edges[0];
  push({now_ns_, next_seq_++, EventType::RequestEnterEdge, o.id, e0});

  if (verbose_events_) {
    std::ostringstream ss;
    ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"spawn\",\"oht\":" << o.id << "}";
    log_line(ss.str());
  }
}

void Engine::on_request_enter(const Event& ev) {
  if (traffic_.try_reserve(ev.edge_id, ev.oht_id)) {
    push({now_ns_, next_seq_++, EventType::EnterEdge, ev.oht_id, ev.edge_id});
    return;
  }

  // reserve 실패(대기열에 들어감)
  stats_.wait_fail_count++;
  const size_t qlen = traffic_.waiters_size(ev.edge_id);
  if (qlen > stats_.max_queue_len) stats_.max_queue_len = qlen;
}

void Engine::on_enter_edge(const Event& ev) {
  stats_.enter_count++;

  const auto& o = world_.ohts[ev.oht_id];
  const auto& e = world_.edges[ev.edge_id];

  const double dt_s = e.length / o.v_mps;
  const int64_t dt_ns = (int64_t)std::llround(dt_s * 1'000'000'000.0);

  push({now_ns_ + dt_ns, next_seq_++, EventType::ExitEdge, ev.oht_id, ev.edge_id});

  if (verbose_events_) {
    std::ostringstream ss;
    ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"enter\",\"oht\":" << o.id
       << ",\"edge\":" << e.id << ",\"tail\":" << e.tail << ",\"head\":" << e.head << "}";
    log_line(ss.str());
  }
}

void Engine::on_exit_edge(const Event& ev) {
  stats_.exit_count++;

  auto& o = world_.ohts[ev.oht_id];
  const auto& e = world_.edges[ev.edge_id];

  // 현재 엣지 해제 + 다음 대기자 즉시 진입 이벤트
  const int next_oht = traffic_.release_and_pick_next(ev.edge_id, ev.oht_id);
  if (next_oht != -1) {
    push({now_ns_, next_seq_++, EventType::EnterEdge, next_oht, ev.edge_id});
  }

  if (verbose_events_) {
    std::ostringstream ss;
    ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"exit\",\"oht\":" << o.id
       << ",\"edge\":" << e.id << "}";
    log_line(ss.str());
  }

  // 다음 edge로 진행
  o.route_idx++;
  if (o.route_idx >= o.route_edges.size()) {
    push({now_ns_, next_seq_++, EventType::ArriveStation, o.id, -1});
    return;
  }

  const int next_edge = o.route_edges[o.route_idx];
  push({now_ns_, next_seq_++, EventType::RequestEnterEdge, o.id, next_edge});
}

void Engine::on_arrive(const Event& ev) {
  stats_.completed_oht++;

  if (verbose_events_) {
    std::ostringstream ss;
    ss << "{\"t_ns\":" << now_ns_ << ",\"type\":\"arrive\",\"oht\":" << ev.oht_id << "}";
    log_line(ss.str());
  }
}

} // namespace sim

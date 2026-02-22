#pragma once
#include "event_queue.hpp"
#include "traffic.hpp"
#include "world.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

namespace sim {

class Engine {
public:
  struct Stats {
    uint64_t events_popped = 0;
    uint64_t enter_count = 0;
    uint64_t exit_count = 0;
    uint64_t wait_fail_count = 0;
    uint64_t completed_oht = 0;
    size_t max_queue_len = 0;
  };

public:
  explicit Engine(World w) : world_(std::move(w)) {}

  void set_log_path(const std::string& path) {
    log_.open(path, std::ios::out | std::ios::trunc);
  }

  // 통계 JSONL 1줄/초 출력 여부 (기본 ON)
  void set_stats_output(bool enable) { stats_output_ = enable; }

  // 기존 enter/exit 이벤트 JSONL을 남기고 싶으면 true로 (기본 OFF)
  void set_verbose_events(bool enable) { verbose_events_ = enable; }

  const Stats& stats() const { return stats_; }

  void schedule_spawn(int64_t t_ns, int oht_id) {
    push({t_ns, next_seq_++, EventType::SpawnOht, oht_id, -1});
  }

  void run(int64_t end_time_ns);

private:
  void push(const Event& e) { eq_.push(e); }

  void on_spawn(const Event& ev);
  void on_request_enter(const Event& ev);
  void on_enter_edge(const Event& ev);
  void on_exit_edge(const Event& ev);
  void on_arrive(const Event& ev);

  void maybe_dump_stats();

  void log_line(const std::string& s) {
    if (log_.is_open()) log_ << s << "\n";
  }

private:
  static constexpr int64_t kStatIntervalNs = 1'000'000'000LL; // 1초

private:
  int64_t now_ns_ = 0;
  uint64_t next_seq_ = 1;

  World world_;
  EventQueue eq_;
  Traffic traffic_;
  std::ofstream log_;

  Stats stats_;
  Stats last_dump_stats_;
  int64_t next_stat_dump_ns_ = kStatIntervalNs;

  bool stats_output_ = true;
  bool verbose_events_ = false;
};

} // namespace sim

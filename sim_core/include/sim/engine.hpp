#pragma once
#include "event_queue.hpp"
#include "traffic.hpp"
#include "world.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

namespace sim {

class Engine {
public:
  explicit Engine(World w) : world_(std::move(w)) {}

  void set_log_path(const std::string& path) {
    log_.open(path, std::ios::out | std::ios::trunc);
  }

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

  void log_line(const std::string& s) {
    if (log_.is_open()) log_ << s << "\n";
  }

private:
  int64_t now_ns_ = 0;
  uint64_t next_seq_ = 1;

  World world_;
  EventQueue eq_;
  Traffic traffic_;
  std::ofstream log_;
};

} // namespace sim

#pragma once
#include <cstdint>

namespace sim {

enum class EventType : uint8_t {
  SpawnOht,
  RequestEnterEdge,
  EnterEdge,
  ExitEdge,
  ArriveStation,
};

struct Event {
  int64_t  time_ns = 0;
  uint64_t seq     = 0;
  EventType type   = EventType::SpawnOht;

  int oht_id  = -1;
  int edge_id = -1;
};

} // namespace sim

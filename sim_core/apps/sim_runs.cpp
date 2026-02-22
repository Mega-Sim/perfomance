#include "sim/engine.hpp"
#include "sim/path_finder.hpp"
#include "sim/world.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "json.hpp"
using json = nlohmann::json;

struct StationInfo {
  int node_id = -1;
  double x = 0.0;
  double y = 0.0;
};

static void print_usage() {
  std::cerr
      << "usage:\n"
      << "  sim_run <out.json> <out.jsonl> --start <station> --end <station>\n"
      << "  sim_run <out.json> <out.jsonl> --start-node <id> --end-node <id>\n";
}

static sim::World load_world(const std::string& path,
                             std::unordered_map<std::string, StationInfo>& stations_out) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open json: " + path);

  json j;
  ifs >> j;

  sim::World w;

  // nodes
  w.nodes.resize(j.at("nodes").size());
  for (const auto& n : j.at("nodes")) {
    const int id = n.at("id").get<int>();
    if ((size_t)id >= w.nodes.size()) w.nodes.resize((size_t)id + 1);
    w.nodes[id] = sim::Node{id, n.at("x").get<double>(), n.at("y").get<double>()};
  }

  // edges
  w.edges.resize(j.at("edges").size());
  for (const auto& e : j.at("edges")) {
    const int id = e.at("id").get<int>();
    if ((size_t)id >= w.edges.size()) w.edges.resize((size_t)id + 1);
    w.edges[id] = sim::Edge{id,
                            e.at("tail").get<int>(),
                            e.at("head").get<int>(),
                            e.at("length").get<double>()};
  }

  // stations (optional)
  stations_out.clear();
  if (j.contains("stations") && j.at("stations").is_object()) {
    for (auto it = j["stations"].begin(); it != j["stations"].end(); ++it) {
      const std::string name = it.key();
      const auto& s = it.value();
      StationInfo info;
      info.node_id = s.at("node_id").get<int>();
      info.x = s.at("x").get<double>();
      info.y = s.at("y").get<double>();
      stations_out.emplace(name, info);
    }
  }

  return w;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage();
    return 2;
  }

  const std::string in_json = argv[1];
  const std::string out_log = argv[2];

  std::string start_station, end_station;
  int start_node = -1;
  int end_node = -1;

  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* opt) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << opt << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };

    if (a == "--start")
      start_station = need("--start");
    else if (a == "--end")
      end_station = need("--end");
    else if (a == "--start-node")
      start_node = std::stoi(need("--start-node"));
    else if (a == "--end-node")
      end_node = std::stoi(need("--end-node"));
    else {
      std::cerr << "unknown option: " << a << "\n";
      print_usage();
      return 2;
    }
  }

  std::unordered_map<std::string, StationInfo> stations;
  sim::World w = load_world(in_json, stations);

  if (start_node < 0 && !start_station.empty()) {
    auto it = stations.find(start_station);
    if (it != stations.end()) start_node = it->second.node_id;
  }
  if (end_node < 0 && !end_station.empty()) {
    auto it = stations.find(end_station);
    if (it != stations.end()) end_node = it->second.node_id;
  }

  if (start_node < 0 || end_node < 0) {
    std::cerr << "start/end not resolved. (stations in json: " << stations.size() << ")\n";
    print_usage();
    return 2;
  }

  // A* on overlay(decision nodes), output is original edge_id list.
  sim::PathFinder pf(w);
  std::vector<int> route = pf.find_path_edges_astar_overlay(start_node, end_node);
  if (route.empty() && start_node != end_node) {
    std::cerr << "no path found: start_node=" << start_node << " end_node=" << end_node << "\n";
    return 3;
  }

  sim::Oht o;
  o.id = 0;
  o.v_mps = 1.0;
  o.route_edges = std::move(route);

  w.ohts.resize(1);
  w.ohts[0] = std::move(o);

  sim::Engine eng(std::move(w));
  eng.set_log_path(out_log);
  eng.schedule_spawn(0, 0);
  eng.run(60LL * 1'000'000'000LL);
  return 0;
}

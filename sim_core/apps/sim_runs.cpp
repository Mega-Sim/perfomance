#include "sim/engine.hpp"
#include "sim/path_finder.hpp"
#include "sim/world.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
      << "  sim_run <world.json> <out.stats.jsonl> [options]\n"
      << "\n"
      << "single path (optional):\n"
      << "  --start <station> --end <station>\n"
      << "  --start-node <id> --end-node <id>\n"
      << "\n"
      << "multi vehicles:\n"
      << "  --vehicles N            (default: 1)\n"
      << "  --seed S                (default: 1)\n"
      << "  --sim-sec T             (default: 60)\n"
      << "  --spawn-interval-ms K   (default: 0)\n";
}

static sim::World load_world(const std::string& path,
                            std::unordered_map<std::string, StationInfo>& stations_out) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open json: " + path);

  json j;
  ifs >> j;

  sim::World w;

  w.nodes.resize(j.at("nodes").size());
  for (const auto& n : j.at("nodes")) {
    const int id = n.at("id").get<int>();
    if ((size_t)id >= w.nodes.size()) w.nodes.resize((size_t)id + 1);
    w.nodes[id] = sim::Node{id, n.at("x").get<double>(), n.at("y").get<double>()};
  }

  w.edges.resize(j.at("edges").size());
  for (const auto& e : j.at("edges")) {
    const int id = e.at("id").get<int>();
    if ((size_t)id >= w.edges.size()) w.edges.resize((size_t)id + 1);
    w.edges[id] = sim::Edge{id,
                            e.at("tail").get<int>(),
                            e.at("head").get<int>(),
                            e.at("length").get<double>()};
  }

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

static void build_degrees(const sim::World& w, std::vector<int>& in_deg, std::vector<int>& out_deg) {
  in_deg.assign(w.nodes.size(), 0);
  out_deg.assign(w.nodes.size(), 0);
  for (const auto& e : w.edges) {
    if (e.tail >= 0 && e.tail < (int)w.nodes.size()) out_deg[e.tail]++;
    if (e.head >= 0 && e.head < (int)w.nodes.size()) in_deg[e.head]++;
  }
}

static std::vector<int> build_candidate_nodes(
    const sim::World& w,
    const std::unordered_map<std::string, StationInfo>& stations) {
  std::vector<int> in_deg, out_deg;
  build_degrees(w, in_deg, out_deg);

  std::vector<unsigned char> mark(w.nodes.size(), 0);

  for (int n = 0; n < (int)w.nodes.size(); ++n) {
    if (in_deg[n] != 1 || out_deg[n] != 1) mark[n] = 1;
  }

  for (const auto& kv : stations) {
    const int nid = kv.second.node_id;
    if (nid >= 0 && nid < (int)mark.size()) mark[nid] = 1;
  }

  std::vector<int> cand;
  cand.reserve(w.nodes.size());
  for (int n = 0; n < (int)mark.size(); ++n) {
    if (mark[n]) cand.push_back(n);
  }

  if (cand.size() < 4) {
    cand.clear();
    cand.reserve(w.nodes.size());
    for (int n = 0; n < (int)w.nodes.size(); ++n) cand.push_back(n);
  }

  return cand;
}

static int pick_node(std::mt19937_64& rng, const std::vector<int>& cand) {
  std::uniform_int_distribution<size_t> dist(0, cand.size() - 1);
  return cand[dist(rng)];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage();
    return 2;
  }

  const std::string in_json = argv[1];
  const std::string out_log = argv[2];

  std::string start_station;
  std::string end_station;
  int start_node = -1;
  int end_node = -1;

  int vehicles = 1;
  uint64_t seed = 1;
  int sim_sec = 60;
  int spawn_interval_ms = 0;

  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];

    auto need = [&](const char* opt) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << opt << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };

    if (a == "--start") start_station = need("--start");
    else if (a == "--end") end_station = need("--end");
    else if (a == "--start-node") start_node = std::stoi(need("--start-node"));
    else if (a == "--end-node") end_node = std::stoi(need("--end-node"));
    else if (a == "--vehicles") vehicles = std::stoi(need("--vehicles"));
    else if (a == "--seed") seed = (uint64_t)std::stoull(need("--seed"));
    else if (a == "--sim-sec") sim_sec = std::stoi(need("--sim-sec"));
    else if (a == "--spawn-interval-ms") spawn_interval_ms = std::stoi(need("--spawn-interval-ms"));
    else {
      std::cerr << "unknown option: " << a << "\n";
      print_usage();
      return 2;
    }
  }

  if (vehicles <= 0) {
    std::cerr << "--vehicles must be > 0\n";
    return 2;
  }
  if (sim_sec <= 0) {
    std::cerr << "--sim-sec must be > 0\n";
    return 2;
  }
  if (spawn_interval_ms < 0) {
    std::cerr << "--spawn-interval-ms must be >= 0\n";
    return 2;
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

  sim::PathFinder pf(w);

  std::vector<int> cand_nodes = build_candidate_nodes(w, stations);
  std::mt19937_64 rng(seed);

  w.ohts.clear();
  w.ohts.resize((size_t)vehicles);

  for (int i = 0; i < vehicles; ++i) {
    int s = start_node;
    int t = end_node;

    if (vehicles > 1 || (start_node < 0 || end_node < 0)) {
      do {
        s = pick_node(rng, cand_nodes);
        t = pick_node(rng, cand_nodes);
      } while (s == t);
    } else {
      if (s < 0 || t < 0) {
        std::cerr << "start/end not resolved.\n";
        print_usage();
        return 2;
      }
    }

    std::vector<int> route = pf.find_path_edges_astar_overlay(s, t);
    if (route.empty() && s != t) {
      bool ok = false;
      for (int retry = 0; retry < 8 && !ok; ++retry) {
        int rs = pick_node(rng, cand_nodes);
        int rt = pick_node(rng, cand_nodes);
        if (rs == rt) continue;
        route = pf.find_path_edges_astar_overlay(rs, rt);
        if (!route.empty()) {
          ok = true;
        }
      }
      if (!ok) {
        for (int rs : cand_nodes) {
          for (int rt : cand_nodes) {
            if (rs == rt) continue;
            route = pf.find_path_edges_astar_overlay(rs, rt);
            if (!route.empty()) {
              ok = true;
              break;
            }
          }
          if (ok) break;
        }
      }
      if (!ok) {
        std::cerr << "no path for oht " << i << " (graph disconnected?)\n";
        return 3;
      }
    }

    sim::Oht o;
    o.id = i;
    o.v_mps = 1.0;
    o.route_edges = std::move(route);
    o.route_idx = 0;
    w.ohts[(size_t)i] = std::move(o);
  }

  sim::Engine eng(std::move(w));
  eng.set_log_path(out_log);
  eng.set_stats_output(true);
  eng.set_verbose_events(false);

  const int64_t spawn_dt_ns = (int64_t)spawn_interval_ms * 1'000'000LL;
  for (int i = 0; i < vehicles; ++i) {
    const int64_t t_ns = (spawn_dt_ns == 0) ? 0 : (int64_t)i * spawn_dt_ns;
    eng.schedule_spawn(t_ns, i);
  }

  eng.run((int64_t)sim_sec * 1'000'000'000LL);
  return 0;
}

#include "sim/engine.hpp"
#include "sim/world.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

static sim::World load_world(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open json: " + path);

  json j;
  ifs >> j;

  sim::World w;

  for (const auto& n : j.at("nodes")) {
    sim::Node node;
    node.id = n.at("id").get<int>();
    node.x  = n.at("x").get<double>();
    node.y  = n.at("y").get<double>();
    w.nodes[node.id] = node;
  }

  for (const auto& e : j.at("edges")) {
    sim::Edge edge;
    edge.id     = e.at("id").get<int>();
    edge.tail   = e.at("tail").get<int>();
    edge.head   = e.at("head").get<int>();
    edge.length = e.at("length").get<double>();
    w.edges[edge.id] = edge;
  }

  return w;
}

static std::vector<int> make_route_linear(const sim::World& w) {
  std::vector<int> r;
  r.reserve(w.edges.size());
  for (int id = 0; id < (int)w.edges.size(); ++id) {
    if (w.edges.find(id) != w.edges.end()) r.push_back(id);
  }
  return r;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: sim_run <out.json> <out.jsonl>\n";
    return 2;
  }

  sim::World w = load_world(argv[1]);

  sim::Oht o;
  o.id = 0;
  o.v_mps = 1.0;
  o.route_edges = make_route_linear(w);
  w.ohts[o.id] = o;

  sim::Engine eng(std::move(w));
  eng.set_log_path(argv[2]);
  eng.schedule_spawn(0, 0);
  eng.run(60LL * 1'000'000'000LL);

  return 0;
}

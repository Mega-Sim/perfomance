#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outer-loop CW 고정 + 분기(특히 45° diagonal/N-branch) 방향 전파/제약 기반 단방향 그래프 생성기
(✅ DXF의 ARC는 '직선 chord'로 대체하지 않고, 원래 ARC 형상으로 렌더링/덤프)

사용:
  python build_directed_graph_outer_cw_propagate_curves.py Drawing1.dxf directed_graph.png [graph_dump.json]
"""
from __future__ import annotations
import math, sys, json, collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PREC = 3              # node 좌표 스냅(표시/노드키)
EPS_ON = 1e-2         # arc endpoint가 line 위에 얹히면 split
STATION_PROJ_EPS = 300
ARC_SAMPLE_N = 64

def nk(p):
    return (round(p[0], PREC), round(p[1], PREC))

def parse_dxf(path: Path):
    lines=[]; arcs=[]; texts=[]
    c = path.read_text(errors="ignore").splitlines()
    i=0; in_entities=False; cur=None; ent={}
    def flush():
        nonlocal cur, ent
        if cur=="LINE" and ent: lines.append(ent.copy())
        elif cur=="ARC" and ent: arcs.append(ent.copy())
        elif cur=="TEXT" and ent: texts.append(ent.copy())
        ent={}
    while i < len(c)-1:
        code=c[i].strip(); val=c[i+1].strip(); i+=2
        if code=="0" and val=="SECTION":
            if i+1 < len(c) and c[i].strip()=="2" and c[i+1].strip()=="ENTITIES":
                in_entities=True
        if not in_entities:
            continue
        if code=="0":
            if val in ("LINE","ARC","TEXT"):
                flush(); cur=val; ent={}
            else:
                flush(); cur=None; ent={}
        if cur=="LINE":
            if code=="10": ent["x1"]=float(val)
            elif code=="20": ent["y1"]=float(val)
            elif code=="11": ent["x2"]=float(val)
            elif code=="21": ent["y2"]=float(val)
        elif cur=="ARC":
            if code=="10": ent["cx"]=float(val)
            elif code=="20": ent["cy"]=float(val)
            elif code=="40": ent["r"]=float(val)
            elif code=="50": ent["a0"]=float(val)
            elif code=="51": ent["a1"]=float(val)
        elif cur=="TEXT":
            if code=="10": ent["x"]=float(val)
            elif code=="20": ent["y"]=float(val)
            elif code=="1": ent["t"]=val
    flush()
    return lines, arcs, texts

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def seg_len(p,q):
    return dist(p,q)

def point_on_segment(p,a,b,eps=EPS_ON):
    # p가 선분 ab 위에 있는지 (거리 + 범위)
    ax,ay=a; bx,by=b; px,py=p
    vx=bx-ax; vy=by-ay
    wx=px-ax; wy=py-ay
    vv=vx*vx+vy*vy
    if vv==0: return False
    t=(wx*vx+wy*vy)/vv
    if t<0-eps or t>1+eps: return False
    cx=ax+t*vx; cy=ay+t*vy
    return dist((px,py),(cx,cy))<=eps

def arc_endpoints(arc):
    cx,cy,r,a0,a1 = arc["cx"],arc["cy"],arc["r"],math.radians(arc["a0"]),math.radians(arc["a1"])
    p0=(cx+r*math.cos(a0), cy+r*math.sin(a0))
    p1=(cx+r*math.cos(a1), cy+r*math.sin(a1))
    return p0,p1

def sample_arc_points(arc, n=ARC_SAMPLE_N):
    cx,cy,r=arc["cx"],arc["cy"],arc["r"]
    a0=math.radians(arc["a0"]); a1=math.radians(arc["a1"])
    # DXF ARC는 CCW로 a0->a1; a1<a0면 wrap
    if a1 < a0:
        a1 += 2*math.pi
    ts=np.linspace(a0,a1,n)
    pts=[(cx+r*math.cos(t), cy+r*math.sin(t)) for t in ts]
    return pts

def polyline_len(pts):
    s=0.0
    for i in range(len(pts)-1):
        s += dist(pts[i], pts[i+1])
    return s

def segment_intersection(a,b,c,d):
    # 간단한 2D 선분 교차(끝점 제외 포함)
    def ccw(p,q,r):
        return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
    return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))

def build_graph(lines, arcs, texts):
    # 1) station nodes: TEXT 중 "station" 포함
    station_nodes=[]
    for t in texts:
        s=t.get("t","").lower()
        if "station" in s:
            station_nodes.append((t["x"], t["y"], t["t"]))

    # 2) split points: 선분/호 endpoints + 교차점 + station projection
    #    - simplify: line-line 교차만 처리 (ARC는 endpoint 연결만)
    pts=set()
    for ln in lines:
        pts.add(nk((ln["x1"],ln["y1"])))
        pts.add(nk((ln["x2"],ln["y2"])))
    for ac in arcs:
        p0,p1=arc_endpoints(ac)
        pts.add(nk(p0)); pts.add(nk(p1))

    # line-line intersections (O(n^2) ok for phase1)
    lsegs=[((ln["x1"],ln["y1"]),(ln["x2"],ln["y2"])) for ln in lines]
    for i in range(len(lsegs)):
        a,b=lsegs[i]
        for j in range(i+1,len(lsegs)):
            c,d=lsegs[j]
            if segment_intersection(a,b,c,d):
                # compute intersection point (robust enough)
                x1,y1=a; x2,y2=b; x3,y3=c; x4,y4=d
                den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                if abs(den) < 1e-9:
                    continue
                px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
                py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
                pts.add(nk((px,py)))

    # station projection: nearest segment endpoint or on-segment projection to snap to graph
    # (간단 버전) station point 자체를 node로 추가
    for sx,sy,name in station_nodes:
        pts.add(nk((sx,sy)))

    # 3) adjacency: split each LINE by all points on it
    adj=collections.defaultdict(set)
    edge_list=[]
    by_ends=collections.defaultdict(list)  # for quick search
    eid=0

    def add_edge(u,v,kind="L", geom=None):
        nonlocal eid
        if u==v: return
        edge={"id":eid,"u":u,"v":v,"kind":kind}
        if geom is not None:
            edge["geom"]=geom
        edge_list.append(edge)
        adj[u].add(v); adj[v].add(u)
        by_ends[(u,v)].append(eid); by_ends[(v,u)].append(eid)
        eid+=1

    pts_list=[(p[0],p[1]) for p in pts]

    # helper: points on line segment
    for ln in lines:
        a=(ln["x1"],ln["y1"]); b=(ln["x2"],ln["y2"])
        on=[]
        for p in pts_list:
            if point_on_segment(p,a,b,eps=1e-1):  # DXF scale tolerance
                on.append(p)
        # sort along t
        ax,ay=a; bx,by=b
        vx=bx-ax; vy=by-ay
        vv=vx*vx+vy*vy if (vx*vx+vy*vy)!=0 else 1.0
        on.sort(key=lambda p: ((p[0]-ax)*vx+(p[1]-ay)*vy)/vv)
        for i in range(len(on)-1):
            u=nk(on[i]); v=nk(on[i+1])
            add_edge(u,v,kind="L",geom={"type":"LINE","a":on[i],"b":on[i+1]})

    # 4) ARC: endpoints connect as single curved edge (no splitting by intersections here)
    for ac in arcs:
        p0,p1=arc_endpoints(ac)
        u=nk(p0); v=nk(p1)
        pts_curve=sample_arc_points(ac, n=ARC_SAMPLE_N)
        add_edge(u,v,kind="A",geom={"type":"ARC","cx":ac["cx"],"cy":ac["cy"],"r":ac["r"],
                                    "a0":ac["a0"],"a1":ac["a1"],"pts":pts_curve})

    # station mapping: nearest node id (same as station point snap)
    station_map={}
    for sx,sy,name in station_nodes:
        station_map[name]=nk((sx,sy))

    return edge_list, adj, by_ends, station_map

def angle(p,q,r):
    # angle at q from qp to qr in [-pi,pi]
    a=(p[0]-q[0], p[1]-q[1]); b=(r[0]-q[0], r[1]-q[1])
    aa=math.hypot(*a); bb=math.hypot(*b)
    if aa*bb==0: return 0.0
    dot=(a[0]*b[0]+a[1]*b[1])/(aa*bb)
    dot=max(-1,min(1,dot))
    cross=a[0]*b[1]-a[1]*b[0]
    return math.atan2(cross, dot)

def polygon_area(poly):
    # signed area
    s=0.0
    for i in range(len(poly)):
        x1,y1=poly[i]
        x2,y2=poly[(i+1)%len(poly)]
        s += x1*y2-x2*y1
    return 0.5*s


def _normalize(vec):
    x, y = vec
    n = math.hypot(x, y)
    if n == 0:
        return (0.0, 0.0)
    return (x / n, y / n)


def tangent_angle_deg(vec_a, vec_b):
    """Smallest angle in degrees between two tangent vectors."""
    ax, ay = _normalize(vec_a)
    bx, by = _normalize(vec_b)
    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    return math.degrees(math.acos(dot))


def edge_tangent_at_node(edge, node):
    """Return tangent vector pointing away from `node` along `edge` geometry."""
    geom = edge.get("geom", {})
    u = edge["u"]
    v = edge["v"]
    if node != u and node != v:
        raise ValueError(f"node {node} is not an endpoint of edge {edge.get('id')}")

    if geom.get("type") == "LINE":
        if node == u:
            return (v[0] - u[0], v[1] - u[1])
        return (u[0] - v[0], u[1] - v[1])

    if geom.get("type") == "ARC":
        pts = geom.get("pts", [])
        if len(pts) < 2:
            return (0.0, 0.0)

        d0 = dist(node, pts[0])
        d1 = dist(node, pts[-1])
        if d0 <= d1:
            # node is near first point; tangent points to the next sample.
            return (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        # node is near last point; tangent points backward into the curve.
        return (pts[-2][0] - pts[-1][0], pts[-2][1] - pts[-1][1])

    # fallback: endpoint chord direction
    if node == u:
        return (v[0] - u[0], v[1] - u[1])
    return (u[0] - v[0], u[1] - v[1])


def is_smooth_pair(edge_a, edge_b, node, smooth_thresh_deg=35.0):
    """True when two edges form near-colinear continuation through `node`."""
    ta = edge_tangent_at_node(edge_a, node)
    tb = edge_tangent_at_node(edge_b, node)
    # For pass-through, away-from-node tangents should be near opposite directions.
    return tangent_angle_deg(ta, tb) >= (180.0 - smooth_thresh_deg)


def classify_node_flow_pattern(node, incident_eids, edge_list, assign):
    incoming = []
    outgoing = []
    for eid in incident_eids:
        e = edge_list[eid]
        src, dst = (e["u"], e["v"]) if assign[eid] == 0 else (e["v"], e["u"])
        if dst == node:
            incoming.append(eid)
        elif src == node:
            outgoing.append(eid)

    smooth_pairs = []
    for i in range(len(incident_eids)):
        for j in range(i + 1, len(incident_eids)):
            ea = edge_list[incident_eids[i]]
            eb = edge_list[incident_eids[j]]
            if is_smooth_pair(ea, eb, node):
                smooth_pairs.append((incident_eids[i], incident_eids[j]))

    return {
        "incoming": incoming,
        "outgoing": outgoing,
        "smooth_pairs": smooth_pairs,
        "degree": len(incident_eids),
    }


def violates_nonholonomic_branch_rule(node, incident_eids, edge_list, assign):
    """Local physical validity check:
    - degree-2 smooth continuation must be one-in/one-out
    - degree>=3 node must be representable as split/merge
    - smooth pair at a branch must preserve continuous heading
    """
    pattern = classify_node_flow_pattern(node, incident_eids, edge_list, assign)
    n_in = len(pattern["incoming"])
    n_out = len(pattern["outgoing"])
    deg = pattern["degree"]

    if deg <= 1:
        return False

    if deg == 2 and pattern["smooth_pairs"]:
        return not (n_in == 1 and n_out == 1)

    if deg >= 3:
        # Must be split (1 in, N out) or merge (N in, 1 out)
        if not (n_in == 1 or n_out == 1):
            return True
        # Smooth continuation pair should be pass-through (one in, one out).
        for a, b in pattern["smooth_pairs"]:
            a_is_in = a in pattern["incoming"]
            b_is_in = b in pattern["incoming"]
            if a_is_in == b_is_in:
                return True

    return False


def count_nonholonomic_branch_violations(edge_list, adj, assign):
    inc = collections.defaultdict(list)
    for e in edge_list:
        inc[e["u"]].append(e["id"])
        inc[e["v"]].append(e["id"])
    n_bad = 0
    for node in adj.keys():
        if violates_nonholonomic_branch_rule(node, inc[node], edge_list, assign):
            n_bad += 1
    return n_bad


def _count_dead_ends(edge_list, adj, assign):
    incoming = collections.defaultdict(int)
    outgoing = collections.defaultdict(int)
    for e in edge_list:
        src, dst = (e["u"], e["v"]) if assign[e["id"]] == 0 else (e["v"], e["u"])
        outgoing[src] += 1
        incoming[dst] += 1
    n_dead = 0
    for node in adj.keys():
        if incoming[node] == 0 or outgoing[node] == 0:
            n_dead += 1
    return n_dead


def _count_station_reachability_issues(edge_list, station_nodes, assign):
    stations = list(station_nodes.values())
    if len(stations) <= 1:
        return 0
    directed = collections.defaultdict(set)
    for e in edge_list:
        src, dst = (e["u"], e["v"]) if assign[e["id"]] == 0 else (e["v"], e["u"])
        directed[src].add(dst)

    def bfs(start):
        seen = {start}
        q = collections.deque([start])
        while q:
            cur = q.popleft()
            for nxt in directed.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    issues = 0
    for s in stations:
        seen = bfs(s)
        if any(t not in seen for t in stations if t != s):
            issues += 1
    return issues

def find_outer_loop(adj):
    # 매우 단순한 outer loop 추정: (x+y) 최대 노드에서 시작하여 left-hand rule로 cycle 탐색
    nodes=list(adj.keys())
    start=max(nodes, key=lambda n:(n[0],n[1]))
    # pick initial neighbor with smallest angle from +x axis
    neigh=list(adj[start])
    if not neigh:
        return []
    # choose neighbor with smallest atan2
    neigh.sort(key=lambda v: math.atan2(v[1]-start[1], v[0]-start[0]))
    cur=start; prev=neigh[0]
    loop=[start]
    for _ in range(20000):
        loop.append(prev)
        # choose next at prev with maximal right turn (CW)
        cand=list(adj[prev])
        if cur in cand:
            cand.remove(cur)
        if not cand:
            break
        # current direction: prev->cur ? actually we traveled cur->prev
        # want choose next to make CW hugging outside => choose smallest signed angle (right turn)
        def score(nxt):
            # angle from (cur->prev) to (prev->nxt)
            a=(prev[0]-cur[0], prev[1]-cur[1])
            b=(nxt[0]-prev[0], nxt[1]-prev[1])
            aa=math.atan2(a[1],a[0])
            bb=math.atan2(b[1],b[0])
            d=bb-aa
            while d<=-math.pi: d+=2*math.pi
            while d> math.pi: d-=2*math.pi
            return d
        cand.sort(key=score) # most negative first (right)
        nxt=cand[0]
        cur,prev=prev,nxt
        if prev==start:
            break
        if len(loop)>3 and prev in loop[:-1]:
            break
    # ensure cycle
    if loop and loop[-1]==start:
        loop=loop[:-1]
    return loop

def solve(edge_list, adj, by_ends, station_nodes):
    # edge 방향 assign: 0=as-is (u->v), 1=flip (v->u)
    # We build undirected edges list but need directed edges for sim.
    # 규칙:
    #  - 외곽 loop는 CW 방향 고정
    #  - 분기점에서는 일관되게 방향 전파
    #  - station은 경로 가능하도록 연결 강제 (간단: station node degree>=1 유지)
    # 간단화: greedy propagation

    assign={e["id"]:0 for e in edge_list}

    # 1) outer loop 추정 및 CW로 맞추기
    loop=find_outer_loop(adj)
    bits={}
    if len(loop)>=3:
        # compute signed area; CW면 area<0 (좌표계에 따라 다를 수 있음)
        area=polygon_area(loop)
        cw = area < 0
        # want CW True. if not cw, reverse loop traversal
        if not cw:
            loop=list(reversed(loop))
        # apply direction along loop: n[i] -> n[i+1]
        for i in range(len(loop)):
            u=loop[i]; v=loop[(i+1)%len(loop)]
            # find edge id connecting u-v (could be multiple; pick first)
            eids = by_ends.get((u,v), [])
            if not eids:
                continue
            eid=eids[0]
            e=edge_list[eid]
            # e stores u,v endpoints in undirected meaning; assign 0 means e.u->e.v
            if e["u"]==u and e["v"]==v:
                assign[eid]=0
            elif e["u"]==v and e["v"]==u:
                assign[eid]=1
            bits[eid]=1

    # 2) propagate from already-directed edges outward with BFS
    #    If a node has one incoming and remaining undirected edges, set them outgoing etc.
    #    We define for each node: outgoing/incoming based on assigned edges.
    def edge_dir(eid):
        e=edge_list[eid]
        if assign[eid]==0: return e["u"],e["v"]
        else: return e["v"],e["u"]

    # map node -> incident edge ids
    inc=collections.defaultdict(list)
    for e in edge_list:
        inc[e["u"]].append(e["id"])
        inc[e["v"]].append(e["id"])

    q=collections.deque()
    for n in inc:
        q.append(n)

    changed=True
    it=0
    while changed and it<20000:
        it+=1
        changed=False
        for _ in range(len(q)):
            n=q.popleft()
            eids=inc[n]
            # classify already oriented wrt n
            out=[]; inn=[]; und=[]
            for eid in eids:
                u,v=edge_dir(eid)
                # if eid is directed? always directed by assign, but some are still "unknown" conceptually
                # We treat unknown as those not in bits map and not yet visited; but assign exists anyway.
                # We'll use bits as "fixed" set; others can still flip.
                if eid in bits:
                    if u==n: out.append(eid)
                    elif v==n: inn.append(eid)
                else:
                    und.append(eid)
            # heuristic:
            # if node has at least one fixed incoming and has und edges, set all und edges as outgoing (n->neighbor)
            if inn and und:
                for eid in und:
                    e=edge_list[eid]
                    # set direction n -> other
                    other = e["v"] if e["u"]==n else e["u"]
                    if e["u"]==n and e["v"]==other:
                        assign[eid]=0
                    elif e["u"]==other and e["v"]==n:
                        assign[eid]=1
                    bits[eid]=1
                    changed=True
                q.append(n)
                continue
            # if node has at least one fixed outgoing and has und edges, set all und edges as incoming (other->n)
            if out and und:
                for eid in und:
                    e=edge_list[eid]
                    other = e["v"] if e["u"]==n else e["u"]
                    # set other -> n
                    if e["u"]==other and e["v"]==n:
                        assign[eid]=0
                    elif e["u"]==n and e["v"]==other:
                        assign[eid]=1
                    bits[eid]=1
                    changed=True
                q.append(n)
                continue
            q.append(n)

    # 3) nonholonomic local repair pass:
    #    enforce split/merge and smooth pass-through at junctions.
    def objective(cur_assign):
        return (
            count_nonholonomic_branch_violations(edge_list, adj, cur_assign),
            _count_station_reachability_issues(edge_list, station_nodes, cur_assign),
            _count_dead_ends(edge_list, adj, cur_assign),
        )

    base_obj = objective(assign)
    for _ in range(8):
        improved = False
        for n in inc:
            if not violates_nonholonomic_branch_rule(n, inc[n], edge_list, assign):
                continue
            best_edge = None
            best_obj = base_obj
            for eid in inc[n]:
                trial = dict(assign)
                trial[eid] = 1 - trial[eid]
                trial_obj = objective(trial)
                if trial_obj < best_obj:
                    best_obj = trial_obj
                    best_edge = eid
            if best_edge is not None:
                assign[best_edge] = 1 - assign[best_edge]
                bits[best_edge] = 1
                base_obj = best_obj
                improved = True
        if not improved:
            break

    # scoring (simple)
    score = len(bits) - 100 * count_nonholonomic_branch_violations(edge_list, adj, assign)

    return bits, score, assign

def render(edge_list, adj, station_nodes, assign, out_png: Path, arrow_scale=6):
    fig,ax=plt.subplots(figsize=(10,10))
    # draw edges with geometry
    for e in edge_list:
        u=e["u"]; v=e["v"]
        uu,vv = (u,v) if assign[e["id"]]==0 else (v,u)
        if e["geom"]["type"]=="LINE":
            a=e["geom"]["a"]; b=e["geom"]["b"]
            xs=[a[0],b[0]]; ys=[a[1],b[1]]
            ax.plot(xs,ys, color="green", lw=2)
            # arrow at mid
            mx=(a[0]+b[0])/2; my=(a[1]+b[1])/2
            dx=(b[0]-a[0]); dy=(b[1]-a[1])
            if assign[e["id"]]==1:
                dx=-dx; dy=-dy
            ax.arrow(mx, my, dx*0.001*arrow_scale, dy*0.001*arrow_scale,
                     head_width=arrow_scale*2, head_length=arrow_scale*3,
                     fc="green", ec="green", length_includes_head=True)
        elif e["geom"]["type"]=="ARC":
            pts=e["geom"]["pts"]
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            ax.plot(xs,ys, color="green", lw=2)
            # arrow at middle sample
            mid=len(pts)//2
            p0=pts[mid-1]; p1=pts[mid+1]
            dx=(p1[0]-p0[0]); dy=(p1[1]-p0[1])
            mx=pts[mid][0]; my=pts[mid][1]
            # if assigned flip, reverse arrow tangent
            if assign[e["id"]]==1:
                dx=-dx; dy=-dy
            ax.arrow(mx, my, dx*0.05, dy*0.05,
                     head_width=arrow_scale*2, head_length=arrow_scale*3,
                     fc="green", ec="green", length_includes_head=True)

    # draw nodes
    xs=[n[0] for n in adj.keys()]; ys=[n[1] for n in adj.keys()]
    ax.scatter(xs,ys, s=20, c="red")

    # station nodes
    for name, p in station_nodes.items():
        ax.scatter([p[0]],[p[1]], s=40, c="blue", marker="s")
        ax.text(p[0], p[1], name, fontsize=8)

    ax.set_aspect('equal','box')
    ax.invert_yaxis()  # CAD y down
    ax.axis('off')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")

def dump_graph(edge_list, station_nodes, assign, out_json: Path | str):
    nodes={}
    def nid(p):
        if p not in nodes:
            nodes[p]=len(nodes)
        return nodes[p]

    edges=[]
    for e in edge_list:
        u=e["u"]; v=e["v"]
        uu,vv = (u,v) if assign[e["id"]]==0 else (v,u)
        edges.append({
            "id": e["id"],
            "u": nid(uu),
            "v": nid(vv),
            "kind": e["kind"],
            "geom": e["geom"],
        })

    out={
        "nodes":[{"id":i,"x":p[0],"y":p[1]} for p,i in nodes.items()],
        "edges":edges,
        "stations": {name: nid(p) for name,p in station_nodes.items()}
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] graph dump -> {out_json}")

def render_svg(edge_list, adj, station_nodes, assign, out_svg: Path):
    # very simple svg renderer for preview
    xs=[p[0] for p in adj.keys()]
    ys=[p[1] for p in adj.keys()]
    if not xs:
        return
    minx,maxx=min(xs),max(xs)
    miny,maxy=min(ys),max(ys)
    w=maxx-minx; h=maxy-miny
    pad=50
    vb=(minx-pad, miny-pad, w+2*pad, h+2*pad)

    def tr(p):
        # keep CAD coords; viewer can invert if needed. We'll not invert for svg.
        return p

    lines_out=[]
    lines_out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}">')
    lines_out.append('<rect x="-1e9" y="-1e9" width="2e9" height="2e9" fill="white"/>')

    # edges
    for e in edge_list:
        if e["geom"]["type"]=="LINE":
            a=e["geom"]["a"]; b=e["geom"]["b"]
            # direction
            if assign[e["id"]]==1:
                a,b=b,a
            lines_out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#00aa00" stroke-width="6" />')
            # arrow marker (simple triangle)
            mx=(a[0]+b[0])/2; my=(a[1]+b[1])/2
            dx=b[0]-a[0]; dy=b[1]-a[1]
            L=math.hypot(dx,dy) or 1.0
            ux,uy=dx/L,dy/L
            size=25
            px,py=-uy,ux
            tip=(mx+ux*size, my+uy*size)
            lft=(mx-ux*size+px*size*0.6, my-uy*size+py*size*0.6)
            rgt=(mx-ux*size-px*size*0.6, my-uy*size-py*size*0.6)
            lines_out.append(f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {lft[0]:.1f},{lft[1]:.1f} {rgt[0]:.1f},{rgt[1]:.1f}" fill="#00aa00"/>')
        elif e["geom"]["type"]=="ARC":
            pts=e["geom"]["pts"]
            if assign[e["id"]]==1:
                pts=list(reversed(pts))
            d="M "+ " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in pts])
            lines_out.append(f'<path d="{d}" fill="none" stroke="#00aa00" stroke-width="6"/>')
            # arrow at middle
            mid=len(pts)//2
            p0=pts[mid-1]; p1=pts[mid+1]
            mx,my=pts[mid]
            dx=p1[0]-p0[0]; dy=p1[1]-p0[1]
            L=math.hypot(dx,dy) or 1.0
            ux,uy=dx/L,dy/L
            size=25
            px,py=-uy,ux
            tip=(mx+ux*size, my+uy*size)
            lft=(mx-ux*size+px*size*0.6, my-uy*size+py*size*0.6)
            rgt=(mx-ux*size-px*size*0.6, my-uy*size-py*size*0.6)
            lines_out.append(f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {lft[0]:.1f},{lft[1]:.1f} {rgt[0]:.1f},{rgt[1]:.1f}" fill="#00aa00"/>')

    # nodes
    for p in adj.keys():
        x,y=tr(p)
        lines_out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="#e74c3c" />')

    # stations
    for name,p in station_nodes.items():
        x,y=tr(p)
        lines_out.append(
            f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" '
            f'fill="#3498db" stroke="#2980b9" stroke-width="1" rx="2"/>'
        )
        lines_out.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" '
            f'font-size="9" font-family="sans-serif" text-anchor="middle" '
            f'fill="#2c3e50">{name}</text>'
        )

    lines_out.append('</svg>')
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"[OK] SVG preview -> {out_svg}")


def main():
    if len(sys.argv) < 3:
        print("usage: python build_graph.py <in.dxf> <out.png> [out.json]")
        return 2
    in_dxf=Path(sys.argv[1])
    out_png=Path(sys.argv[2])
    out_json=Path(sys.argv[3]) if len(sys.argv)>=4 else None

    lines, arcs, texts = parse_dxf(in_dxf)
    edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)

    bits, score, assign = solve(edge_list, adj, by_ends, station_nodes)
    render(edge_list, adj, station_nodes, assign, out_png, arrow_scale=6)

    if out_json is not None:
        dump_graph(edge_list, station_nodes, assign, out_json)

    # SVG preview 생성 (PNG 경로에서 확장자만 변경)
    out_svg = out_png.with_suffix('.svg')
    render_svg(edge_list, adj, station_nodes, assign, out_svg)

    # brief stdout (optional)
    diag_eids=[e["id"] for e in edge_list if e["kind"]=="D"]
    print(f"diag_bits={bits} diag_eids={diag_eids} score={score}")

if __name__=="__main__":
    raise SystemExit(main())

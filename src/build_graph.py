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
            if i+1 < len(c) and c[i].strip()=="2":
                sec=c[i+1].strip()
                in_entities = (sec=="ENTITIES")
            continue
        if code=="0" and val=="ENDSEC":
            in_entities=False; continue
        if not in_entities:
            continue
        if code=="0":
            flush(); cur=val; continue
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
            elif code=="1": ent["text"]=val
    flush()
    return lines, arcs, texts

def arc_endpoints(arc):
    cx,cy,r=arc["cx"],arc["cy"],arc["r"]
    a0=math.radians(arc["a0"]); a1=math.radians(arc["a1"])
    return (cx+r*math.cos(a0), cy+r*math.sin(a0)), (cx+r*math.cos(a1), cy+r*math.sin(a1))

def ccw_sweep_deg(a0, a1):
    s = (a1 - a0) % 360.0
    if s <= 1e-9:
        s = 360.0
    return s

def point_seg_distance(px, py, ax, ay, bx, by):
    vx = bx - ax; vy = by - ay
    wx = px - ax; wy = py - ay
    vv = vx*vx + vy*vy
    if vv == 0:
        return math.hypot(px-ax, py-ay), 0.0, (ax, ay)
    t = (wx*vx + wy*vy)/vv
    t = max(0.0, min(1.0, t))
    qx = ax + t*vx; qy = ay + t*vy
    return math.hypot(px-qx, py-qy), t, (qx, qy)

def build_graph(lines, arcs, texts):
    arc_pts=[]
    arc_edges=[]
    for a in arcs:
        p0,p1=arc_endpoints(a)
        arc_pts.extend([p0,p1])
        arc_edges.append((p0,p1,a))

    station_points=[(t["x"],t["y"],t.get("text","")) for t in texts if "x" in t and "y" in t and "text" in t]

    def split_points_for_line(p1,p2):
        pts=[p1,p2]
        for ap in arc_pts:
            d,t,proj=point_seg_distance(ap[0],ap[1],p1[0],p1[1],p2[0],p2[1])
            if d <= EPS_ON:
                pts.append(proj)
        for sx,sy,_ in station_points:
            d,t,proj=point_seg_distance(sx,sy,p1[0],p1[1],p2[0],p2[1])
            if d <= STATION_PROJ_EPS:
                pts.append(proj)
        uniq={}
        for q in pts:
            uniq[nk(q)] = q
        vx=p2[0]-p1[0]; vy=p2[1]-p1[1]; vv=vx*vx+vy*vy
        def param(q):
            if vv==0: return 0.0
            return ((q[0]-p1[0])*vx + (q[1]-p1[1])*vy)/vv
        return sorted(uniq.values(), key=param)

    edge_list=[]
    for l in lines:
        p1=(l["x1"],l["y1"]); p2=(l["x2"],l["y2"])
        splits=split_points_for_line(p1,p2)
        for a,b in zip(splits, splits[1:]):
            if math.hypot(b[0]-a[0], b[1]-a[1]) < 1e-6:
                continue
            dx=b[0]-a[0]; dy=b[1]-a[1]
            if abs(dy) < 1e-6: kind="H"
            elif abs(dx) < 1e-6: kind="V"
            else: kind="D"
            edge_list.append({"src":"LINE","u":nk(a),"v":nk(b),"kind":kind,
                              "geom":{"p1":(a[0],a[1]),"p2":(b[0],b[1])}})

    # ARC: endpoint끼리 1 edge (새 edge 추가 금지) + 원래 ARC 파라미터 저장
    for p0,p1,a in arc_edges:
        u=nk(p0); v=nk(p1)
        edge_list.append({"src":"ARC","u":u,"v":v,"kind":"A",
                          "geom":{"cx":a["cx"],"cy":a["cy"],"r":a["r"],
                                  "a0":a["a0"],"a1":a["a1"],
                                  "p0":(p0[0],p0[1]),"p1":(p1[0],p1[1])}})

    station_nodes={}
    for sx,sy,name in station_points:
        best=None
        for l in lines:
            p1=(l["x1"],l["y1"]); p2=(l["x2"],l["y2"])
            d,t,proj=point_seg_distance(sx,sy,p1[0],p1[1],p2[0],p2[1])
            if best is None or d < best[0]:
                best=(d,proj)
        station_nodes[name]=nk(best[1])

    adj=collections.defaultdict(list)
    by_ends=collections.defaultdict(list)
    for eid,e in enumerate(edge_list):
        e["id"]=eid
        adj[e["u"]].append((e["v"], eid))
        adj[e["v"]].append((e["u"], eid))
        by_ends[frozenset((e["u"],e["v"]))].append(eid)

    return edge_list, adj, by_ends, station_nodes

def build_embedding(adj):
    emb={}
    for u, lst in adj.items():
        uniq=[]; seen=set()
        for v,_ in lst:
            if v not in seen:
                seen.add(v); uniq.append(v)
        def ang(v):
            return math.atan2(v[1]-u[1], v[0]-u[0])
        uniq.sort(key=ang)  # CCW
        emb[u]=uniq
    return emb

def next_halfedge(u,v,emb):
    neigh=emb[v]
    if len(neigh)==1:
        return None
    idx=neigh.index(u)
    w=neigh[(idx-1)%len(neigh)]  # right-face
    return (v,w)

def enumerate_faces(emb):
    used=set()
    faces=[]
    for u in emb:
        for v in emb[u]:
            he=(u,v)
            if he in used:
                continue
            cycle=[]
            cur=he
            while True:
                if cur in used:
                    break
                used.add(cur)
                cycle.append(cur[0])
                nxt=next_halfedge(cur[0],cur[1],emb)
                if nxt is None:
                    cycle=[]; break
                cur=nxt
                if cur==he:
                    cycle.append(cur[0])
                    break
            if len(cycle)>=4 and cycle[0]==cycle[-1]:
                poly=cycle[:-1]
                area=0.0
                for i in range(len(poly)):
                    x1,y1=poly[i]; x2,y2=poly[(i+1)%len(poly)]
                    area += x1*y2 - x2*y1
                area*=0.5
                faces.append({"poly":poly,"area":area})
    return faces

def get_edge_id(by_ends, edge_list, a,b):
    lst=by_ends.get(frozenset((a,b)),[])
    if not lst:
        return None
    if len(lst)==1:
        return lst[0]
    for eid in lst:
        if edge_list[eid]["src"]=="LINE":
            return eid
    return lst[0]

def outer_loop_cw_fixed(edge_list, adj, by_ends):
    emb=build_embedding(adj)
    faces=enumerate_faces(emb)
    outer=max(faces, key=lambda f: abs(f["area"]))
    poly=outer["poly"]
    if outer["area"]>0:
        poly=list(reversed(poly))  # CW
    fixed={}
    for i in range(len(poly)):
        a=poly[i]; b=poly[(i+1)%len(poly)]
        eid=get_edge_id(by_ends, edge_list, a,b)
        if eid is None:
            continue
        fixed[eid]=a
    return fixed, poly

def unit(v):
    n=np.linalg.norm(v)
    if n<1e-12:
        return np.array([0.0,0.0])
    return v/n

def angle_deg(u,v):
    u=unit(u); v=unit(v)
    dot=float(np.clip(np.dot(u,v),-1,1))
    return math.degrees(math.acos(dot))

def tail_head(edge_list, eid, val):
    e=edge_list[eid]; u,v=e["u"],e["v"]
    return (u,v) if val==0 else (v,u)

def set_dir(assign, edge_list, eid, tail):
    u,v=edge_list[eid]["u"], edge_list[eid]["v"]
    if tail==u: assign[eid]=0
    elif tail==v: assign[eid]=1
    else: raise ValueError("tail not endpoint")

def edge_dir_at_node(edge_list, eid, val, node):
    t,h=tail_head(edge_list,eid,val)
    if t==node: return "out"
    if h==node: return "in"
    return None

def build_incident(adj):
    inc=collections.defaultdict(list)
    for u,lst in adj.items():
        for v,eid in lst:
            inc[u].append(eid)
    return inc

def classify_deg3(edge_list, inc, node):
    eids=inc[node]
    vecs=[]
    for eid in eids:
        u,v=edge_list[eid]["u"],edge_list[eid]["v"]
        other = v if u==node else u
        vecs.append((eid, unit(np.array([other[0]-node[0], other[1]-node[1]],dtype=float))))
    best=-1; pair=None
    for i in range(3):
        for j in range(i+1,3):
            d=abs(float(np.dot(vecs[i][1], vecs[j][1])))
            if d>best:
                best=d; pair=(vecs[i][0], vecs[j][0])
    main1,main2=pair
    branch=[eid for eid in eids if eid not in pair][0]
    return (main1,main2,branch)

def node_counts(edge_list, inc, deg, assign, node):
    cin=0; cout=0; un=[]
    for eid in inc[node]:
        if eid in assign:
            d=edge_dir_at_node(edge_list,eid,assign[eid],node)
            if d=="in": cin+=1
            elif d=="out": cout+=1
        else:
            un.append(eid)
    return cin,cout,un

def check_node_feasible(edge_list, inc, deg, assign, node):
    cin,cout,un=node_counts(edge_list,inc,deg,assign,node)
    d=deg[node]
    if d==2:
        if cin>1 or cout>1: return False
        if cin+len(un) < 1 or cout+len(un) < 1: return False
        if len(un)==0 and not (cin==1 and cout==1): return False
    elif d==3:
        if cin>2 or cout>2: return False
        if cin+len(un) < 1 or cout+len(un) < 1: return False
        if len(un)==0 and not ((cin==1 and cout==2) or (cin==2 and cout==1)): return False
    return True

def turn_angle_ok(edge_list, inc, deg, deg3_info, assign, node):
    if deg[node]!=3:
        return True
    cin,cout,un=node_counts(edge_list,inc,deg,assign,node)
    if un:
        return True
    m1,m2,branch=deg3_info[node]
    d1=edge_dir_at_node(edge_list,m1,assign[m1],node)
    d2=edge_dir_at_node(edge_list,m2,assign[m2],node)
    if d1==d2:
        return False
    in_me = m1 if d1=="in" else m2
    out_me = m1 if d1=="out" else m2

    t_in,h_in = tail_head(edge_list,in_me,assign[in_me])
    other = t_in  # h_in==node
    v_in = np.array([node[0]-other[0], node[1]-other[1]],dtype=float)

    t_out,h_out = tail_head(edge_list,out_me,assign[out_me])
    other2 = h_out # t_out==node
    v_out = np.array([other2[0]-node[0], other2[1]-node[1]],dtype=float)

    tb,hb = tail_head(edge_list,branch,assign[branch])
    if tb==node:
        other3 = hb
        v_bout = np.array([other3[0]-node[0], other3[1]-node[1]],dtype=float)
        ang=angle_deg(v_in, v_bout)
    else:
        other3 = tb
        v_bin = np.array([node[0]-other3[0], node[1]-other3[1]],dtype=float)
        ang=angle_deg(v_bin, v_out)
    return ang <= 95

def propagate(edge_list, adj, inc, deg, deg3_info, assign):
    queue=collections.deque(inc.keys())
    inq=set(queue)
    while queue:
        node=queue.popleft(); inq.discard(node)

        if not check_node_feasible(edge_list,inc,deg,assign,node):
            return False, assign

        if deg[node]==2:
            cin,cout,un=node_counts(edge_list,inc,deg,assign,node)
            if len(un)==1:
                eid=un[0]
                u,v=edge_list[eid]["u"], edge_list[eid]["v"]
                other = v if u==node else u
                if cin==1 and cout==0:
                    set_dir(assign,edge_list,eid,node)
                elif cout==1 and cin==0:
                    set_dir(assign,edge_list,eid,other)
                if eid in assign:
                    for n2 in (u,v):
                        if n2 not in inq:
                            queue.append(n2); inq.add(n2)

        elif deg[node]==3:
            m1,m2,_=deg3_info[node]

            if m1 in assign and m2 not in assign:
                d=edge_dir_at_node(edge_list,m1,assign[m1],node)
                u,v=edge_list[m2]["u"], edge_list[m2]["v"]
                other = v if u==node else u
                if d=="in": set_dir(assign,edge_list,m2,node)
                else:       set_dir(assign,edge_list,m2,other)
                for n2 in (u,v):
                    if n2 not in inq:
                        queue.append(n2); inq.add(n2)

            elif m2 in assign and m1 not in assign:
                d=edge_dir_at_node(edge_list,m2,assign[m2],node)
                u,v=edge_list[m1]["u"], edge_list[m1]["v"]
                other = v if u==node else u
                if d=="in": set_dir(assign,edge_list,m1,node)
                else:       set_dir(assign,edge_list,m1,other)
                for n2 in (u,v):
                    if n2 not in inq:
                        queue.append(n2); inq.add(n2)

            if m1 in assign and m2 in assign:
                if edge_dir_at_node(edge_list,m1,assign[m1],node)==edge_dir_at_node(edge_list,m2,assign[m2],node):
                    return False, assign

            cin,cout,un=node_counts(edge_list,inc,deg,assign,node)
            if len(un)==1:
                eid=un[0]
                u,v=edge_list[eid]["u"], edge_list[eid]["v"]
                other = v if u==node else u
                if cin==0 or cout==2:
                    set_dir(assign,edge_list,eid,other) # incoming
                elif cout==0 or cin==2:
                    set_dir(assign,edge_list,eid,node)  # outgoing
                if eid in assign:
                    for n2 in (u,v):
                        if n2 not in inq:
                            queue.append(n2); inq.add(n2)

            if not turn_angle_ok(edge_list,inc,deg,deg3_info,assign,node):
                return False, assign

        if not check_node_feasible(edge_list,inc,deg,assign,node):
            return False, assign

    return True, assign

def kosaraju_scc(nodes, edges):
    dadj=collections.defaultdict(list)
    radj=collections.defaultdict(list)
    for t,h in edges:
        dadj[t].append(h)
        radj[h].append(t)
    visited=set(); order=[]
    sys.setrecursionlimit(10000)
    def dfs(u):
        visited.add(u)
        for v in dadj[u]:
            if v not in visited:
                dfs(v)
        order.append(u)
    for n in nodes:
        if n not in visited:
            dfs(n)
    visited=set()
    cid=0
    def rdfs(u):
        visited.add(u)
        for v in radj[u]:
            if v not in visited:
                rdfs(v)
    for u in reversed(order):
        if u not in visited:
            cid+=1
            rdfs(u)
    return cid

def station_reachability_score(station_nodes, edges):
    dadj=collections.defaultdict(list)
    for t,h in edges:
        dadj[t].append(h)
    st=list(station_nodes.values())
    total=0
    for src in st:
        q=collections.deque([src]); vis={src}
        while q:
            u=q.popleft()
            for v in dadj[u]:
                if v not in vis:
                    vis.add(v); q.append(v)
        total += sum(1 for dst in st if dst in vis)
    return total

def solve(edge_list, adj, by_ends, station_nodes):
    inc=build_incident(adj)
    deg={n: len(inc[n]) for n in inc}
    deg3_nodes=[n for n in inc if deg[n]==3]
    deg3_info={n: classify_deg3(edge_list,inc,n) for n in deg3_nodes}

    fixed_outer, _ = outer_loop_cw_fixed(edge_list, adj, by_ends)
    diag_eids=[e["id"] for e in edge_list if e["kind"]=="D"]  # 4개 예상

    best=None
    best_bits=None

    def solve_with_fixed(bits):
        assign={}
        for eid,tail in fixed_outer.items():
            set_dir(assign,edge_list,eid,tail)
        for i,eid in enumerate(diag_eids):
            assign[eid]=(bits>>i)&1

        ok, assign2 = propagate(edge_list,adj,inc,deg,deg3_info,assign)
        if not ok:
            return None

        all_eids=[e["id"] for e in edge_list]
        edge_score={eid: deg[edge_list[eid]["u"]]+deg[edge_list[eid]["v"]] for eid in all_eids}

        def pick_unassigned(a):
            un=[eid for eid in all_eids if eid not in a]
            if not un: return None
            un.sort(key=lambda eid: (-edge_score[eid], eid))
            return un[0]

        calls=0
        def backtrack(a):
            nonlocal calls
            calls+=1
            if calls>200000:
                return None
            ok2, a2 = propagate(edge_list,adj,inc,deg,deg3_info,dict(a))
            if not ok2:
                return None
            un=[eid for eid in all_eids if eid not in a2]
            if not un:
                return a2
            eid=pick_unassigned(a2)
            u,v=edge_list[eid]["u"],edge_list[eid]["v"]
            for tail in (u,v):
                a_try=dict(a2)
                set_dir(a_try,edge_list,eid,tail)
                if not check_node_feasible(edge_list,inc,deg,a_try,u) or not check_node_feasible(edge_list,inc,deg,a_try,v):
                    continue
                if not turn_angle_ok(edge_list,inc,deg,deg3_info,a_try,u) or not turn_angle_ok(edge_list,inc,deg,deg3_info,a_try,v):
                    continue
                res=backtrack(a_try)
                if res is not None:
                    return res
            return None
        return backtrack(assign2)

    for bits in range(1<<len(diag_eids)):
        sol=solve_with_fixed(bits)
        if sol is None:
            continue
        edges=[tail_head(edge_list,eid,val) for eid,val in sol.items()]
        indeg=collections.Counter(); outdeg=collections.Counter()
        for t,h in edges:
            outdeg[t]+=1; indeg[h]+=1
        dead=sum(1 for n in adj if len(adj[n])>1 and (indeg[n]==0 or outdeg[n]==0))
        reach=station_reachability_score(station_nodes, edges)
        scc_count=kosaraju_scc(list(adj.keys()), edges)
        score=(dead, -reach, scc_count)
        if best is None or score < best[0]:
            best=(score, sol); best_bits=bits

    if best is None:
        raise RuntimeError("유효한 단방향 해를 찾지 못했습니다.")
    return best_bits, best[0], best[1]

def sample_arc_points(cx, cy, r, a0, a1, ccw=True, n=ARC_SAMPLE_N):
    sweep=ccw_sweep_deg(a0, a1)
    if ccw:
        angs=np.linspace(a0, a0+sweep, n)
    else:
        angs=np.linspace(a0+sweep, a0, n)
    pts=[(cx+r*math.cos(math.radians(a)), cy+r*math.sin(math.radians(a))) for a in angs]
    return pts, (a0 + sweep/2.0)

def tangent_vec(ang_deg, ccw=True):
    ang=math.radians(ang_deg)
    # CCW tangent: (-sin, cos); CW tangent: (sin, -cos)
    if ccw:
        return np.array([-math.sin(ang), math.cos(ang)], dtype=float)
    else:
        return np.array([ math.sin(ang),-math.cos(ang)], dtype=float)

def render(edge_list, adj, station_nodes, assign, out_png: Path, arrow_scale=6):
    kind_color={"H":"tab:orange","V":"tab:green","D":"tab:cyan","A":"tab:purple"}
    fig, ax = plt.subplots(figsize=(12,6))
    for e in edge_list:
        eid=e["id"]
        t,h=tail_head(edge_list,eid,assign[eid])

        if e["src"]=="LINE":
            ax.plot([t[0],h[0]],[t[1],h[1]],linewidth=2,alpha=0.9,color=kind_color.get(e["kind"],"tab:blue"))
            mx=(t[0]+h[0])/2; my=(t[1]+h[1])/2
            dx=h[0]-t[0]; dy=h[1]-t[1]
            L=math.hypot(dx,dy)
            if L>1e-6:
                ux=dx/L; uy=dy/L
                alen=L*0.12
                ax.annotate("", xy=(mx+ux*alen*0.5, my+uy*alen*0.5), xytext=(mx-ux*alen*0.5, my-uy*alen*0.5),
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=arrow_scale, linewidth=1.0, color="black"))
        else:
            g=e["geom"]
            cx,cy,r=float(g["cx"]),float(g["cy"]),float(g["r"])
            # DXF 기준: start=a0, end=a1, CCW sweep
            a0=float(g["a0"])%360.0
            a1=float(g["a1"])%360.0
            # 본 edge의 '원래' 방향: u(start)->v(end)
            u=e["u"]; v=e["v"]
            travel_ccw = (t==u and h==v)
            pts, mid_ang = sample_arc_points(cx,cy,r,a0,a1,ccw=travel_ccw,n=ARC_SAMPLE_N)
            ax.plot([p[0] for p in pts],[p[1] for p in pts],linewidth=2,alpha=0.9,color=kind_color["A"])

            # arrow on arc (tangent)
            mid_pt=(cx+r*math.cos(math.radians(mid_ang)), cy+r*math.sin(math.radians(mid_ang)))
            tv=tangent_vec(mid_ang, ccw=travel_ccw)
            tv=tv/(np.linalg.norm(tv)+1e-12)
            alen=r*0.12
            ax.annotate("", xy=(mid_pt[0]+tv[0]*alen, mid_pt[1]+tv[1]*alen),
                        xytext=(mid_pt[0]-tv[0]*alen*0.2, mid_pt[1]-tv[1]*alen*0.2),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=arrow_scale, linewidth=1.0, color="black"))

    xs=[n[0] for n in adj.keys()]
    ys=[n[1] for n in adj.keys()]
    ax.scatter(xs,ys,s=10,color="tab:blue",alpha=0.65)

    for name,p in station_nodes.items():
        ax.text(p[0],p[1],name,fontsize=9,ha="center",va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Directed Graph (Outer CW propagate) — with ORIGINAL ARCs")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def dump_graph(edge_list, station_nodes, assign, out_json_path):
    """
    sim_core가 읽는 out.json 포맷으로 덤프:
      - nodes: [{id,x,y}]
      - edges: [{id,tail,head,length}]
      - stations: {name:{node_id,x,y}}
    tail/head는 '방향 그래프(assign)' 기준으로 결정됨.
    """
    import json
    import math

    # 1) 노드 수집: edge endpoint (u,v) 좌표를 node로 만든다
    coord_set = []
    for e in edge_list:
        coord_set.append(e["u"])
        coord_set.append(e["v"])

    # u/v는 이미 nk()로 round 된 (x,y) 튜플
    unique_coords = list(dict.fromkeys(coord_set))  # 순서 보존 유니크
    coord_to_id = {c: i for i, c in enumerate(unique_coords)}

    nodes = [{"id": i, "x": float(c[0]), "y": float(c[1])} for i, c in enumerate(unique_coords)]

    # 2) 엣지 생성: assign[eid]로 tail 결정 (0이면 u->v, 1이면 v->u)
    edges = []
    for e in edge_list:
        eid = e["id"]
        u = e["u"]
        v = e["v"]

        # 방향 결정
        if assign.get(eid, 0) == 0:
            tail = u
            head = v
        else:
            tail = v
            head = u

        tail_id = coord_to_id[tail]
        head_id = coord_to_id[head]

        # 길이 계산
        if e.get("src") == "ARC":
            g = e["geom"]
            r = float(g["r"])
            a0 = float(g["a0"]) % 360.0
            a1 = float(g["a1"]) % 360.0

            # ARC의 "원래" 방향은 u(start)->v(end)로 저장돼 있음.
            # 실제 이동 방향이 u->v면 CCW sweep, v->u면 반대 sweep으로 간주.
            travel_u_to_v = (tail == u and head == v)

            sweep_deg = ccw_sweep_deg(a0, a1)
            if not travel_u_to_v:
                sweep_deg = 360.0 - sweep_deg

            length = (math.radians(sweep_deg) * r)
        else:
            # LINE
            length = math.hypot(head[0] - tail[0], head[1] - tail[1])

        edges.append({
            "id": int(eid),
            "tail": int(tail_id),
            "head": int(head_id),
            "length": float(length),
        })

    # 3) stations: station_nodes는 name -> nk(coord) 형태
    stations = {}
    for name, coord in station_nodes.items():
        if coord in coord_to_id:
            nid = coord_to_id[coord]
            stations[name] = {"node_id": int(nid), "x": float(coord[0]), "y": float(coord[1])}

    graph_data = {"nodes": nodes, "edges": edges, "stations": stations}

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"[OK] Graph exported: {out_json_path}")
def main():
    if len(sys.argv) < 3:
        print("usage: python build_directed_graph_outer_cw_propagate_curves.py <in.dxf> <out.png> [out.json]")
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

    # brief stdout (optional)
    diag_eids=[e["id"] for e in edge_list if e["kind"]=="D"]
    print(f"diag_bits={bits} diag_eids={diag_eids} score={score}")

if __name__=="__main__":
    raise SystemExit(main())

"""Genetic algorithm for warehouse selection and truck routing.

Two classic supply-chain problems, solved by one genetic algorithm (GA):

* CVRP, the Capacitated Vehicle Routing Problem. One warehouse, trucks with a
  load limit and customers with demands: find the truck routes that serve
  every customer at the lowest cost.
* LRP, the Location-Routing Problem (the "warehouse problem"). Several
  candidate warehouses, each with a running cost, a stock limit and its own
  trucks: decide which warehouses to open, which customers each one serves
  and the truck routes, trading warehouse cost against transport cost.

A CVRP is an LRP with a single warehouse, so `solve` handles both.

Chromosome = (order, assign)
    order   a permutation of all customers (a "giant tour")
    assign  assign[c] is the warehouse that serves customer c
Each warehouse's customers, taken in `order`, are cut into truck routes by the
Split procedure (Prins, 2004), which finds the cheapest way to cut that
sequence. A customer whose warehouse has run out of stock is served by the
nearest one that still has some. Every chromosome therefore decodes to a
valid plan and needs no repair; a penalty is added only when the fleet or
the total stock is too small.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import time
from dataclasses import dataclass

EPS = 1e-9


# --------------------------------------------------------------- data model

@dataclass
class Customer:
    id: str
    x: float
    y: float
    demand: float


@dataclass
class Warehouse:
    id: str
    x: float
    y: float
    stock: float = math.inf     # units it can ship this period
    fixed_cost: float = 0.0     # cost of running it this period (rent, staff)
    trucks: int | None = None   # trucks based here, one route each; None = no limit


@dataclass
class Problem:
    customers: list[Customer]
    warehouses: list[Warehouse]
    truck_capacity: float
    cost_per_km: float = 1.0
    cost_per_truck: float = 0.0  # cost of sending out one truck (driver, dispatch)
    # Optional road distances: rows and columns are the customers, then the
    # warehouses, in list order. Defaults to straight-line distance.
    distance_matrix: list[list[float]] | None = None

    def __post_init__(self):
        if not self.customers or not self.warehouses:
            raise ValueError("need at least one customer and one warehouse")
        if self.truck_capacity <= 0:
            raise ValueError("truck_capacity must be positive")
        self.n, self.m = len(self.customers), len(self.warehouses)
        if self.distance_matrix is None:
            pts = [(c.x, c.y) for c in self.customers] + [(w.x, w.y) for w in self.warehouses]
            self.distance_matrix = [[math.dist(a, b) for b in pts] for a in pts]
        elif len(self.distance_matrix) != self.n + self.m:
            raise ValueError("distance_matrix must have one row per customer and warehouse")
        D, cap = self.distance_matrix, self.truck_capacity
        # Warehouses sorted by distance, per customer.
        self.nearest = [sorted(range(self.m), key=lambda w, c=c: D[c][self.n + w])
                        for c in range(self.n)]
        # An order bigger than a truckload is served by dedicated full-truck
        # trips; only the remainder (up to one truckload) shares a route.
        self.full_trips = [max(0, math.ceil(c.demand / cap) - 1) for c in self.customers]
        self.route_demand = [c.demand - k * cap for c, k in zip(self.customers, self.full_trips)]
        self.total_demand = sum(c.demand for c in self.customers)

        # When a warehouse runs short, customers claim its stock in this order:
        # those that would add most distance by being served from elsewhere first.
        def loss(c):
            near = self.nearest[c]
            extra = D[c][self.n + near[1]] - D[c][self.n + near[0]] if self.m > 1 else 0.0
            return (self.full_trips[c] + 1) * extra

        self.claim_order = sorted(range(self.n), key=loss, reverse=True)
        # Every feasible plan costs less than this, so adding it once per
        # broken limit ranks any feasible plan above any infeasible one.
        worst_trips = sum(
            (k + 1) * (2 * D[c][self.n + self.nearest[c][-1]] * self.cost_per_km + self.cost_per_truck)
            for c, k in enumerate(self.full_trips))
        self.penalty_weight = sum(w.fixed_cost for w in self.warehouses) + worst_trips + 1.0


@dataclass
class Route:
    warehouse: int           # index into problem.warehouses
    stops: list[int]         # customer indices, in visiting order
    load: float
    km: float
    full_load: bool = False  # dedicated trip with one truckload of an oversized order


@dataclass
class Solution:
    routes: list[Route]
    open_warehouses: list[int]
    shipped: dict[int, float]  # warehouse index -> units shipped
    fixed_cost: float
    transport_cost: float
    truck_cost: float
    penalty: float

    @property
    def total_cost(self) -> float:
        return self.fixed_cost + self.transport_cost + self.truck_cost + self.penalty

    @property
    def feasible(self) -> bool:
        return self.penalty == 0

    @property
    def km(self) -> float:
        return sum(r.km for r in self.routes)

    def report(self, p: Problem) -> str:
        lines = []
        for w in self.open_warehouses:
            wh = p.warehouses[w]
            routes = [r for r in self.routes if r.warehouse == w]
            stock = "unlimited" if math.isinf(wh.stock) else f"{wh.stock:g}"
            trucks = "unlimited" if wh.trucks is None else wh.trucks
            lines.append(f"{wh.id}: ships {self.shipped[w]:g} of {stock} units, "
                         f"uses {len(routes)} of {trucks} trucks")
            for k, r in enumerate(routes, 1):
                path = " -> ".join([wh.id] + [p.customers[c].id for c in r.stops] + [wh.id])
                tag = "  (full truck)" if r.full_load else ""
                lines.append(f"  truck {k}: {path} | load {r.load:g}/{p.truck_capacity:g}"
                             f" | {r.km:.1f} km{tag}")
        closed = [wh.id for w, wh in enumerate(p.warehouses) if w not in self.open_warehouses]
        if closed:
            lines.append("Not used: " + ", ".join(closed))
        parts = [f"{name} {value:,.2f}" for name, value in
                 (("warehouses", self.fixed_cost), ("transport", self.transport_cost),
                  ("trucks", self.truck_cost)) if value or name == "transport"]
        if self.penalty:
            parts.append(f"PENALTY {self.penalty:,.2f} (not enough trucks or stock)")
        lines.append(f"Cost: {' + '.join(parts)} = {self.total_cost:,.2f}"
                     f"   [{self.km:,.2f} km driven, {len(self.routes)} trucks]")
        return "\n".join(lines)


# ----------------------------------------------------------------- decoding

def _route_km(D, depot, stops):
    km, here = 0.0, depot
    for c in stops:
        km += D[here][c]
        here = c
    return km + D[here][depot]


def _feasible_routes(p, seq, depot):
    """Every (i, j, cost) such that seq[i:j] fits in one truck."""
    D, q = p.distance_matrix, p.route_demand
    arcs = []
    for i in range(len(seq)):
        load = km = 0.0
        for j in range(i, len(seq)):
            c = seq[j]
            load += q[c]
            if load > p.truck_capacity + EPS:
                break
            if j == i:
                km = D[depot][c] + D[c][depot]
            else:
                b = seq[j - 1]
                km += D[b][c] + D[c][depot] - D[b][depot]
            arcs.append((i, j + 1, km * p.cost_per_km + p.cost_per_truck))
    return arcs


def _split(p, seq, depot, max_routes=None):
    """Cut `seq` into consecutive truck routes at the lowest cost (Prins' Split).

    A shortest path over cut points. With `max_routes`, layer k holds the
    cheapest way to cover each prefix with exactly k routes. Returns None
    when `max_routes` trucks cannot cover the sequence.
    """
    n = len(seq)
    arcs = _feasible_routes(p, seq, depot)
    if max_routes is None or max_routes >= n:
        best = [0.0] + [math.inf] * n
        pred = [0] * (n + 1)
        for i, j, cost in arcs:
            if best[i] + cost < best[j]:
                best[j], pred[j] = best[i] + cost, i
        cuts = [n]
        while cuts[-1] > 0:
            cuts.append(pred[cuts[-1]])
    else:
        prev = [0.0] + [math.inf] * n
        preds, best_k, best_cost = [], None, math.inf
        for k in range(1, max_routes + 1):
            cur, pred = [math.inf] * (n + 1), [0] * (n + 1)
            for i, j, cost in arcs:
                if prev[i] + cost < cur[j]:
                    cur[j], pred[j] = prev[i] + cost, i
            preds.append(pred)
            if cur[n] < best_cost:
                best_k, best_cost = k, cur[n]
            prev = cur
        if best_k is None:
            return None
        cuts = [n]
        for k in range(best_k, 0, -1):
            cuts.append(preds[k - 1][cuts[-1]])
    cuts.reverse()
    return [seq[a:b] for a, b in zip(cuts, cuts[1:])]


def decode(p: Problem, order: list[int], assign: list[int]) -> Solution:
    """Turn a chromosome into a delivery plan and price it.

    Customers claim stock in `p.claim_order`. One whose warehouse has run
    out is served by the nearest warehouse that still has stock, preferring
    those the chromosome already uses, so plans only break a stock limit
    when there is not enough stock to go round.
    """
    D = p.distance_matrix
    left = [w.stock for w in p.warehouses]
    in_use = set(assign)
    served_from = list(assign)
    for c in p.claim_order:
        w, demand = assign[c], p.customers[c].demand
        if left[w] < demand - EPS:
            has_stock = [u for u in p.nearest[c] if left[u] >= demand - EPS]
            w = next((u for u in has_stock if u in in_use), has_stock[0] if has_stock else w)
        left[w] -= demand
        served_from[c] = w
    groups = [[] for _ in range(p.m)]
    for c in order:
        groups[served_from[c]].append(c)
    routes, open_w, shipped = [], [], {}
    fixed = penalty = 0.0
    for w, seq in enumerate(groups):
        if not seq:
            continue
        wh, depot = p.warehouses[w], p.n + w
        open_w.append(w)
        fixed += wh.fixed_cost
        mine = [Route(w, [c], p.truck_capacity, 2 * D[depot][c], full_load=True)
                for c in seq for _ in range(p.full_trips[c])]
        spare = None if wh.trucks is None else wh.trucks - len(mine)
        cuts = _split(p, seq, depot, spare)
        if cuts is None:  # not enough trucks: plan it anyway and penalise
            cuts = _split(p, seq, depot)
        mine += [Route(w, s, sum(p.route_demand[c] for c in s), _route_km(D, depot, s))
                 for s in cuts]
        shipped[w] = sum(p.customers[c].demand for c in seq)
        if shipped[w] > wh.stock + EPS:
            penalty += p.penalty_weight * (1 + (shipped[w] - wh.stock) / p.truck_capacity)
        if wh.trucks is not None and len(mine) > wh.trucks:
            penalty += p.penalty_weight * (1 + len(mine) - wh.trucks)
        routes += mine
    km = sum(r.km for r in routes)
    return Solution(routes, open_w, shipped, fixed, km * p.cost_per_km,
                    len(routes) * p.cost_per_truck, penalty)


# ------------------------------------------------------- simple heuristics

def _nearest_allowed(p, c, allowed):
    return next(w for w in p.nearest[c] if w in allowed)


def _nearest_neighbour_order(p, assign):
    D = p.distance_matrix
    order = []
    for w in range(p.m):
        todo = {c for c in range(p.n) if assign[c] == w}
        here = p.n + w
        while todo:
            here = min(todo, key=D[here].__getitem__)
            todo.remove(here)
            order.append(here)
    return order


def rule_of_thumb(p: Problem) -> tuple[list[int], list[int]]:
    """A plan made by hand: each customer (biggest first) goes to the nearest
    warehouse that still has stock; trucks drive to the nearest unvisited
    customer next. Returns a chromosome (order, assign)."""
    left = [w.stock for w in p.warehouses]
    assign = [0] * p.n
    for c in sorted(range(p.n), key=lambda c: -p.customers[c].demand):
        demand = p.customers[c].demand
        options = {w for w in range(p.m) if left[w] >= demand} or set(range(p.m))
        assign[c] = _nearest_allowed(p, c, options)
        left[assign[c]] -= demand
    return _nearest_neighbour_order(p, assign), assign


def baseline(p: Problem) -> Solution:
    return decode(p, *rule_of_thumb(p))


def price_routes(p: Problem, routes: list[tuple[int, list[int]]]) -> Solution:
    """Price a plan given as (warehouse index, [customer indices]) routes, such
    as one copied from a report, so it can be compared with a GA plan."""
    D = p.distance_matrix
    rs = [Route(w, list(s), sum(p.customers[c].demand for c in s), _route_km(D, p.n + w, s))
          for w, s in routes if s]
    open_w = sorted({r.warehouse for r in rs})
    shipped = {w: sum(r.load for r in rs if r.warehouse == w) for w in open_w}
    km = sum(r.km for r in rs)
    return Solution(rs, open_w, shipped, sum(p.warehouses[w].fixed_cost for w in open_w),
                    km * p.cost_per_km, len(rs) * p.cost_per_truck, 0.0)


def _greedy_assign(p, allowed, assign, todo):
    """Give each customer in `todo` the nearest warehouse in `allowed` that
    still has stock for it; everyone else keeps their warehouse. Customers
    who would lose most by missing their first choice pick first."""
    D = p.distance_matrix
    left = [w.stock for w in p.warehouses]
    for c in range(p.n):
        if c not in todo:
            left[assign[c]] -= p.customers[c].demand
    options = {c: [w for w in p.nearest[c] if w in allowed] for c in todo}

    def regret(c):
        o = options[c]
        if len(o) < 2:
            return math.inf
        return (p.full_trips[c] + 1) * (D[c][p.n + o[1]] - D[c][p.n + o[0]])

    for c in sorted(todo, key=regret, reverse=True):
        demand = p.customers[c].demand
        w = next((w for w in options[c] if left[w] >= demand - EPS), options[c][0])
        assign[c] = w
        left[w] -= demand


def _random_network(p, rng):
    """A random set of warehouses with enough stock between them, if possible."""
    ws = list(range(p.m))
    rng.shuffle(ws)
    chosen = set(ws[:rng.randint(1, p.m)])
    for w in ws:
        if sum(p.warehouses[u].stock for u in chosen) >= p.total_demand:
            break
        chosen.add(w)
    return chosen


def _random_plan(p, rng):
    """Open a random set of warehouses, assign customers greedily, and order
    them randomly or by a sweep around their warehouse."""
    assign = [0] * p.n
    _greedy_assign(p, _random_network(p, rng), assign, set(range(p.n)))
    order = list(range(p.n))
    if rng.random() < 0.5:
        rng.shuffle(order)
    else:
        turn = rng.uniform(0, 2 * math.pi)

        def angle(c):
            cu, wh = p.customers[c], p.warehouses[assign[c]]
            return (math.atan2(cu.y - wh.y, cu.x - wh.x) - turn) % (2 * math.pi)

        order.sort(key=angle)
    return order, assign


# -------------------------------------------------------- genetic operators

class _Individual:
    __slots__ = ("order", "assign", "solution", "cost")

    def __init__(self, p, order, assign):
        self.order, self.assign = order, assign
        self.solution = decode(p, order, assign)
        self.cost = self.solution.total_cost

    def key(self):
        return tuple(self.order), tuple(self.assign)


def _tournament(pop, size, rng):
    return min(rng.sample(pop, min(size, len(pop))), key=lambda ind: ind.cost)


def _order_crossover(a, b, rng):
    """OX: keep a slice of parent a, fill the rest in parent b's order.
    The child is always a valid permutation, so no repair is needed."""
    n = len(a)
    if n < 2:
        return a[:]
    i, j = sorted(rng.sample(range(n + 1), 2))
    middle = a[i:j]
    taken = set(middle)
    rest = [g for g in b[j:] + b[:j] if g not in taken]
    return rest[n - j:] + middle + rest[:n - j]


def _spatial_crossover(p, a, b, rng):
    """Customers on one side of a random line keep parent a's warehouse,
    the rest take parent b's, so neighbouring customers move together."""
    t = rng.uniform(0, math.pi)
    proj = [c.x * math.cos(t) + c.y * math.sin(t) for c in p.customers]
    cut = rng.choice(proj)
    return [a[c] if proj[c] <= cut else b[c] for c in range(p.n)]


def _mutate_order(order, rng):
    if len(order) < 2:
        return
    i, j = sorted(rng.sample(range(len(order)), 2))
    op = rng.randrange(3)
    if op == 0:    # swap two customers
        order[i], order[j] = order[j], order[i]
    elif op == 1:  # reverse a stretch of the tour
        order[i:j + 1] = order[i:j + 1][::-1]
    else:          # move one customer elsewhere
        order.insert(j, order.pop(i))


def _mutate_assign(p, assign, rng):
    """Change which warehouse serves whom. Returns True for the big moves
    that redraw the network, so the caller can tidy the routes afterwards."""
    used = set(assign)
    everyone = set(range(p.n))
    op = rng.randrange(4)
    if op == 0:                      # hand one customer to a nearby warehouse
        c = rng.randrange(p.n)
        assign[c] = rng.choice(p.nearest[c][:3])
        return False
    if op == 1 and len(used) > 1:    # close a warehouse, its customers move elsewhere
        w = rng.choice(sorted(used))
        _greedy_assign(p, used - {w}, assign, {c for c in everyone if assign[c] == w})
    elif op == 2:                    # open one more warehouse and re-assign everyone
        _greedy_assign(p, used | {rng.randrange(p.m)}, assign, everyone)
    else:                            # redesign the whole network
        _greedy_assign(p, _random_network(p, rng), assign, everyone)
    return True


def _two_opt(D, depot, stops):
    """Uncross a route: reverse any stretch that makes it shorter."""
    tour = [depot] + stops + [depot]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b, c, d = tour[i - 1], tour[i], tour[j], tour[j + 1]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - EPS:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour[1:-1]


def _relocate(p, depot, routes):
    """Move single customers to the cheapest spot in another truck from the
    same warehouse, as long as that lowers the cost. Edits `routes` in place."""
    D, q = p.distance_matrix, p.route_demand
    loads = [sum(q[c] for c in r) for r in routes]
    moved = True
    while moved:
        moved = False
        for r1, src in enumerate(routes):
            for pos, c in enumerate(src):
                a = src[pos - 1] if pos else depot
                b = src[pos + 1] if pos + 1 < len(src) else depot
                saving = (D[a][c] + D[c][b] - D[a][b]) * p.cost_per_km
                if len(src) == 1:
                    saving += p.cost_per_truck  # that truck stays at home
                best = None
                for r2, dst in enumerate(routes):
                    if r2 == r1 or not dst or loads[r2] + q[c] > p.truck_capacity + EPS:
                        continue
                    for k in range(len(dst) + 1):
                        x = dst[k - 1] if k else depot
                        y = dst[k] if k < len(dst) else depot
                        extra = (D[x][c] + D[c][y] - D[x][y]) * p.cost_per_km
                        if extra < saving - EPS and (best is None or extra < best[0]):
                            best = (extra, r2, k)
                if best:
                    _, r2, k = best
                    routes[r2].insert(k, src.pop(pos))
                    loads[r1] -= q[c]
                    loads[r2] += q[c]
                    moved = True
                    break
            if moved:
                break
    routes[:] = [r for r in routes if r]


def _polish(p, ind):
    """Local search on the decoded plan: relocate customers between trucks,
    uncross each route with 2-opt, then rebuild the giant tour."""
    D = p.distance_matrix
    order, assign = [], ind.assign[:]
    for w in ind.solution.open_warehouses:
        depot = p.n + w
        routes = [r.stops[:] for r in ind.solution.routes if r.warehouse == w and not r.full_load]
        _relocate(p, depot, routes)
        for stops in routes:
            for c in _two_opt(D, depot, stops):
                order.append(c)
                assign[c] = w  # where decode actually sent it
    better = _Individual(p, order, assign)
    return better if better.cost <= ind.cost else ind


# ------------------------------------------------------------------ solver

@dataclass
class GAParams:
    population_size: int = 60
    max_generations: int = 400
    stall_generations: int = 80     # stop after this many generations without improvement
    crossover_rate: float = 0.9
    mutation_rate: float = 0.4
    elite_count: int = 2            # best plans copied unchanged into the next generation
    tournament_size: int = 3        # each parent is the best of this many random picks
    local_search_rate: float = 0.2  # share of children improved with 2-opt
    seed: int | None = None


@dataclass
class GAResult:
    best: Solution
    order: list[int]
    assign: list[int]
    history: list[dict]  # per generation: best cost, median feasible cost, warehouses open
    generations: int
    seconds: float


def solve(p: Problem, params: GAParams | None = None, verbose: bool = False) -> GAResult:
    params = params or GAParams()
    rng = random.Random(params.seed)
    start = time.perf_counter()

    pop = [_Individual(p, *rule_of_thumb(p))]
    while len(pop) < params.population_size:
        pop.append(_Individual(p, *_random_plan(p, rng)))
    pop.sort(key=lambda ind: ind.cost)

    history, stall, gen = [], 0, 0
    best = pop[0]
    for gen in range(1, params.max_generations + 1):
        nxt = pop[:params.elite_count]
        seen = {ind.key() for ind in nxt}
        tries = 0
        while len(nxt) < params.population_size:
            a = _tournament(pop, params.tournament_size, rng)
            b = _tournament(pop, params.tournament_size, rng)
            if rng.random() < params.crossover_rate:
                order = _order_crossover(a.order, b.order, rng)
                assign = _spatial_crossover(p, a.assign, b.assign, rng) if p.m > 1 else a.assign[:]
            else:
                order, assign = a.order[:], a.assign[:]
            redesigned = False
            if rng.random() < params.mutation_rate:
                if p.m > 1 and rng.random() < 0.5:
                    redesigned = _mutate_assign(p, assign, rng)
                else:
                    _mutate_order(order, rng)
            child = _Individual(p, order, assign)
            # A redrawn network starts with clumsy routes; tidy them before it
            # competes, or good warehouse choices die young.
            if redesigned or rng.random() < params.local_search_rate:
                child = _polish(p, child)
            tries += 1
            # Skip duplicates to keep the population diverse, but never loop forever.
            if child.key() in seen and tries < 5 * params.population_size:
                continue
            seen.add(child.key())
            nxt.append(child)
        pop = sorted(nxt, key=lambda ind: ind.cost)

        if pop[0].cost < best.cost - EPS:
            best, stall = pop[0], 0
        else:
            stall += 1
        feasible = [ind.cost for ind in pop if ind.solution.feasible]
        history.append({
            "generation": gen,
            "best": pop[0].cost,
            "median": feasible[len(feasible) // 2] if feasible else math.nan,
            "open": len(pop[0].solution.open_warehouses),
        })
        if verbose and (gen == 1 or gen % 50 == 0):
            print(f"generation {gen:4d}: best cost {pop[0].cost:,.0f}")
        if stall >= params.stall_generations:
            break

    best = _polish(p, best)
    if verbose:
        print(f"stopped after {gen} generations: best cost {best.cost:,.0f}")
    return GAResult(best.solution, best.order, best.assign, history, gen,
                    time.perf_counter() - start)


def brute_force(p: Problem, max_customers: int = 7) -> Solution:
    """Try every plan. Only practical for tiny problems; used to check the GA."""
    if p.n > max_customers:
        raise ValueError(f"brute force is limited to {max_customers} customers")
    best = None
    for assign in itertools.product(range(p.m), repeat=p.n):
        groups = [[c for c in range(p.n) if assign[c] == w] for w in range(p.m)]
        for perms in itertools.product(*(itertools.permutations(g) for g in groups)):
            sol = decode(p, [c for perm in perms for c in perm], list(assign))
            if best is None or sol.total_cost < best.total_cost - EPS:
                best = sol
    return best


# --------------------------------------------------------------- input/out

def load_problem(customers_csv: str, warehouses_csv: str, truck_capacity: float,
                 **kwargs) -> Problem:
    """Read customers (id,x,y,demand) and warehouses (id,x,y,stock,fixed_cost,trucks).
    A blank stock or trucks cell means no limit."""
    with open(customers_csv, newline="", encoding="utf-8") as f:
        customers = [Customer(r["id"], float(r["x"]), float(r["y"]), float(r["demand"]))
                     for r in csv.DictReader(f)]
    with open(warehouses_csv, newline="", encoding="utf-8") as f:
        warehouses = [Warehouse(r["id"], float(r["x"]), float(r["y"]),
                                stock=float(r.get("stock") or math.inf),
                                fixed_cost=float(r.get("fixed_cost") or 0),
                                trucks=int(r["trucks"]) if r.get("trucks") else None)
                      for r in csv.DictReader(f)]
    return Problem(customers, warehouses, truck_capacity, **kwargs)


def problem_from_input() -> Problem:
    """Type a small problem in at the keyboard, like the original notebook."""
    customers = []
    for i in range(1, int(input("Number of customers: ")) + 1):
        x = float(input(f"Customer {i} x: "))
        y = float(input(f"Customer {i} y: "))
        customers.append(Customer(f"C{i}", x, y, float(input(f"Customer {i} demand: "))))
    warehouses = []
    for i in range(1, int(input("Number of candidate warehouses: ")) + 1):
        x = float(input(f"Warehouse {i} x: "))
        y = float(input(f"Warehouse {i} y: "))
        stock = input(f"Warehouse {i} stock (blank = unlimited): ")
        cost = input(f"Warehouse {i} running cost (blank = 0): ")
        trucks = input(f"Warehouse {i} trucks (blank = unlimited): ")
        warehouses.append(Warehouse(f"W{i}", x, y, float(stock or math.inf),
                                    float(cost or 0), int(trucks) if trucks else None))
    return Problem(customers, warehouses,
                   truck_capacity=float(input("Truck capacity: ")),
                   cost_per_km=float(input("Cost per km (blank = 1): ") or 1),
                   cost_per_truck=float(input("Cost per truck sent out (blank = 0): ") or 0))


# ---------------------------------------------------------------- plotting

def plot_plan(p: Problem, sol: Solution | None, ax=None, title: str | None = None):
    """Map of the plan: squares are warehouses (hollow = not used), circles are
    customers sized by demand, each coloured line is one truck. Dotted lines
    are full-truck trips for oversized orders. With sol=None, just the map."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    colours = plt.get_cmap("tab10").colors
    for k, r in enumerate(sol.routes if sol else []):
        wh = p.warehouses[r.warehouse]
        xs = [wh.x] + [p.customers[c].x for c in r.stops] + [wh.x]
        ys = [wh.y] + [p.customers[c].y for c in r.stops] + [wh.y]
        ax.plot(xs, ys, ls=":" if r.full_load else "-", lw=1.6,
                color=colours[k % len(colours)], alpha=0.9, zorder=1)
    biggest = max(c.demand for c in p.customers) or 1
    ax.scatter([c.x for c in p.customers], [c.y for c in p.customers],
               s=[15 + 110 * c.demand / biggest for c in p.customers],
               c="white", edgecolors="#333", zorder=2)
    for c in p.customers:
        ax.annotate(c.id, (c.x, c.y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for w, wh in enumerate(p.warehouses):
        used = sol is None or w in sol.open_warehouses
        ax.scatter(wh.x, wh.y, marker="s", s=180, c="#222" if used else "white",
                   edgecolors="#222", linewidths=1.5, zorder=3)
        ax.annotate(wh.id + ("" if used else " (not used)"), (wh.x, wh.y), xytext=(7, -12),
                    textcoords="offset points", fontsize=9, weight="bold")
    if title is None:
        title = (f"{p.n} customers, {p.m} warehouse(s)" if sol is None else
                 f"{len(sol.open_warehouses)} warehouse(s), {len(sol.routes)} trucks, "
                 f"cost {sol.total_cost:,.0f}")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.grid(alpha=0.2)
    return ax


def plot_convergence(result: GAResult, ax=None, title: str = "How the GA improved"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    gens = [h["generation"] for h in result.history]
    ax.plot(gens, [h["median"] for h in result.history], color="#999", label="median plan in population")
    ax.plot(gens, [h["best"] for h in result.history], color="#1f77b4", lw=2, label="best plan")
    ax.set_xlabel("generation")
    ax.set_ylabel("total cost (lower is better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_costs(plans: dict[str, Solution], ax=None, title: str = "Where the money goes"):
    """Stacked bars of warehouse, transport and truck cost for each named plan."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    names = list(plans)
    bottom = [0.0] * len(names)
    for label, part, colour in (("warehouses", "fixed_cost", "#8c6d31"),
                                ("transport (km)", "transport_cost", "#1f77b4"),
                                ("trucks sent out", "truck_cost", "#9ecae1"),
                                ("penalty", "penalty", "#d62728")):
        values = [getattr(plans[n], part) for n in names]
        if any(values):
            ax.bar(names, values, bottom=bottom, label=label, color=colour)
            bottom = [b + v for b, v in zip(bottom, values)]
    for i, total in enumerate(bottom):
        ax.annotate(f"{total:,.0f}", (i, total), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("cost")
    ax.set_title(title)
    ax.legend()
    return ax


# --------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan warehouses and truck routes with a genetic algorithm.")
    ap.add_argument("customers", help="CSV with id,x,y,demand")
    ap.add_argument("warehouses", help="CSV with id,x,y,stock,fixed_cost,trucks")
    ap.add_argument("--capacity", type=float, required=True, help="units one truck can carry")
    ap.add_argument("--cost-per-km", type=float, default=1.0)
    ap.add_argument("--cost-per-truck", type=float, default=0.0)
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--plot", help="save a map of the plan to this PNG file")
    args = ap.parse_args(argv)

    p = load_problem(args.customers, args.warehouses, args.capacity,
                     cost_per_km=args.cost_per_km, cost_per_truck=args.cost_per_truck)
    result = solve(p, GAParams(max_generations=args.generations, seed=args.seed), verbose=True)
    print()
    print(result.best.report(p))
    base = baseline(p).total_cost
    print(f"\nRule of thumb costs {base:,.0f}; the GA plan saves "
          f"{100 * (base - result.best.total_cost) / base:.1f}%.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_plan(p, result.best)
        ax.figure.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Map saved to {args.plot}")


if __name__ == "__main__":
    main()

"""Checks the GA against exhaustive search. Run: python test_logistics_ga.py"""
import math
import random

import logistics_ga as lg
from logistics_ga import Customer, GAParams, Problem, Warehouse


def check_plan(p, sol):
    """Every unit of demand delivered once, no truck overloaded, distances right."""
    got = [0.0] * p.n
    for r in sol.routes:
        assert r.load <= p.truck_capacity + 1e-9, r
        if r.full_load:
            got[r.stops[0]] += p.truck_capacity
        else:
            for c in r.stops:
                got[c] += p.route_demand[c]
        assert abs(lg._route_km(p.distance_matrix, p.n + r.warehouse, r.stops) - r.km) < 1e-6
    for c in range(p.n):
        assert abs(got[c] - p.customers[c].demand) < 1e-9, (c, got[c])
    assert sorted(sol.open_warehouses) == sorted({r.warehouse for r in sol.routes})


def test_report_example():
    p = Problem([Customer("1", 23, -41, 16), Customer("2", -43, -12, 19),
                 Customer("3", 34, 24, 12), Customer("4", 30, 9, 14)],
                [Warehouse("D", 0, 0, trucks=3)], truck_capacity=28)
    assert abs(lg.brute_force(p).total_cost - 271.77) < 0.01
    assert abs(lg.solve(p, GAParams(seed=1)).best.total_cost - 271.77) < 0.01
    print("report's 4-store example: 271.77 km, optimal")


def test_split_with_truck_limit():
    rng = random.Random(0)
    for t in range(300):
        n = rng.randint(1, 7)
        cs = [Customer(str(i), rng.uniform(0, 50), rng.uniform(0, 50), rng.randint(1, 10)) for i in range(n)]
        p = Problem(cs, [Warehouse("W", 25, 25)], truck_capacity=12, cost_per_truck=rng.choice([0, 30]))
        seq = list(range(n))
        rng.shuffle(seq)
        k = rng.randint(1, n)
        cost = lambda parts: sum(lg._route_km(p.distance_matrix, n, s) + p.cost_per_truck for s in parts)
        best = math.inf
        for mask in range(1 << (n - 1)):  # every way to cut the sequence
            cuts = [0] + [i + 1 for i in range(n - 1) if mask >> i & 1] + [n]
            parts = [seq[a:b] for a, b in zip(cuts, cuts[1:])]
            if len(parts) <= k and all(sum(p.route_demand[c] for c in s) <= 12 for s in parts):
                best = min(best, cost(parts))
        got = lg._split(p, seq, n, k)
        if best == math.inf:
            assert got is None, t
        else:
            assert len(got) <= k and abs(cost(got) - best) < 1e-6, t
    print("Split with a truck limit: 300 random cases match exhaustive search")


def test_ga_matches_brute_force():
    for t in range(25):
        r = random.Random(100 + t)
        n, m = r.randint(4, 6), r.randint(1, 3)
        cs = [Customer(f"C{i}", r.uniform(0, 100), r.uniform(0, 100), r.randint(5, 40)) for i in range(n)]
        if r.random() < 0.4:
            cs[0].demand = r.randint(45, 110)  # oversized order
        total = sum(c.demand for c in cs)
        ws = [Warehouse(f"W{j}", r.uniform(0, 100), r.uniform(0, 100),
                        stock=r.choice([math.inf, total * 0.6, total]),
                        fixed_cost=r.choice([0, 50, 200]), trucks=r.choice([None, 3, 4]))
              for j in range(m)]
        p = Problem(cs, ws, truck_capacity=40, cost_per_truck=r.choice([0, 20]))
        exact = lg.brute_force(p)
        ga = lg.solve(p, GAParams(seed=t, population_size=40, max_generations=200, stall_generations=60)).best
        check_plan(p, exact)
        check_plan(p, ga)
        assert abs(ga.total_cost - exact.total_cost) < 1e-6 * exact.total_cost, (t, ga.total_cost, exact.total_cost)
    print("GA found the brute-force optimum on 25 of 25 small multi-warehouse problems")


def test_random_chromosomes_decode():
    rng = random.Random(1)
    p = Problem([Customer(str(i), rng.uniform(0, 100), rng.uniform(0, 100), rng.randint(1, 250)) for i in range(25)],
                [Warehouse(f"W{j}", rng.uniform(0, 100), rng.uniform(0, 100), stock=1500, trucks=6)
                 for j in range(3)], truck_capacity=100)
    for _ in range(500):
        order = list(range(p.n))
        rng.shuffle(order)
        check_plan(p, lg.decode(p, order, [rng.randrange(p.m) for _ in range(p.n)]))
    print("500 random chromosomes decode to valid plans")


if __name__ == "__main__":
    test_report_example()
    test_split_with_truck_limit()
    test_ga_matches_brute_force()
    test_random_chromosomes_decode()

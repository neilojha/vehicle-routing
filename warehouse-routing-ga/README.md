# Warehouse & Vehicle Routing with a Genetic Algorithm

A follow-up to **"Vehicle Routing Optimization – A Genetic Algorithm Based Solution"** (NIT Kurukshetra, 2025). That project solved the single-warehouse CVRP. This one adds the report's first future research direction, a **multi-depot, inventory-constrained VRP**, and then lets the GA decide **which warehouses to run** (location-routing).

| Problem | What the GA decides | Objective |
|---|---|---|
| CVRP (one warehouse) | truck routes | min distance |
| Multi-depot VRP with stock limits | store → warehouse allocation, routes | min distance |
| Location-routing (network design) | which warehouses to run, allocation, routes | min daily cost (₹) |

## Files

| File | What it is |
|---|---|
| `Warehouse_Routing_GA.ipynb` | Walkthrough with explanations, results and plots (already run) |
| `logistics_ga.py` | The GA library; also a command-line tool |
| `test_logistics_ga.py` | Checks the GA against brute force on small problems |
| `data/customers.csv`, `data/warehouses.csv` | Sample region: 30 stores, 4 candidate warehouses |
| `data/report_case_studies.csv`, `data/report_case_settings.csv` | The report's 10 case studies and its best chromosomes |

## Quick start

```
pip install -r requirements.txt
jupyter notebook Warehouse_Routing_GA.ipynb
```

From a terminal:

```
python logistics_ga.py data/customers.csv data/warehouses.csv --capacity 100 --cost-per-km 40 --cost-per-truck 2000 --plot plan.png
```

From Python:

```python
import logistics_ga as lg
p = lg.load_problem("data/customers.csv", "data/warehouses.csv", truck_capacity=100,
                    cost_per_km=40, cost_per_truck=2000)
result = lg.solve(p, lg.GAParams(seed=1))
print(result.best.report(p))
lg.plot_plan(p, result.best)
```

CSV formats: customers `id,x,y,demand`; warehouses `id,x,y,stock,fixed_cost,trucks`. A blank `stock` or `trucks` means no limit. For road distances, pass `distance_matrix=` to `lg.Problem`.

## Results (seed 7, as in the notebook)

**The report's 10 case studies, re-run (distance in km):**

| Case | Stores | Report | New GA | Saving |
|---:|---:|---:|---:|---:|
| 1 | 4 | 271.77 | 271.77 | 0% (proven optimal) |
| 2 | 7 | 1,155.19 | 1,155.19 | 0% (proven optimal) |
| 3 | 10 | 24,908.65 | 23,518.55 | 5.6% |
| 4 | 13 | 58,382.25* | 57,418.28 | 1.7% |
| 5 | 16 | 15,029.14* | 13,816.53 | 8.1% |
| 6 | 19 | 134,207.21 | 113,908.76 | 15.1% |
| 7 | 25 | 452,609.99 | 385,976.94 | 14.7% |
| 8 | 30 | 6,782.40 | 5,025.71 | 25.9% |
| 9 | 35 | 18,104.84 | 16,846.93 | 6.9% |
| 10 | 40 | 8,300.26 | 4,152.15 | 50.0% |

\* Recomputed from the report's own per-vehicle distances. The stated totals (59,882.50 and 15,031.38) contain small arithmetic slips.

**Multi-depot with stock limits (30 stores, 4 warehouses):** 443 km against 495 km for a hand-made rule of thumb (10.5% less driving). WH-West cannot supply all its nearby stores, so the GA chooses which of them to serve from WH-Central.

**Network design:** ₹59,505/day against ₹69,861 for the rule of thumb (14.8% cheaper). The cheapest plan closes the expensive central warehouse and runs the three regional ones. As warehouse rent rises, the best network shrinks from 4 warehouses to 3 and then to 2.

Each GA run takes 1–9 seconds.

## How the GA works

* **Chromosome:** a giant tour of all stores, plus a warehouse gene per store.
* **Decoding:**
  * stores claim stock, and a store whose warehouse has run out goes to the nearest one that still has stock;
  * the **Split** procedure (Prins, 2004) cuts each warehouse's sequence into truck trips at the lowest cost;
  * orders larger than a truck get full-truck trips plus the remainder.

  So every chromosome is a valid plan and no repair is needed.
* **Operators:**
  * tournament selection and elitism;
  * order crossover (OX) for routes, spatial crossover for warehouse choices;
  * mutations: swap, reverse and move for routes; reassign, close, open and redesign for the network;
  * relocate + 2-opt local search on a share of the children (a memetic GA).
* **Stopping rule:** stop after N generations without improvement, or at a generation cap.

The notebook explains each step and compares it with the original code.

## Tests

```
python test_logistics_ga.py
```

The tests check four things:
- On 25 random small problems, the GA's answer matches brute force. The problems mix several warehouses, stock and fleet limits, running costs and oversized orders.
- Split with a truck limit matches an exhaustive search over every way to cut a tour.
- 500 random chromosomes all decode to valid plans.
- The report's 4-store example gives 271.77 km.

## Possible next steps

These follow the report's future research list:
- trucks of different sizes and costs (heterogeneous fleet);
- delivery time windows (VRPTW);
- road-network distances;
- two-echelon networks (warehouse → hub → store);
- multi-objective GAs that weigh cost against delivery time or CO₂.

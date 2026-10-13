"""Choose the shift experiment's load from the H100 measurements: the batch size where the
translation and math tables disagree most, and the predicted loss of using the wrong one.

  python pick_conc.py --repo $REPO --costs "$TUNE/costs_k*.json" --a $X/acc-$TAG-translation --b $X/acc-$TAG-math_reasoning --max-b 256
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--costs", required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--max-b", type=int, default=256)
a = ap.parse_args()
sys.path.insert(0, a.repo)
from dsd_tuning.core import expand, goodput_table, load_costs, load_tokens_per_step

costs = load_costs(a.costs)
ta, _ = load_tokens_per_step(f"{a.a}/acceptance.json"); tb, _ = load_tokens_per_step(f"{a.b}/acceptance.json")
Ta = json.load(open(f"{a.a}/schedule.json")); Tb = json.load(open(f"{a.b}/schedule.json"))
print("table A:", Ta); print("table B:", Tb)
ga, gb = goodput_table(costs, ta, a.max_b), goodput_table(costs, tb, a.max_b)
la, lb = expand(Ta, a.max_b), expand(Tb, a.max_b)
best = (0, None)
for B in range(8, a.max_b + 1, 8):
    ka, kb = la[B], lb[B]
    loss_b_on_a = 1 - gb[B][ka] / gb[B][kb]      # B traffic served with A's table
    loss_a_on_b = 1 - ga[B][kb] / ga[B][ka]      # A traffic served with B's table
    print(f"B={B:3d}  A-table k={ka}  B-table k={kb}  B-traffic on A-table -{loss_b_on_a*100:4.1f}%  A-traffic on B-table -{loss_a_on_b*100:4.1f}%")
    score = min(loss_b_on_a, loss_a_on_b)
    if score > best[0]:
        best = (score, B)
if best[1] is None:
    print("\nThe two tables agree everywhere: the shift experiment would show nothing on this GPU. Report that.")
else:
    print(f"\nLargest two-way disagreement at batch size {best[1]} (both directions lose >= {best[0]*100:.1f}%).")
    print(f"Use about {round(best[1] / 0.9)} open requests: the live batch ran at about 0.9x the open requests on the A10G.")
    print("Then check the first shift run's 'B=' column and adjust so the live batch sits in the disagreeing range.")

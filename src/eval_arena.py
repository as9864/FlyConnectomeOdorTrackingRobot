"""실험장 크기에 따른 냄새 의존도 평가 (중간논문 4장 표 2).

작은 실험장(4x2 m)에서는 냄새 없이 넓게 돌아다니기만 해도 냄새원에 닿을 수 있다. 그래서 성공률만 보지 않고
'냄새 기여 = 정상 성공률 - 냄새 차단 성공률'을 함께 보고, 맹목 탐색이 통하지 않는 큰 실험장(8x4 m)에서도 잰다.
run_separation.py 가 고른 (odor_gain, cx_gate) 를 그대로 쓴다.

사용법: python3 src/eval_arena.py data [--trials 50]
결과: results/separation/arena.csv
"""
import argparse

import numpy as np
import pandas as pd

import connectome
from controllers import Braitenberg, ConnectomeController, SurgeCast
from run_separation import VARIANTS
from sim import run_trial

ARENAS = {"small_4x2": dict(arena=(4.0, 2.0), source=(0.3, 1.0), T=120.0),
          "large_8x4": dict(arena=(8.0, 4.0), source=(0.5, 2.0), T=240.0)}


class Blind:
    """센서 입력을 0으로 만들어 넘기는 래퍼: 냄새 없이 같은 제어기가 얼마나 찾는지(맹목 탐색 기준선)."""

    def __init__(self, ctl):
        self.ctl, self.name = ctl, ctl.name + "-blind"

    def reset(self):
        self.ctl.reset()

    def act(self, L, R, theta, dt):
        return self.ctl.act(0.0, 0.0, theta, dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    d = a.data
    files = (f"{d}/connections_princeton.csv.gz", f"{d}/classification.csv.gz")
    tuned = pd.read_csv("results/separation/summary.csv").drop_duplicates("variant").set_index("variant")
    ctls = [("baseline", SurgeCast()), ("baseline", Blind(SurgeCast())), ("baseline", Braitenberg())]
    for name, groups, iso in VARIANTS:
        g, gate = tuned.loc[name, "odor_gain"], tuned.loc[name, "cx_gate"]
        net = connectome.build_from_flywire(*files, groups, cell_types_csv=f"{d}/consolidated_cell_types.csv.gz",
                                            cx_gate=gate, isolate=iso)
        kw = dict(cast_osc=False, odor_gain=g, deriv_gain=g / 4)
        ctls += [(name, ConnectomeController(net, **kw)), (name, ConnectomeController(net, lesion=("ORN_L", "ORN_R"), **kw))]
    rows = []
    for aname, cfg in ARENAS.items():
        for group, ctl in ctls:
            res = []
            for s in range(1000, 1000 + a.trials):
                ok, t, tr = run_trial(ctl, seed=s, T=cfg["T"], arena=cfg["arena"], source=cfg["source"])
                path = np.sum(np.hypot(np.diff(tr[:, 1]), np.diff(tr[:, 2])))
                straight = np.hypot(tr[0, 1] - cfg["source"][0], tr[0, 2] - cfg["source"][1])
                res.append((ok, t, straight / max(path, 1e-6) if ok else 0.0))
            ok, t, eff = map(np.array, zip(*res))
            rows.append(dict(arena=aname, variant=group, controller=ctl.name, success=ok.mean(),
                             median_time=np.median(t), path_efficiency=eff.mean()))
            print(f"{aname:10s} {group:22s} {ctl.name:24s} {ok.mean():5.0%}  eff {eff.mean():.2f}", flush=True)
    pd.DataFrame(rows).to_csv("results/separation/arena.csv", index=False)


if __name__ == "__main__":
    main()

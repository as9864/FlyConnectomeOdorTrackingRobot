"""시뮬레이션 비교 실험.

사용법:
  python3 src/run_experiments.py                  # 임시 행렬로 전체 비교
  python3 src/run_experiments.py --trials 100     # 조건당 100회
  python3 src/run_experiments.py --flywire data/connections.csv data/classification.csv configs/groups.json
결과: results/baseline/summary.csv, results/baseline/trajectories.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import run_trial
from controllers import Braitenberg, SurgeCast, ConnectomeController
import connectome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--sensor-tau", type=float, default=1.0, help="MOX ~1-3 s, PID ~0.05 s")
    ap.add_argument("--wind", type=float, default=0.3)
    ap.add_argument("--flywire", nargs=3, metavar=("CONN", "CLASS", "GROUPS"))
    ap.add_argument("--cell-types", help="Codex consolidated_cell_types.csv.gz")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--no-cast", action="store_true", help="커넥톰 밖 탐색 진동자 끄기")
    ap.add_argument("--cx-gate", type=float, default=0.0, help="CX(PFL3) 문턱, 자유 파라미터")
    ap.add_argument("--odor-gain", type=float, default=8.0)
    ap.add_argument("--seed-offset", type=int, default=0, help="튜닝과 다른 시드로 평가할 때")
    args = ap.parse_args()

    net = connectome.build_from_flywire(*args.flywire, hops=args.hops, cell_types_csv=args.cell_types, cx_gate=args.cx_gate) if args.flywire else connectome.build_placeholder()
    print("connectome source:", net["source"])
    controllers = [
        Braitenberg(),
        SurgeCast(),
    ]
    kw = dict(cast_osc=not args.no_cast, odor_gain=args.odor_gain, deriv_gain=args.odor_gain / 4)
    lesions = [(), ("LHN",), ("MBON",), ("FBT",), ("MBON", "FBT"), ("PN_L", "PN_R"), ("CX_L", "CX_R"), ("ORN_L", "ORN_R")]  # 마지막: 냄새 차단 대조군
    controllers += [ConnectomeController(net, lesion=l, **kw) for l in lesions if all(n in net["nodes"] for n in l)]
    rows, examples = [], {}
    for ctl in controllers:
        for s in range(args.trials):
            ok, t, traj = run_trial(ctl, seed=s + args.seed_offset, sensor_tau=args.sensor_tau, wind=args.wind)
            path = np.sum(np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2])))
            straight = np.hypot(traj[0, 1] - 0.3, traj[0, 2] - 1.0)
            rows.append(dict(controller=ctl.name, seed=s, success=ok, time=t,
                             path_efficiency=straight / max(path, 1e-6) if ok else 0.0))
            if s < 5:
                examples.setdefault(ctl.name, []).append(traj)
        df = pd.DataFrame([r for r in rows if r["controller"] == ctl.name])
        print(f"{ctl.name:28s} success {df.success.mean():5.0%}  "
              f"median time {df[df.success].time.median() if df.success.any() else float('nan'):6.1f}s")

    os.makedirs("results/baseline", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/baseline/trials.csv", index=False)
    summary = df.groupby("controller").agg(success_rate=("success", "mean"),
                                           median_time=("time", "median"),
                                           path_efficiency=("path_efficiency", "mean"))
    summary.to_csv("results/baseline/summary.csv")

    n = len(examples)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 5.5), squeeze=False)
    for ax, (name, trajs) in zip(axes.flat, examples.items()):
        for tr in trajs:
            ax.plot(tr[:, 1], tr[:, 2], lw=0.8)
            ax.plot(tr[0, 1], tr[0, 2], "k.", ms=4)
        ax.plot(0.3, 1.0, "r*", ms=12)
        ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.set_title(name, fontsize=9)
        ax.annotate("", xy=(3.9, 0.15), xytext=(3.3, 0.15), arrowprops=dict(arrowstyle="->"))
        ax.text(3.3, 0.22, "wind", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/baseline/trajectories.png", dpi=130)
    print(summary)


if __name__ == "__main__":
    main()

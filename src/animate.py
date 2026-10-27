"""시뮬레이션 한 번을 GIF로 저장: 냄새 퍼프, 로봇 궤적, 뉴런 그룹 활동.

사용법:
  python3 src/animate.py --seed 1003 --out results/flywire/demo.gif \
    --flywire data/connections_princeton.csv.gz data/classification.csv.gz configs/groups.json \
    --cell-types data/consolidated_cell_types.csv.gz --cx-gate 0.6 --odor-gain 8
왼쪽: 커넥톰 제어기, 오른쪽: 같은 조건에서 냄새 차단(ORN 절제). 아래: 왼쪽 로봇의 뉴런 그룹 활동.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from sim import PuffPlume, Robot
from controllers import ConnectomeController
import connectome


def record(ctl, seed, T=120.0, dt=0.05, sensor_tau=1.0, every=4):
    rng = np.random.default_rng(seed)
    plume = PuffPlume(rng=rng)
    for _ in range(int(15 / dt)):
        plume.step(dt)
    start = (rng.uniform(2.8, 3.6), rng.uniform(0.4, 1.6), rng.uniform(-np.pi, np.pi))
    robot = Robot(*start, sensor_tau=sensor_tau, rng=rng)
    ctl.reset()
    frames, found = [], None
    for k in range(int(T / dt)):
        plume.step(dt)
        cL, cR = plume.concentration(robot.sensor_points())
        L, R = robot.sL.read(cL, dt), robot.sR.read(cR, dt)
        v, w = ctl.act(L, R, robot.theta, dt)
        robot.move(v, w, dt, plume.arena)
        if k % every == 0:
            frames.append(dict(t=k * dt, puffs=plume.pos.copy(), age=plume.age.copy(),
                               x=robot.x, y=robot.y, th=robot.theta, r=ctl.r.copy()))
        if found is None and np.hypot(robot.x - plume.source[0], robot.y - plume.source[1]) < 0.3:
            found = k * dt
            frames.append(frames[-1])
            break
    return frames, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--out", default="results/baseline/demo.gif")
    ap.add_argument("--flywire", nargs=3)
    ap.add_argument("--cell-types")
    ap.add_argument("--cx-gate", type=float, default=0.6)
    ap.add_argument("--odor-gain", type=float, default=8.0)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="영상 길이 상한(파일 크기 절약)")
    ap.add_argument("--dpi", type=int, default=70)
    args = ap.parse_args()
    net = (connectome.build_from_flywire(*args.flywire, cell_types_csv=args.cell_types, cx_gate=args.cx_gate)
           if args.flywire else connectome.build_placeholder())
    kw = dict(odor_gain=args.odor_gain, deriv_gain=args.odor_gain / 4, cast_osc=False)
    runs = [("Connectome (FlyWire)", *record(ConnectomeController(net, **kw), args.seed)),
            ("Odor blocked (ORN lesion)", *record(ConnectomeController(net, lesion=("ORN_L", "ORN_R"), **kw), args.seed))]
    n = min(max(len(r[1]) for r in runs), int(args.max_seconds / 0.2) + 1)
    nodes = [nd for nd in net["nodes"] if not nd.startswith("WIND")]
    idx = [net["nodes"].index(nd) for nd in nodes]

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axb = fig.add_subplot(gs[1, :])
    art = []
    for ax, (title, frames, found) in zip(axes, runs):
        ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.set_aspect("equal")
        ax.set_title(title + (f"  found at {found:.0f}s" if found else "  not found in 120s"), fontsize=10)
        ax.plot(0.3, 1.0, "r*", ms=14, zorder=5)
        ax.add_patch(plt.Circle((0.3, 1.0), 0.3, fill=False, ls="--", color="r", lw=0.8))
        ax.annotate("", xy=(3.95, 0.12), xytext=(3.35, 0.12), arrowprops=dict(arrowstyle="->"))
        ax.text(3.35, 0.18, "wind", fontsize=8)
        sc = ax.scatter([], [], s=12, c="tab:purple", alpha=0.25, lw=0)
        path, = ax.plot([], [], "-", color="tab:blue", lw=1.2)
        body, = ax.plot([], [], "o", color="k", ms=8)
        head, = ax.plot([], [], "-", color="orange", lw=2.5)
        tt = ax.text(0.05, 1.88, "", fontsize=9)
        art.append((sc, path, body, head, tt, frames))
    bars = axb.bar(range(len(nodes)), np.zeros(len(nodes)), color="tab:green")
    axb.set_xticks(range(len(nodes))); axb.set_xticklabels(nodes, rotation=30, fontsize=8)
    axb.set_ylim(0, 1.5); axb.set_ylabel("activity (left robot)")
    fig.tight_layout()

    def update(i):
        out = []
        for sc, path, body, head, tt, frames in art:
            f = frames[min(i, len(frames) - 1)]
            sc.set_offsets(f["puffs"] if len(f["puffs"]) else np.zeros((0, 2)))
            sc.set_sizes(8 + 60 * np.sqrt(f["age"]))
            hx = [fr["x"] for fr in frames[:min(i, len(frames) - 1) + 1]]
            hy = [fr["y"] for fr in frames[:min(i, len(frames) - 1) + 1]]
            path.set_data(hx, hy)
            body.set_data([f["x"]], [f["y"]])
            head.set_data([f["x"], f["x"] + 0.15 * np.cos(f["th"])], [f["y"], f["y"] + 0.15 * np.sin(f["th"])])
            tt.set_text(f"t = {f['t']:.0f} s")
            out += [sc, path, body, head, tt]
        r = art[0][5][min(i, len(art[0][5]) - 1)]["r"]
        for b, v in zip(bars, r[idx]):
            b.set_height(min(v, 1.5))
        return out + list(bars)

    anim = FuncAnimation(fig, update, frames=n, interval=66, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=15), dpi=args.dpi)
    print("saved", args.out, "frames", n, [r[2] for r in runs])


if __name__ == "__main__":
    main()

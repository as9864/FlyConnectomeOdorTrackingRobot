"""센서 연결 위치 대조 실험: 가스센서 신호를 후각 뉴런(ORN)이 아닌 다른 감각 뉴런에 넣으면 추적이 되는가.

V4(완전 분리) 회로에서 ORN_L/ORN_R 그룹만 다른 감각 분류로 바꾸고, 나머지 그룹은 그대로 둔다.
각 위치마다 튜닝 시드(0..29)에서 odor_gain x cx_gate 를 고른 뒤 평가 시드(1000..)에서 정상/센서 차단을 잰다.

사용법: python3 src/sensor_entry.py data [--trials 50]
결과: results/separation/sensor_entry.csv
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

import connectome
from controllers import ConnectomeController
from sim import run_trial

ENTRIES = ["olfactory", "hygrosensory", "thermosensory", "gustatory", "mechanosensory", "visual"]
GAINS, GATES = [4.0, 8.0, 16.0, 32.0], [0.0, 0.6]
ARENAS = {"small_4x2": dict(arena=(4.0, 2.0), source=(0.3, 1.0), T=120.0),
          "large_8x4": dict(arena=(8.0, 4.0), source=(0.5, 2.0), T=240.0)}


def rate(net, gain, lesion, seeds, **kw):
    ctl = ConnectomeController(net, lesion=lesion, cast_osc=False, odor_gain=gain, deriv_gain=gain / 4)
    return float(np.mean([run_trial(ctl, seed=s, **kw)[0] for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    d = a.data
    files = (f"{d}/connections_princeton.csv.gz", f"{d}/classification.csv.gz")
    base = json.load(open("configs/groups.json"))
    rows = []
    os.makedirs("results/separation", exist_ok=True)
    for entry in ENTRIES:
        g = dict(base)
        g["ORN_L"] = {"class": entry, "side": "left"}
        g["ORN_R"] = {"class": entry, "side": "right"}
        path = f"results/separation/groups.entry_{entry}.json"
        json.dump(g, open(path, "w"), indent=1)
        nets = {gt: connectome.build_from_flywire(*files, path, cell_types_csv=f"{d}/consolidated_cell_types.csv.gz",
                                                   cx_gate=gt) for gt in GATES}
        i = {n: k for k, n in enumerate(nets[0.0]["nodes"])}
        W = nets[0.0]["W"]
        best = max(((rate(nets[gt], gn, (), range(30)), gn, gt) for gn, gt in itertools.product(GAINS, GATES)))
        _, gn, gt = best
        row = dict(entry=entry, odor_gain=gn, cx_gate=gt, tune=best[0],
                   w_to_PN=W[i["PN_L"], i["ORN_L"]], w_to_LN=W[i["LN"], i["ORN_L"]])
        for an, cfg in ARENAS.items():
            seeds = range(1000, 1000 + a.trials)
            kw = dict(T=cfg["T"], arena=cfg["arena"], source=cfg["source"])
            row[f"{an}_intact"] = rate(nets[gt], gn, (), seeds, **kw)
            row[f"{an}_blocked"] = rate(nets[gt], gn, ("ORN_L", "ORN_R"), seeds, **kw)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv("results/separation/sensor_entry.csv", index=False)


if __name__ == "__main__":
    main()

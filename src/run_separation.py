"""커넥텀 분리 수준 비교 실험 (중간논문 4장).

같은 FlyWire 데이터에서 회로를 뽑는 방식만 바꿔, 냄새가 실제로 추적에 기여하는지 비교한다.
  V1 분리 없음:      클래스 단위 그룹(groups.coarse.json), 경로 분리
  V2 출력부만 분리:  LH/MB 는 클래스 전체, CX/DN 은 세포유형(groups.class_bridge.json), 경로 분리
  V3 경로 비분리:    세포유형 그룹(groups.json), 모델 뉴런 경유 허용(isolate=False)
  V4 완전 분리(제안): 세포유형 그룹(groups.json), 비모델 뉴런만 경유(isolate=True)
각 변형마다 튜닝 시드(0..29)에서 odor_gain x cx_gate 를 고르고, 평가 시드(1000..)에서 정상/냄새 차단/절제를 잰다.

사용법: python3 src/run_separation.py data [--trials 50]
결과: results/separation/summary.csv, tuning.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import connectome
from controllers import ConnectomeController
from sim import run_trial

VARIANTS = [
    ("V1_no_separation", "configs/groups.coarse.json", True),
    ("V2_output_only", "configs/groups.class_bridge.json", True),
    ("V3_types_no_isolation", "configs/groups.json", False),
    ("V4_full_separation", "configs/groups.json", True),
]
GAINS, GATES = [4.0, 8.0, 16.0], [0.0, 0.6, 1.2]


def rate(net, gain, lesion, seeds):
    ctl = ConnectomeController(net, lesion=lesion, cast_osc=False, odor_gain=gain, deriv_gain=gain / 4)
    res = [run_trial(ctl, seed=s)[:2] for s in seeds]
    return np.mean([ok for ok, _ in res]), np.median([t for _, t in res])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--tune-trials", type=int, default=30)
    a = ap.parse_args()
    d = a.data
    files = (f"{d}/connections_princeton.csv.gz", f"{d}/classification.csv.gz")
    tune_seeds, test_seeds = range(a.tune_trials), range(1000, 1000 + a.trials)
    rows, tune_rows = [], []
    for name, groups, iso in VARIANTS:
        nets = {g: connectome.build_from_flywire(*files, groups, cell_types_csv=f"{d}/consolidated_cell_types.csv.gz",
                                                  cx_gate=g, isolate=iso) for g in GATES}
        best = None
        for gain, gate in itertools.product(GAINS, GATES):
            sr, _ = rate(nets[gate], gain, (), tune_seeds)
            tune_rows.append(dict(variant=name, odor_gain=gain, cx_gate=gate, tune_success=sr))
            if best is None or sr > best[0]:
                best = (sr, gain, gate)
        _, gain, gate = best
        net = nets[gate]
        print(f"{name}: best gain {gain} gate {gate} (tune {best[0]:.0%})", flush=True)
        lesions = [(), ("ORN_L", "ORN_R"), ("PN_L", "PN_R"), ("LHN",), ("MBON",), ("FBT",), ("CX_L", "CX_R")]
        for les in lesions:
            if any(n not in groups_nodes(groups) for n in les):
                continue
            sr, mt = rate(net, gain, les, test_seeds)
            rows.append(dict(variant=name, lesion="-".join(les) or "none", odor_gain=gain, cx_gate=gate,
                             success=sr, median_time=mt))
            print(f"  {'-'.join(les) or 'intact':12s} {sr:5.0%}", flush=True)
    os.makedirs("results/separation", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/separation/summary.csv", index=False)
    pd.DataFrame(tune_rows).to_csv("results/separation/tuning.csv", index=False)


def groups_nodes(groups_json):
    import json
    return set(json.load(open(groups_json))) | {"WIND_L", "WIND_R"}


if __name__ == "__main__":
    main()

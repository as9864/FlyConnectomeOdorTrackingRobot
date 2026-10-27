"""냄새 이벤트 하나가 회로를 지나 바퀴 명령이 되는 과정을 숫자로 찍어 본다.

사용법:
  python3 src/trace_event.py data
상황: 로봇은 바람에 수직(북쪽, theta=+90°)을 보고 서 있고, 상풍은 로봇의 왼쪽이다.
0.5초에 왼쪽 센서에만 냄새가 들어오고 1.5초에 끊긴다. 같은 상황을 냄새 없이도 돌려 비교한다.
"""
import sys, os
import numpy as np
import connectome
from controllers import ConnectomeController

D = sys.argv[1] if len(sys.argv) > 1 else "data"
net = connectome.build_from_flywire(os.path.join(D, "connections_princeton.csv.gz"),
                                    os.path.join(D, "classification.csv.gz"), "configs/groups.json",
                                    cell_types_csv=os.path.join(D, "consolidated_cell_types.csv.gz"),
                                    cx_gate=0.6)
names = net["nodes"]
i = {n: k for k, n in enumerate(names)}
print("바람 입력 교차 여부(wind_crossed):", net.get("wind_crossed", False))
print("\n[주요 가중치 W[받는 쪽, 보내는 쪽]]")
for post, pre in [("PN_L", "ORN_L"), ("PN_R", "ORN_L"), ("FBT", "PN_L"), ("CX_L", "FBT"), ("CX_R", "FBT"),
                  ("CX_R", "WIND_L"), ("DN_L", "CX_R"), ("DN_L", "CX_L"), ("DN_FWD", "PN_L")]:
    print(f"  {pre:>6} -> {post:<6} {net['W'][i[post], i[pre]]:+.2f}")

cols = ["ORN_L", "PN_L", "FBT", "CX_L", "CX_R", "DN_L", "DN_R"]
for label, odor in [("냄새 있음", True), ("냄새 없음", False)]:
    ctl = ConnectomeController(net, odor_gain=8, deriv_gain=2, cast_osc=False)
    ctl.reset()
    print(f"\n[{label}]  t(s)   L     R   | " + " ".join(f"{c:>6}" for c in cols) + " |  omega   v")
    dt, theta = 0.05, np.pi / 2
    for k in range(int(2.5 / dt)):
        t = k * dt
        on = odor and 0.5 <= t < 1.5
        L, R = (0.30, 0.05) if on else (0.0, 0.0)
        v, w = ctl.act(L, R, theta, dt)
        if k % 2 == 0:
            r = ctl.r
            print(f"          {t:4.1f}  {L:.2f}  {R:.2f} | " + " ".join(f"{r[i[c]]:6.2f}" for c in cols)
                  + f" | {w:+6.2f} {v:.2f}")

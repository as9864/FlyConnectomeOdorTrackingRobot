"""비교용 제어기 (베이스라인 + 커넥톰 제어기).

모든 제어기는 act(L, R, theta, dt) -> (v [m/s], omega [rad/s]) 를 구현한다.
theta 는 로봇 방위(IMU), 상풍 방향은 sim.UPWIND_HEADING 으로 알고 있다고 가정한다(바람 방향 고정 실험장).
"""
import numpy as np

from sim import UPWIND_HEADING


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Braitenberg:
    """좌우 센서 차이로만 회전하는 단순 양측 화학주성."""
    name = "braitenberg"

    def __init__(self, v=0.12, k=3.0):
        self.v, self.k = v, k

    def reset(self):
        pass

    def act(self, L, R, theta, dt):
        return self.v, self.k * (L - R) / (L + R + 0.05)


class SurgeCast:
    """냄새 감지 시 상풍 돌진, 놓치면 바람에 수직으로 지그재그 탐색."""
    name = "surge_cast"

    def __init__(self, thr=0.05, hold=1.0, v=0.15, k=2.5):
        self.thr, self.hold, self.v, self.k = thr, hold, v, k

    def reset(self):
        self.t_lost = 1e9
        self.t_cast = 0.0
        self.base = 0.0

    def act(self, L, R, theta, dt):
        c = max(L, R)
        if c > self.thr:
            self.t_lost = 0.0
            self.base = max(self.base, c)
        else:
            self.t_lost += dt
        if self.t_lost < self.hold:
            target = UPWIND_HEADING
            self.t_cast = 0.0
        else:
            self.t_cast += dt
            period = 2.0 + self.t_cast * 0.5  # 탐색 폭을 점점 넓힘
            side = 1 if (self.t_cast % (2 * period)) < period else -1
            target = UPWIND_HEADING + side * np.deg2rad(80)
        return self.v, self.k * wrap(target - theta)


class ConnectomeController:
    """세포유형 단위 rate 모델: tau dr/dt = -r + relu(W r + b + input).

    W 와 노드 이름은 connectome.py 가 만든다(FlyWire 데이터 또는 임시 행렬).
    입력 노드: ORN_L, ORN_R (가스센서), WIND_L, WIND_R (상풍이 왼쪽/오른쪽에 있음).
    출력 노드: DN_L, DN_R (회전), DN_FWD (전진).
    """
    name = "connectome"

    def __init__(self, net, odor_gain=8.0, deriv_gain=2.0, k_turn=3.0, v_max=0.18, v_min=0.06,
                 lesion=(), cast_osc=True):
        self.net = net
        self.idx = {n: i for i, n in enumerate(net["nodes"])}
        self.W = net["W"].copy()
        self.b = net["b"].copy()
        self.tau = net["tau"].copy()
        for n in lesion:  # 절제 실험: 해당 노드의 출력을 0으로
            self.W[:, self.idx[n]] = 0.0
        self.odor_gain, self.deriv_gain = odor_gain, deriv_gain
        self.k_turn, self.v_max, self.v_min = k_turn, v_max, v_min
        self.cast_osc = cast_osc
        self.lesion = tuple(lesion)
        if lesion:
            self.name = "connectome-" + "-".join(lesion)

    def reset(self):
        self.r = np.zeros(len(self.idx))
        self.prev = (0.0, 0.0)
        self.phase = 0.0

    def act(self, L, R, theta, dt):
        u = np.zeros_like(self.r)
        dL = max(0.0, (L - self.prev[0]) / dt)
        dR = max(0.0, (R - self.prev[1]) / dt)
        self.prev = (L, R)
        u[self.idx["ORN_L"]] = self.odor_gain * L + self.deriv_gain * dL
        u[self.idx["ORN_R"]] = self.odor_gain * R + self.deriv_gain * dR
        e = wrap(UPWIND_HEADING - theta)  # +면 상풍이 왼쪽
        u[self.idx["WIND_L"]] = max(0.0, np.sin(e)) + 0.5 * max(0.0, -np.cos(e))
        u[self.idx["WIND_R"]] = max(0.0, -np.sin(e))
        drive = self.W @ self.r + self.b + u
        self.r += dt / self.tau * (-self.r + np.maximum(0.0, drive))
        r = self.r
        omega = self.k_turn * (r[self.idx["DN_L"]] - r[self.idx["DN_R"]])
        v = max(self.v_min, self.v_max * np.tanh(r[self.idx["DN_FWD"]]))
        if self.cast_osc:
            # 커넥톰 밖의 구성요소(명시): 냄새 활동이 약할 때 좌우로 흔드는 진동자
            odor = r[self.idx["PN_L"]] + r[self.idx["PN_R"]]
            self.phase += dt * 2 * np.pi / 3.0
            omega += 1.5 * np.sin(self.phase) * np.exp(-3 * odor)
        return v, omega

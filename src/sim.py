"""2D 냄새 플룸 + 차동구동 로봇 + MOX 센서 시뮬레이션.

좌표: 바람은 +x 방향으로 분다. 냄새원은 왼쪽(x 작음)에 있고 로봇은 바람 아래쪽(x 큼)에서 출발한다.
따라서 상풍(upwind) 방향은 heading = pi.
"""
import numpy as np

UPWIND_HEADING = np.pi


class PuffPlume:
    """퍼프(puff) 기반 간헐적 냄새 플룸 (Farrell et al. 2002 방식을 단순화)."""

    def __init__(self, source=(0.3, 1.0), wind=0.3, release_hz=10.0, diffusion=0.002,
                 meander_sigma=0.08, meander_tau=2.0, sigma0=0.02, arena=(4.0, 2.0), rng=None):
        self.source = np.array(source, float)
        self.wind = wind
        self.release_dt = 1.0 / release_hz
        self.D = diffusion
        self.meander_sigma = meander_sigma
        self.meander_tau = meander_tau
        self.sigma0 = sigma0
        self.arena = arena
        self.rng = rng or np.random.default_rng()
        self.pos = np.zeros((0, 2))
        self.vy = np.zeros(0)
        self.age = np.zeros(0)
        self.v_global = 0.0  # 공유되는 측면 바람(사행) 성분, OU 과정
        self._t_since_release = 0.0

    def step(self, dt):
        # 측면 바람 OU 과정
        a = dt / self.meander_tau
        self.v_global += -a * self.v_global + self.meander_sigma * np.sqrt(2 * a) * self.rng.standard_normal()
        self._t_since_release += dt
        while self._t_since_release >= self.release_dt:
            self._t_since_release -= self.release_dt
            self.pos = np.vstack([self.pos, self.source])
            self.vy = np.append(self.vy, 0.0)
            self.age = np.append(self.age, 0.0)
        # 퍼프 이동: 바람 + 측면 사행 + 개별 난류
        self.vy += -a * (self.vy - self.v_global) + 0.03 * np.sqrt(dt) * self.rng.standard_normal(len(self.vy))
        self.pos[:, 0] += self.wind * dt
        self.pos[:, 1] += self.vy * dt
        self.age += dt
        keep = (self.pos[:, 0] < self.arena[0] + 0.5)
        self.pos, self.vy, self.age = self.pos[keep], self.vy[keep], self.age[keep]

    def concentration(self, pts):
        pts = np.atleast_2d(pts)
        if len(self.pos) == 0:
            return np.zeros(len(pts))
        s2 = self.sigma0 ** 2 + 2 * self.D * self.age  # (P,)
        d2 = ((pts[:, None, :] - self.pos[None, :, :]) ** 2).sum(-1)  # (N,P)
        c = (np.exp(-d2 / (2 * s2)) * (self.sigma0 ** 2 / s2)).sum(-1)
        return c


class MoxSensor:
    """MOX 가스센서: 1차 지연 + 노이즈. tau=1~3 s(MOX), 0.05 s(PID)."""

    def __init__(self, tau=1.0, noise=0.01, rng=None):
        self.tau, self.noise = tau, noise
        self.rng = rng or np.random.default_rng()
        self.y = 0.0

    def read(self, c, dt):
        self.y += dt / self.tau * (c - self.y)
        return max(0.0, self.y + self.noise * self.rng.standard_normal())


class Robot:
    """차동구동 로봇. 센서 2개는 앞쪽 fwd, 좌우 ±lat 위치."""

    def __init__(self, x, y, theta, fwd=0.10, lat=0.10, sensor_tau=1.0, rng=None):
        self.x, self.y, self.theta = x, y, theta
        self.fwd, self.lat = fwd, lat
        self.sL = MoxSensor(sensor_tau, rng=rng)
        self.sR = MoxSensor(sensor_tau, rng=rng)

    def sensor_points(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        base = np.array([self.x + self.fwd * c, self.y + self.fwd * s])
        left = base + self.lat * np.array([-s, c])
        right = base - self.lat * np.array([-s, c])
        return np.vstack([left, right])

    def move(self, v, omega, dt, arena):
        self.theta = (self.theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
        self.x = np.clip(self.x + v * np.cos(self.theta) * dt, 0.0, arena[0])
        self.y = np.clip(self.y + v * np.sin(self.theta) * dt, 0.0, arena[1])


def run_trial(controller, seed=0, T=120.0, dt=0.05, sensor_tau=1.0, wind=0.3,
              success_radius=0.3, start=None, arena=(4.0, 2.0), source=(0.3, 1.0)):
    """arena/source 를 바꾸면 큰 실험장 평가(맹목 탐색으로는 못 찾는 조건)를 할 수 있다. 기본값은 기존 결과와 동일."""
    rng = np.random.default_rng(seed)
    plume = PuffPlume(source=source, wind=wind, arena=arena, rng=rng)
    for _ in range(int(max(15.0, (arena[0] - 0.5) / wind) / dt)):  # 플룸이 실험장을 채울 때까지 미리 진행 (기본 실험장은 15 s)
        plume.step(dt)
    if start is None:
        start = (rng.uniform(0.7 * arena[0], 0.9 * arena[0]), rng.uniform(0.2 * arena[1], 0.8 * arena[1]),
                 rng.uniform(-np.pi, np.pi))
    robot = Robot(*start, sensor_tau=sensor_tau, rng=rng)
    controller.reset()
    traj = []
    for k in range(int(T / dt)):
        plume.step(dt)
        cL, cR = plume.concentration(robot.sensor_points())
        L, R = robot.sL.read(cL, dt), robot.sR.read(cR, dt)
        v, omega = controller.act(L, R, robot.theta, dt)
        robot.move(v, omega, dt, plume.arena)
        traj.append((k * dt, robot.x, robot.y, robot.theta, L, R))
        if np.hypot(robot.x - plume.source[0], robot.y - plume.source[1]) < success_radius:
            return True, k * dt, np.array(traj)
    return False, T, np.array(traj)

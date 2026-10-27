# 초파리 커넥톰 기반 냄새 추적 지상 로봇

FlyWire 초파리 커넥톰(뇌 배선도)에서 냄새→조향 회로를 학습 없이 뽑아, 가스센서를 단 2륜 로봇이 냄새를 따라 냄새원을 찾게 하는 프로젝트입니다.
지금은 시뮬레이션 단계이고, 중간논문 「커넥텀 경로 분리가 냄새 추적에 미치는 영향」을 썼습니다. 다음 단계는 아두이노 로봇 실험입니다.

## 핵심 아이디어

- 회로: **냄새(ORN) → PN → FB 접선뉴런(FB2H_a, FB2H_b, FB5AB, FB1G) → PFL3 → DNa02(좌우 회전)**. 바람 방향은 WIND 노드로 PFL3에 넣습니다(데이터가 아닌 가정).
- 15개 노드의 rate 모델 `τ dr/dt = −r + ReLU(Wr + b + u)`. 가중치는 FlyWire 시냅스 수를 받는 뉴런의 총 입력으로 정규화하고 신경전달물질로 부호(ACh +, GABA·Glu −)를 붙여 최대 3홉까지 더한 값입니다.
- 주장: 커넥톰을 **세포유형 단위로 분리**하고 **경로를 분리**(중간 경유는 모델 밖 뉴런만 허용)해야 냄새가 실제로 추적에 기여하고, 절제 결과를 해부학 경로로 설명할 수 있습니다.

## 분리 수준 비교 (V1~V4)

| 조건 | 세포유형 분리 | 경로 분리 | 그룹 정의 |
| --- | --- | --- | --- |
| V1 분리 없음 | 없음(분류 단위) | 있음 | `configs/groups.coarse.json` |
| V2 출력부만 분리 | PFL3·DNa02·DNp09만 | 있음 | `configs/groups.class_bridge.json` |
| V3 경로 비분리 | 있음 | 없음 | `configs/groups.json` (`isolate=False`) |
| V4 완전 분리 (제안) | 있음 | 있음 | `configs/groups.json` |

## 주요 결과 (평가 시드 1000~1049, 조건당 50회)

| 방법 | 작은 실험장 4×2 m | 큰 실험장 8×4 m | 냄새 차단 시 (작은/큰) |
| --- | --- | --- | --- |
| surge-cast (손 설계 규칙) | 94% | 82% | 맹목 0% / 0% |
| Braitenberg | 2% | 0% | |
| V1 분리 없음 | 98% | 0% | 96% / 0% |
| V2 출력부만 | 52% | 24% | 48% / 26% |
| V3 경로 비분리 | 76% | 78% | 4% / 26% |
| **V4 완전 분리** | **66%** | **76%** | **4% / 26%** |

- V1은 작은 실험장에서 높지만 냄새를 막아도 96%라 냄새 추적이 아닙니다. 그래서 핵심 지표는 **냄새 기여도 = 정상 − 냄새 차단 성공률**입니다.
- V4 절제(작은 실험장): PN 6%, FB 접선뉴런 14%, PFL3 12%, LHN 66%, MBON 62%. PN→FBT→PFL3 경로가 핵심이고 LH·MB 출력은 필수가 아닙니다. V3는 PN을 막아도 54%가 남아 해부학적으로 맞지 않습니다.
- 센서 입력 위치 대조(`sensor_entry.py`): 후각에 넣을 때만 센서 기여도가 크고(+50%p, 큰 실험장), 습도·온도·미각·시각은 0 근처, 기계감각은 +24%p.
- 원본 수치: `results/separation/summary.csv`, `arena.csv`, `sensor_entry.csv`, `tuning.csv`. 애니메이션: `results/flywire/demo.gif`.

## 폴더 구조

```
src/          시뮬레이터, 제어기, 커넥톰 회로 생성, 실험 스크립트
configs/      노드 그룹 ↔ FlyWire 분류/세포유형 매핑(groups*.json)
data/         FlyWire Codex 원본 데이터 (git에 올리지 않음, 아래 참고)
results/
  baseline/   임시(손으로 정한) 행렬로 돌린 초기 비교
  flywire/    FlyWire 행렬 첫 실행, 다리 세포유형 점수, demo.gif
  separation/ V1~V4 비교, 실험장 크기, 센서 입력 위치 실험 (중간논문 결과)
paper/        중간논문 Word 원고(v1~v5, v5가 최신), figures/
```

| 파일 | 내용 |
| --- | --- |
| `src/sim.py` | 퍼프 기반 간헐 플룸(Farrell 2002 방식), MOX 센서(1차 지연 1 s + 노이즈), 차동구동 로봇, `run_trial()` |
| `src/controllers.py` | `Braitenberg`, `SurgeCast`, `ConnectomeController`(`lesion=`으로 절제: `W[:, j] = 0`) |
| `src/connectome.py` | `build_placeholder()`, FlyWire CSV에서 그룹 간 유효 연결을 구하는 `build_from_flywire()` |
| `src/find_bridge_types.py` | PN→PFL3 다리 세포유형 점수(FBT·MBON 선택 근거) |
| `src/run_experiments.py` | 제어기·절제 조건 비교 (초기 실험) |
| `src/run_separation.py` | V1~V4 튜닝·평가·절제 (논문 표/그림 3) |
| `src/eval_arena.py` | 실험장 크기별 비교 (논문 표 1) |
| `src/sensor_entry.py` | 센서를 다른 감각 뉴런에 연결하는 대조 실험 |
| `src/trace_event.py` | 냄새 이벤트 하나가 회로를 지나 바퀴 명령이 되는 과정 출력 |
| `src/animate.py` | 정상 회로 vs 냄새 차단 애니메이션 |

## 실행

모든 명령은 저장소 최상위 폴더에서 실행합니다.

```bash
pip install numpy scipy pandas matplotlib
python src/run_separation.py data --trials 50
python src/eval_arena.py data --trials 50
python src/sensor_entry.py data --trials 50
python src/trace_event.py data
```

임시 행렬 비교(데이터 불필요): `python src/run_experiments.py --trials 30`

### 데이터 준비

[FlyWire Codex](https://codex.flywire.ai) Download에서 아래 3개 파일을 받아 `data/`에 둡니다(합계 약 70 MB, `.gitignore`로 제외).

- `connections_princeton.csv.gz` (시냅스 5개 이상 뉴런 쌍 약 373만 개, 신경전달물질 예측 포함)
- `classification.csv.gz`
- `consolidated_cell_types.csv.gz`

## 하드웨어 계획 (최종논문)

- 아두이노 UNO 2륜 차동구동 키트, MQ-3 가스센서(앞쪽 좌우, 필요하면 가운데 1개 추가해 최대 3개), MPU6050 IMU, 블루투스 모듈. 부품비 약 5만 원.
- 회로는 노트북에서 돌리고 블루투스로 바퀴 속도(`v ∓ ω·b/2`)를 보냅니다. `ConnectomeController.act(L, R, theta, dt)`는 시뮬레이터와 독립이라 그대로 씁니다.
- 실험: 선풍기로 바람, 에탄올을 냄새원으로. V1 vs V4, 냄새 차단 대조군을 각 10회 이상.

## 관련 연구

- 로봇 냄새원 탐색: surge-cast, infotaxis(Vergassola 2007), 나방 모사 알고리즘(Sensors 2023), GADEN(Monroy 2017), Francis 2022 리뷰.
- 초파리 냄새 탐색 회로: Álvarez-Salvado 2018, Matheson 2022(FB 접선뉴런), Westeinde 2024·Mussells Pires 2024(PFL3), Rayshubskiy(DNa02).
- 커넥톰 기반 모델·제어: FlyWire(Dorkenwald 2024, Schlegel 2024), Shiu 2024 뇌 전체 LIF, NeuroMechFly v2, Jin & Sui 2025, binivin(MaleCNS 532뉴런, 학습된 출력층, 70%), Connectocopter(뇌 전체 모델, 냄새 입력 폭주로 0/4).

전체 비교와 참고문헌은 `paper/중간논문_커넥텀경로분리_v5.docx` 2장·5장에 있습니다.

## 남은 일과 한계

- 실제 로봇 실험, MQ-3 실측 응답 시간 반영(현재 1 s 가정).
- 다리(FBT) 선택의 순환 논리 해소: 무작위 세포유형 4개 대조 실험.
- 전진 노드(DNp09)가 억제돼 전진 속도가 최저값(0.06 m/s)에 고정됨.
- hemibrain·MaleCNS로 교차 검증. WIND→PFL3 가중치(1.0)와 고정 풍향은 가정입니다.
- 논문 4-3의 냄새 확산 분석에 쓴 `spread2.py`는 이 폴더에 없습니다.

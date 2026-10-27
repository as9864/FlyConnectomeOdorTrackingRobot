"""세포유형(그룹) 단위 연결 행렬 만들기.

1) build_placeholder(): FlyWire 데이터 없이 돌려볼 수 있는 임시 행렬. 연결 구조는 문헌의 회로도
   (ORN→PN→LH/MB→CX→DN)를 따르지만 가중치 값은 손으로 정한 것이다. 논문에는 쓰면 안 된다.
2) build_from_flywire(): FlyWire Codex CSV 에서 그룹 간 '유효 연결'(최대 k홉, 부호 포함)을 계산한다.
   그룹 정의는 groups.json 에서 읽는다. Codex 파일의 열 이름은 다운로드한 버전에 맞춰 확인할 것.
"""
import json
import numpy as np

NODES = ["ORN_L", "ORN_R", "LN", "PN_L", "PN_R", "LHN", "MBON", "FBT",
         "WIND_L", "WIND_R", "CX_L", "CX_R", "DN_L", "DN_R", "DN_FWD"]


def build_placeholder():
    i = {n: k for k, n in enumerate(NODES)}
    W = np.zeros((len(NODES), len(NODES)))
    b = np.zeros(len(NODES))
    tau = np.full(len(NODES), 0.05)

    def w(post, pre, val):
        W[i[post], i[pre]] = val

    w("PN_L", "ORN_L", 1.0); w("PN_R", "ORN_R", 1.0)
    w("LN", "ORN_L", 0.3); w("LN", "ORN_R", 0.3)
    w("PN_L", "LN", -0.5); w("PN_R", "LN", -0.5)
    w("LHN", "PN_L", 0.6); w("LHN", "PN_R", 0.6)
    w("MBON", "PN_L", 0.4); w("MBON", "PN_R", 0.4)
    tau[i["MBON"]] = 2.0  # 느린 노드: 냄새를 놓친 뒤에도 잠시 상풍 이동을 유지
    for side in "LR":
        w(f"CX_{side}", f"WIND_{side}", 1.0)
        w(f"CX_{side}", "LHN", 0.8)
        w(f"CX_{side}", "MBON", 0.8)
        b[i[f"CX_{side}"]] = -1.0  # 냄새 AND 바람일 때만 켜지는 문턱
        w(f"DN_{side}", f"CX_{side}", 1.5)
        w(f"DN_{side}", f"PN_{side}", 0.4)  # 양측 화학주성 경로
    w("DN_FWD", "LHN", 0.8); w("DN_FWD", "MBON", 0.6)
    b[i["DN_FWD"]] = 0.2
    return {"nodes": NODES, "W": W, "b": b, "tau": tau, "source": "placeholder"}


SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0}  # 초파리에서 글루탐산은 주로 억제(GluCl)


_CSV_CACHE = {}


def _read(path):
    import pandas as pd
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]


def build_from_flywire(connections_csv, classification_csv, groups_json, hops=3, scale=1.0, norm="row", cell_types_csv=None, cx_gate=0.0, isolate=True):
    """그룹 간 유효 연결 계산.

    A[post, pre] = sign(pre) * syn / (post 의 총 입력 시냅스)  (뉴런 단위, 희소 행렬)
    W_group = mean_over_post_group( sum_{k=1..hops} A^k @ G_pre )
    """
    import pandas as pd
    from scipy import sparse

    conn = _read(connections_csv)
    cls = _read(classification_csv)
    groups = json.load(open(groups_json))

    conn = conn.groupby(["pre_root_id", "post_root_id", "nt_type"], as_index=False)["syn_count"].sum()
    ids = pd.Index(pd.unique(np.concatenate([conn.pre_root_id.values, conn.post_root_id.values])))
    pre = ids.get_indexer(conn.pre_root_id)
    post = ids.get_indexer(conn.post_root_id)
    sign = conn.nt_type.map(SIGN).fillna(1.0).values
    total_in = np.bincount(post, weights=conn.syn_count.values, minlength=len(ids))
    val = sign * conn.syn_count.values / np.maximum(total_in[post], 1)
    A = sparse.csr_matrix((val, (post, pre)), shape=(len(ids), len(ids)))

    if cell_types_csv:  # Codex consolidated_cell_types: root_id, primary_type → cell_type 열로 붙임
        ct = _read(cell_types_csv)[["root_id", "primary_type"]].rename(columns={"primary_type": "cell_type"})
        cls = cls.drop(columns=["cell_type"], errors="ignore").merge(ct, on="root_id", how="left")
    cls = cls.set_index("root_id")
    names = list(groups.keys())
    G = np.zeros((len(ids), len(names)))
    for j, n in enumerate(names):
        sel = np.ones(len(cls), bool)
        for col, pattern in groups[n].items():
            sel &= cls[col].astype(str).str.fullmatch(pattern).values
        members = ids.get_indexer(cls.index[sel])
        members = members[members >= 0]
        if len(members) == 0:
            print(f"[경고] 그룹 {n} 에 해당하는 뉴런이 없음: {groups[n]}")
        G[members, j] = 1.0

    # 경유 뉴런은 모델 그룹에 속하지 않은 뉴런만 허용한다. 그래야 그룹 하나를 절제하면 그 그룹을 지나는
    # 경로가 실제로 끊긴다(예: ORN→PN→LH 경로가 ORN→LH 간선에 중복으로 들어가지 않음).
    # isolate=False 이면 모델 뉴런도 경유 가능(경로 비분리 비교용)
    hidden = (G.sum(1) == 0).astype(float)[:, None] if isolate else np.ones((len(ids), 1))
    acc = np.zeros_like(G)
    cur = A @ G
    acc += cur
    for _ in range(hops - 1):
        cur = A @ (cur * hidden)
        acc += cur
    sizes = np.maximum(G.sum(0), 1)
    W = (G.T @ acc) / sizes[:, None]  # W[post_group, pre_group]
    np.fill_diagonal(W, 0.0)
    if norm == "row":
        # 받는 그룹마다 (데이터에서 온) 입력 절댓값 합을 1로: 긴 경로에서 희석된 신호를 층마다 되살림
        W = scale * W / (np.abs(W).sum(1, keepdims=True) + 1e-12)
    else:
        W = scale * W / (np.abs(W).max() + 1e-12)

    net = build_placeholder()
    i = {n: k for k, n in enumerate(net["nodes"])}
    # 그룹에 정의되지 않은 노드(예: WIND_L/R 입력)의 연결은 임시 행렬 값을 그대로 둔다
    Wn = net["W"].copy()
    for a, pa in enumerate(names):
        for c, pc in enumerate(names):
            if pa in i and pc in i:
                Wn[i[pa], i[pc]] = W[a, c]
    # 바람(목표 방향) 입력을 어느 CX 쪽에 넣을지는 데이터로 정한다:
    # DN_L(왼쪽 회전)을 더 강하게 흥분시키는 CX 쪽이 "목표가 왼쪽" 입력을 받는다. (PFL3→DNa02는 반대쪽으로 투사)
    if Wn[i["DN_L"], i["CX_R"]] > Wn[i["DN_L"], i["CX_L"]]:
        Wn[i["CX_L"], i["WIND_L"]], Wn[i["CX_R"], i["WIND_R"]] = 0.0, 0.0
        Wn[i["CX_R"], i["WIND_L"]], Wn[i["CX_L"], i["WIND_R"]] = 1.0, 1.0
        net["wind_crossed"] = True
    # 임시 행렬의 '냄새 AND 바람' 문턱은 데이터 경로에선 재현되지 않으므로 끈다(논문에 명시할 것)
    net["b"][i["CX_L"]] = net["b"][i["CX_R"]] = -cx_gate  # 자유 파라미터: CX 문턱(0이면 게이팅 없음)
    net["W"] = Wn
    net["source"] = "flywire"
    return net

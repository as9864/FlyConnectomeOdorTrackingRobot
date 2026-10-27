"""PN(냄새) 입력을 받고 PFL3(CX 조향 출력)에 영향을 주는 세포유형 순위 매기기.

사용법: python3 src/find_bridge_types.py data
점수 = (PN에서 2홉 유효 입력) x (PFL3로 3홉 유효 출력), 세포유형 평균. groups.json의 MBON, FBT 선택 근거.
"""
import numpy as np, pandas as pd
from scipy import sparse
SIGN={"ACH":1.0,"GABA":-1.0,"GLUT":-1.0}
import sys, os
D = sys.argv[1] if len(sys.argv) > 1 else "data"
conn=pd.read_csv(os.path.join(D, "connections_princeton.csv.gz")).groupby(["pre_root_id","post_root_id","nt_type"],as_index=False).syn_count.sum()
cls=pd.read_csv(os.path.join(D, "classification.csv.gz")).merge(pd.read_csv(os.path.join(D, "consolidated_cell_types.csv.gz"))[["root_id","primary_type"]],on="root_id",how="left")
ids=pd.Index(pd.unique(np.concatenate([conn.pre_root_id.values,conn.post_root_id.values])))
pre=ids.get_indexer(conn.pre_root_id); post=ids.get_indexer(conn.post_root_id)
tot=np.bincount(post,weights=conn.syn_count.values,minlength=len(ids))
val=conn.nt_type.map(SIGN).fillna(1.0).values*conn.syn_count.values/np.maximum(tot[post],1)
A=sparse.csr_matrix((val,(post,pre)),shape=(len(ids),)*2)
cls=cls.set_index("root_id").reindex(ids)
def ind(mask):
    v=np.zeros(len(ids)); v[np.asarray(mask)]=1.0; return v
pn=ind(cls["class"].eq("ALPN").values)
pfl=ind(cls["primary_type"].eq("PFL3").values)
fin=np.zeros(len(ids)); x=pn
for k in range(2): x=A@x; fin+=x
fout=np.zeros(len(ids)); y=pfl
for k in range(3): y=A.T@y; fout+=y
df=pd.DataFrame({"type":cls.primary_type.values,"cls":cls["class"].values,"super":cls.super_class.values,"fin":fin,"fout":fout})
g=df.groupby(["type","cls"],dropna=False).agg(n=("fin","size"),fin=("fin","mean"),fout=("fout","mean")).reset_index()
g["score"]=g.fin*g.fout
g=g[g.n<=40]
pd.set_option("display.width",200)
print("== top positive bridges (PN excites, excites PFL3 or PN inhibits, inhibits)")
print(g.sort_values("score",ascending=False).head(20).to_string(index=False))
print("== MBON types")
print(g[g.cls=="MBON"].sort_values("score",ascending=False).head(10).to_string(index=False))
print("== LH classes")
print(g[g.cls.isin(["LHLN","LHCENT"])].sort_values("score",ascending=False).head(10).to_string(index=False))
g.to_csv("results/flywire/bridge_scores.csv",index=False)

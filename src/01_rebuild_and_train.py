# =====================================================================================
# RECOVERY RUN  —  session was wiped (no models, no X/Y). Rebuild everything from raw PTB-XL.
# Raw dataset confirmed at: /kaggle/input/datasets/khyeh0719/ptb-xl-dataset/.../1.0.1
# STEP 1 (fast)  : rebuild X.npy, Y.npy, split_index.csv  (deterministic, from raw)
# STEP 2 (heavy) : train 10 seeds -> models/resnet1d_seed*.pt  (RESUMABLE, saves each seed)
# Both cells are VERBATIM from the verified notebook (ecgecgecg.ipynb cells 78 & 85).
#
# >>> CRITICAL: press SAVE VERSION after STEP 1, and again after EVERY seed in STEP 2. <<<
# >>> To make files permanent: after this finishes, also add /kaggle/working as output    <<<
# >>> of a Saved Version, or export models/ + X.npy + Y.npy as a private Kaggle Dataset.   <<<
# After this completes, run kaggle_confirmatory_run.py (axes + injection + GLMM).
# =====================================================================================
!pip -q install wfdb tqdm >/dev/null 2>&1



# =====================================================================================
# STEP 1 — rebuild data from raw PTB-XL (verbatim cell 78)
# =====================================================================================
# ============ REBUILD STEP 1+2: metadata + signals (ORIGINAL logic, from mounted source) ============
import os, ast, json, numpy as np, pandas as pd, wfdb
from tqdm import tqdm

DATA_DIR = "/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
WORK     = "/kaggle/working"
CLASSES  = ["NORM","MI","STTC","CD","HYP"]

# ---- STEP 1: aggregate SCP codes -> 5 superclasses (official method) ----
db  = pd.read_csv(f"{DATA_DIR}/ptbxl_database.csv", index_col="ecg_id")
db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))
agg = pd.read_csv(f"{DATA_DIR}/scp_statements.csv", index_col=0)
agg = agg[agg.diagnostic == 1]                       # keep diagnostic statements only
def to_super(scp_dict):
    out=set()
    for code in scp_dict:
        if code in agg.index:
            out.add(agg.loc[code, "diagnostic_class"])
    return list(out)
db["superclass"] = db.scp_codes.apply(to_super)
# 5 multi-hot columns
for c in CLASSES:
    db[c] = db["superclass"].apply(lambda L: 1.0 if c in L else 0.0)
# official split
db["split"] = db["strat_fold"].apply(lambda f: "train" if f<=8 else ("val" if f==9 else "test"))
db = db.reset_index()                                 # ecg_id back as column
print("meta built:", db.shape, "| cols incl:", [c for c in CLASSES if c in db.columns])
db.to_csv(f"{WORK}/ptbxl_meta_processed.csv", index=False)

# ---- STEP 2: load 12-lead 100Hz signals, exclude no-superclass, standardize (train-only) ----
Y = db[CLASSES].values.astype(np.float32)
keep = Y.sum(1) > 0
print(f"Excluding {int((~keep).sum())} records with no superclass (kept {int(keep.sum())}).")
df = db[keep].reset_index(drop=True); Y = Y[keep]
print("Label matrix:", Y.shape, "| positives/class:", Y.sum(0).astype(int).tolist())
print("split counts:", df["split"].value_counts().to_dict())

def load_signal(fname):
    return wfdb.rdrecord(os.path.join(DATA_DIR, fname)).p_signal.astype(np.float32)  # (1000,12)

X = np.zeros((len(df), 1000, 12), np.float32)
for i, fn in enumerate(tqdm(df["filename_lr"].values, desc="loading signals")):
    X[i] = load_signal(fn)
print("Signal tensor:", X.shape, f"| ~{X.nbytes/1e9:.2f} GB")

tr = (df["split"]=="train").values
mean = X[tr].mean(axis=(0,1), keepdims=True)
std  = X[tr].std(axis=(0,1), keepdims=True) + 1e-8
Xn = (X - mean) / std
print("Standardized with TRAIN-only stats.")

np.save(f"{WORK}/X.npy", Xn); np.save(f"{WORK}/Y.npy", Y)
df[["ecg_id","patient_id","strat_fold","split"]].to_csv(f"{WORK}/split_index.csv", index=False)
np.savez(f"{WORK}/norm_stats.npz", mean=mean, std=std)
for sp in ["train","val","test"]:
    m=(df["split"]==sp).values
    print(f"  {sp}: X={Xn[m].shape}, Y positives/class={Y[m].sum(0).astype(int).tolist()}")
print("\n✓ data rebuilt. Saved X.npy, Y.npy, split_index.csv, norm_stats.npz, ptbxl_meta_processed.csv")

# =====================================================================================
# STEP 2 — train 10 seeds, resumable (verbatim cell 85)
# =====================================================================================
# ============ STEP 3 (10 seeds): ResNet1D backbone — resumable, CPU-friendly ============
import os, json, numpy as np, torch, torch.nn as nn, pandas as pd, random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

WORK="/kaggle/working"; CLASSES=["NORM","MI","STTC","CD","HYP"]
os.makedirs(f"{WORK}/models", exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev)

X=np.load(f"{WORK}/X.npy"); Y=np.load(f"{WORK}/Y.npy")
idx=pd.read_csv(f"{WORK}/split_index.csv")
tr,va,te=(idx.split=="train").values,(idx.split=="val").values,(idx.split=="test").values
def mk(m): return torch.tensor(X[m].transpose(0,2,1)), torch.tensor(Y[m])
Xtr,Ytr=mk(tr); Xva,Yva=mk(va); Xte,Yte=mk(te)
dl_tr=DataLoader(TensorDataset(Xtr,Ytr),batch_size=128,shuffle=True)
dl_va=DataLoader(TensorDataset(Xva,Yva),batch_size=256)
dl_te=DataLoader(TensorDataset(Xte,Yte),batch_size=256)

class Block(nn.Module):
    def __init__(s,ci,co,k=7,st=1,down=None):
        super().__init__()
        s.c1=nn.Conv1d(ci,co,k,st,k//2,bias=False); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,1,k//2,bias=False);  s.b2=nn.BatchNorm1d(co)
        s.r=nn.ReLU(inplace=True); s.down=down
    def forward(s,x):
        i=x; o=s.r(s.b1(s.c1(x))); o=s.b2(s.c2(o))
        if s.down is not None: i=s.down(i)
        return s.r(o+i)
class ResNet1D(nn.Module):
    def __init__(s,ch=12,ncls=5,base=64,nblocks=(2,2,2,2)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(ch,base,15,2,7,bias=False),nn.BatchNorm1d(base),
                             nn.ReLU(inplace=True),nn.MaxPool1d(3,2,1))
        layers=[]; ci=base
        for li,nb in enumerate(nblocks):
            co=base*(2**li)
            for bi in range(nb):
                st=2 if (bi==0 and li>0) else 1; down=None
                if st!=1 or ci!=co:
                    down=nn.Sequential(nn.Conv1d(ci,co,1,st,bias=False),nn.BatchNorm1d(co))
                layers.append(Block(ci,co,7,st,down)); ci=co
        s.body=nn.Sequential(*layers)
        s.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(ci,ncls))
    def forward(s,x): return s.head(s.body(s.stem(x)))

pos=Ytr.sum(0); pw=((len(Ytr)-pos)/pos.clamp(min=1)).to(dev)
def evaluate(model,dl):
    model.eval(); P=[];T=[]
    with torch.no_grad():
        for xb,yb in dl:
            P.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy()); T.append(yb.numpy())
    P=np.concatenate(P);T=np.concatenate(T)
    return roc_auc_score(T,P,average="macro"), f1_score(T,(P>0.5),average="macro",zero_division=0), P,T

SEEDS=[42,1,7,2,3,4,5,6,8,9]
summary={}
for SEED in SEEDS:
    fp=f"{WORK}/models/resnet1d_seed{SEED}.pt"
    mp=f"{WORK}/models/metrics_seed{SEED}.json"
    if os.path.exists(fp) and os.path.exists(mp):
        summary[SEED]=json.load(open(mp)); print(f"seed {SEED} done (AUROC={summary[SEED]['test_macro_AUROC']}), skip"); continue
    print(f"\n=== SEED {SEED} ===")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model=ResNet1D().to(dev)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.3,patience=3)
    best=0;best_state=None;bad=0;patience=6;MAXEP=35
    for ep in range(1,MAXEP+1):
        model.train();tot=0
        for xb,yb in dl_tr:
            opt.zero_grad(); loss=crit(model(xb.to(dev)),yb.to(dev)); loss.backward(); opt.step()
            tot+=loss.item()*len(xb)
        va_auc,_,_,_=evaluate(model,dl_va); sched.step(va_auc)
        print(f"  ep{ep:02d} loss={tot/len(Xtr):.4f} val_AUROC={va_auc:.4f}")
        if va_auc>best: best=va_auc;best_state={k:v.cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=patience: print(f"  early stop @ ep{ep}"); break
    model.load_state_dict(best_state)
    te_auc,te_f1,Pte,Tte=evaluate(model,dl_te)
    per={c:round(roc_auc_score(Tte[:,i],Pte[:,i]),4) for i,c in enumerate(CLASSES)}
    torch.save(best_state,fp)
    met={"seed":SEED,"val_AUROC":round(best,4),"test_macro_AUROC":round(te_auc,4),
         "test_macro_F1":round(te_f1,4),"per_class_AUROC":per}
    json.dump(met,open(mp,"w"),indent=2); summary[SEED]=met
    print(f"  ✓ seed {SEED}: TEST macro-AUROC={te_auc:.4f}  (saved)")
    print(f"  >>> SAVE VERSION NOW before continuing <<<")

aucs=[summary[s]["test_macro_AUROC"] for s in SEEDS if s in summary]
print(f"\n=== {len(aucs)}/10 seeds done ===")
print(f"TEST macro-AUROC: mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  range=[{min(aucs):.4f}, {max(aucs):.4f}]")
json.dump(summary,open(f"{WORK}/models/seeds_summary.json","w"),indent=2)
print("✓ all seeds saved to models/")

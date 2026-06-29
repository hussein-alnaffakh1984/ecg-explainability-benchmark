# =====================================================================================
# ONE COMPREHENSIVE KAGGLE CELL  —  Paper 1 (ECG explanation benchmark) confirmatory run
# Covers reviewer items 1+2+3 in a SINGLE run, using the 10 SAVED models (NO retraining):
#   PART A : 4 verified pipeline functions (VERBATIM from notebook cell, gap=15 imputer)
#   PART B : 5-class x 6-method x 10-seed axes A/B/C   -> full Table 3 + Table 6 (10 seeds)
#   PART C : shortcut-injection detection, ALL 6 methods (adds DeepLIFT/GradXInput -> Table 8)
#   PART D : aggregate 10-seed -> tables with mean + 95% CI
#   PART E : NESTED MIXED MODEL (NEW code; the notebook never had a GLMM)  [see honesty note]
#
# PREREQUISITES (must already exist in /kaggle/working, restore from Drive cache if wiped):
#   models/resnet1d_seed{42,1,7,2,3,4,5,6,8,9}.pt   X.npy   Y.npy   split_index.csv
# Resumable: PART B skips seeds whose seed_*.json already exists.
# After it finishes: press SAVE VERSION (Kaggle wipes /kaggle/working between sessions).
# Standing rule honored: PARTS A/B/C are verbatim from the verified notebook; only PART E is new.
# =====================================================================================
!pip -q install captum neurokit2 statsmodels >/dev/null 2>&1
import os, gc, torch
os.makedirs("/kaggle/working/models", exist_ok=True)
print("CUDA:", torch.cuda.is_available(), "| seeds to run: [42,1,7,2,3,4,5,6,8,9]\n")

# ---------------- PREFLIGHT: verify prerequisites before any heavy compute ----------------
import glob as _glob
_WORK="/kaggle/working"
_need_seeds=[42,1,7,2,3,4,5,6,8,9]
_missing_models=[sd for sd in _need_seeds if not os.path.exists(f"{_WORK}/models/resnet1d_seed{sd}.pt")]
_missing_data=[f for f in ["X.npy","Y.npy","split_index.csv"] if not os.path.exists(f"{_WORK}/{f}")]
if _missing_data:
    raise SystemExit(f"STOP: missing data files {_missing_data}. Run the recovery script (STEP 1) first.")
if _missing_models:
    raise SystemExit(f"STOP: missing trained models for seeds {_missing_models}. "
                     f"Finish the training script (STEP 2) first; it is resumable.")
print(f"preflight OK: {len(_need_seeds)} models + data present.\n")




# =====================================================================================
# PART A — verified pipeline functions (verbatim)
# =====================================================================================
# ============ WIRING: 4 pipeline functions with ORIGINAL gap=15 imputer + MI verify ============
import numpy as np, torch, torch.nn as nn, copy, neurokit2 as nk, warnings, pandas as pd, gc
from scipy.stats import spearmanr
from captum.attr import (Saliency, InputXGradient, DeepLift, IntegratedGradients,
                         LayerGradCam, LayerAttribution, GradientShap)
warnings.filterwarnings("ignore")
if not hasattr(np,"trapz"): np.trapz=np.trapezoid
WORK="/kaggle/working"; dev="cuda" if torch.cuda.is_available() else "cpu"
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; SR=100

class Block(nn.Module):
    def __init__(s,ci,co,k=7,st=1,down=None):
        super().__init__()
        s.c1=nn.Conv1d(ci,co,k,st,k//2,bias=False); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,1,k//2,bias=False);  s.b2=nn.BatchNorm1d(co)
        s.r1=nn.ReLU(False); s.r2=nn.ReLU(False); s.down=down
    def forward(s,x):
        i=x;o=s.r1(s.b1(s.c1(x)));o=s.b2(s.c2(o))
        if s.down is not None:i=s.down(i)
        return s.r2(o+i)
class ResNet1D(nn.Module):
    def __init__(s,ch=12,ncls=5,base=64,nblocks=(2,2,2,2)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(ch,base,15,2,7,bias=False),nn.BatchNorm1d(base),nn.ReLU(False),nn.MaxPool1d(3,2,1))
        layers=[];ci=base
        for li,nb in enumerate(nblocks):
            co=base*(2**li)
            for bi in range(nb):
                st=2 if(bi==0 and li>0)else 1;down=None
                if st!=1 or ci!=co:down=nn.Sequential(nn.Conv1d(ci,co,1,st,bias=False),nn.BatchNorm1d(co))
                layers.append(Block(ci,co,7,st,down));ci=co
        s.body=nn.Sequential(*layers)
        s.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(ci,ncls))
    def forward(s,x): return s.head(s.body(s.stem(x)))

def load_model(seed):
    m=ResNet1D().to(dev)
    p=f"{WORK}/models/resnet1d_seed{seed}.pt"
    import os
    if not os.path.exists(p): p=f"{WORK}/resnet1d_baseline.pt"
    m.load_state_dict(torch.load(p,map_location=dev)); m.eval(); return m

# 1) attribute(method, model, x, cls_idx) -> (T,L) signed ; x is (T,L)
def attribute(method, model, x, cls_idx):
    xt=torch.tensor(x.T[None],dtype=torch.float32,device=dev).requires_grad_(True)  # (1,12,1000)
    if method=="Saliency":      a=Saliency(model).attribute(xt,target=cls_idx)
    elif method=="GradXInput":  a=InputXGradient(model).attribute(xt,target=cls_idx)
    elif method=="DeepLIFT":    a=DeepLift(model).attribute(xt,target=cls_idx)
    elif method=="IG":          a=IntegratedGradients(model).attribute(xt,target=cls_idx,baselines=xt*0,n_steps=32)
    elif method=="DeepSHAP":    a=GradientShap(model).attribute(xt,target=cls_idx,baselines=torch.cat([xt*0,xt*0+torch.randn_like(xt)*0.1]))
    elif method=="GradCAM":
        a=LayerGradCam(model,model.body[-1].c2).attribute(xt,target=cls_idx)
        a=LayerAttribution.interpolate(a,(1000,)).expand(-1,12,-1)
    elif method=="Random":      a=torch.randn_like(xt)
    return a.detach().cpu().numpy()[0].T   # (T,L)

# ---- ORIGINAL imputer (k=8, gap=15) : operates on (L,T) ----
def neigh_impute(sig, mask, k=8, gap=15):
    out=sig.copy(); T=sig.shape[1]; ia=np.arange(T)
    for l in range(sig.shape[0]):
        keep=~mask
        if keep.sum()==0: out[l,:]=0.0; continue
        ki=ia[keep]; kv=sig[l,keep]
        for ri in ia[mask]:
            j=np.searchsorted(ki,ri)
            lo=max(0,j-gap-k//2); a=max(0,j-gap); b=min(len(ki),j+gap); hi=min(len(ki),j+gap+k//2)
            pool=np.concatenate([kv[lo:a],kv[b:hi]])
            out[l,ri]=pool.mean() if len(pool)>0 else kv.mean()
    return out

STEPS=np.linspace(0.0,0.5,11)
@torch.no_grad()
def _prob(model,sig_LT,c):
    return torch.sigmoid(model(torch.tensor(sig_LT[None],dtype=torch.float32,device=dev)))[0,c].item()

# 2) road_abpc(model, X_batch, attrs, cls_idx, per_record=True) -> per-record ABPC
def road_abpc(model, X_batch, attrs, cls_idx, per_record=True):
    N=X_batch.shape[0]; per=[]
    for n in range(N):
        sig=X_batch[n].T            # (L,T)
        a=attrs[n]                  # (T,L)
        imp=np.abs(a).sum(1)        # (T,) sum over leads
        rank=np.argsort(imp); T=len(imp)
        morf=np.zeros(len(STEPS)); lerf=np.zeros(len(STEPS))
        for si,frac in enumerate(STEPS):
            nrem=int(frac*T)
            if nrem==0:
                morf[si]=lerf[si]=_prob(model,sig,cls_idx); continue
            mk_morf=np.zeros(T,bool); mk_morf[rank[::-1][:nrem]]=True
            mk_lerf=np.zeros(T,bool); mk_lerf[rank[:nrem]]=True
            morf[si]=_prob(model,neigh_impute(sig,mk_morf),cls_idx)
            lerf[si]=_prob(model,neigh_impute(sig,mk_lerf),cls_idx)
        per.append(float(np.trapz(lerf-morf,STEPS)))
    per=np.array(per)
    return per if per_record else float(per.mean())

# 3) randomization_curve(model, x, attr, cls_idx, signed=True) -> one C (signed)
_CUR=["DeepSHAP"]
def set_current_method(m): _CUR[0]=m
def _rand_top(model, k):
    mods=[(nm,mo) for nm,mo in model.named_modules() if isinstance(mo,(nn.Conv1d,nn.Linear))]
    rm=copy.deepcopy(model); names={nm for nm,_ in mods[-k:]} if k>0 else set()
    for nm,mo in rm.named_modules():
        if nm in names and isinstance(mo,(nn.Conv1d,nn.Linear)):
            nn.init.kaiming_normal_(mo.weight)
            if mo.bias is not None: nn.init.zeros_(mo.bias)
    rm.eval(); return rm
def randomization_curve(model, x, attr, cls_idx, signed=True):
    mods=[(nm,mo) for nm,mo in model.named_modules() if isinstance(mo,(nn.Conv1d,nn.Linear))]
    rm=_rand_top(model,len(mods))               # full randomization (top->all)
    ar=attribute(_CUR[0], rm, x, cls_idx)
    a0=(attr if signed else np.abs(attr)).ravel()
    a1=(ar   if signed else np.abs(ar)).ravel()
    del rm; gc.collect()
    if a0.std()<1e-9 or a1.std()<1e-9: return 0.0
    return float(spearmanr(a0,a1).correlation)

# 4) delineate(x) -> {'ok':bool,'windows':{'QRS':[(a,b),...],'ST':[...],'T':[...]}}  (all beats)
def delineate(x, lead=1):
    try:
        sig=nk.ecg_clean(x[:,lead],sampling_rate=SR)
        _,info=nk.ecg_peaks(sig,sampling_rate=SR); rp=info["ECG_R_Peaks"]
        if rp is None or len(rp)<2: return {"ok":False,"windows":{}}
        _,w=nk.ecg_delineate(sig,rp,sampling_rate=SR,method="dwt")
        W={"QRS":[],"ST":[],"T":[]}
        ron=w.get("ECG_R_Onsets",[]); roff=w.get("ECG_R_Offsets",[])
        ton=w.get("ECG_T_Onsets",[]); toff=w.get("ECG_T_Offsets",[])
        for a,b in zip(ron,roff):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["QRS"].append((int(a),int(b)))
        for a,b in zip(roff,ton):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["ST"].append((int(a),int(b)))
        for a,b in zip(ton,toff):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["T"].append((int(a),int(b)))
        ok=len(W["QRS"])>0
        return {"ok":ok,"windows":W}
    except Exception:
        return {"ok":False,"windows":{}}

print("✓ 4 functions defined (ORIGINAL gap=15 imputer)\n")
print("\u2713 4 pipeline functions ready (gap=15 imputer, 6 methods + Random)\n")


# =====================================================================================
# PART B — 10-seed x 6-method x 5-class axes (verbatim loop, SEEDS->10)
# =====================================================================================
# ============ PART B: 5-class x 6-method x 10-seed axes (from SAVED models; resumable) ============
import os, json, numpy as np, pandas as pd, torch, copy, gc
from scipy.stats import spearmanr
WORK="/kaggle/working"; os.makedirs(f"{WORK}/ext_seeds",exist_ok=True)
CLASSES=["NORM","MI","STTC","CD","HYP"]
METHODS=["Saliency","GradXInput","DeepLIFT","IG","GradCAM","DeepSHAP","Random"]
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

CLASS_REGION={
    "MI":   {"segments":["QRS","ST"],     "leads":"all"},
    "STTC": {"segments":["ST","T"],       "leads":"all"},
    "CD":   {"segments":["QRS"],          "leads":"all"},
    "HYP":  {"segments":["QRS","ST","T"], "leads":["V1","V2","V3","V4","V5","V6"]},
}
def axis_b_fraction(attr, cls, delin):
    if not delin["ok"]: return np.nan
    T=attr.shape[0]; spec=CLASS_REGION[cls]
    lead_idx=range(len(LEADS)) if spec["leads"]=="all" else [LEADS.index(l) for l in spec["leads"]]
    m=np.zeros(T,bool)
    for s in spec["segments"]:
        for (a,b) in delin["windows"].get(s,[]): m[a:b]=True
    pos=np.clip(attr,0,None)
    inr=sum(pos[m,l].sum() for l in lead_idx)
    tot=sum(pos[:,l].sum() for l in lead_idx)+1e-12
    return float(inr/tot)
def norm_dispersion(attr):
    a=np.abs(attr); a=a/(a.sum(0,keepdims=True)+1e-12)
    ent=-(a*np.log(a+1e-12)).sum(0)
    return float(ent.mean()/np.log(attr.shape[0]))

def randomize_half_model(model):
    mods=[(nm,mo) for nm,mo in model.named_modules() if isinstance(mo,(torch.nn.Conv1d,torch.nn.Linear))]
    rm=copy.deepcopy(model); names={nm for nm,_ in mods[len(mods)//2:]}
    for nm,mo in rm.named_modules():
        if nm in names and isinstance(mo,(torch.nn.Conv1d,torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(mo.weight)
            if mo.bias is not None: torch.nn.init.zeros_(mo.bias)
    rm.eval(); return rm

X=np.load(f"{WORK}/X.npy"); Y=np.load(f"{WORK}/Y.npy"); idx=pd.read_csv(f"{WORK}/split_index.csv")
te=(idx.split=="test").values; Xte=X[te]; Yte=Y[te]; pid=idx[te].patient_id.values
rng=np.random.default_rng(42); uniq=np.unique(pid); rng.shuffle(uniq); keep=set(uniq[:300])
sel=np.where([p in keep for p in pid])[0]

# delineation cache (signal-only, seed-independent) — saved once
dcache=f"{WORK}/ext_delineation.json"
if os.path.exists(dcache):
    DELN={int(k):v for k,v in json.load(open(dcache)).items()}
    print("loaded delineation cache")
else:
    print("delineating subsample (once)...")
    DELN={}
    for k,i in enumerate(sel):
        DELN[int(i)]=delineate(Xte[i])
        if (k+1)%100==0: print(f"  {k+1}/{len(sel)}")
    json.dump({str(k):v for k,v in DELN.items()},open(dcache,"w"))
ok_rate=np.mean([DELN[int(i)]["ok"] for i in sel]); print(f"delineation ok-rate={ok_rate:.2f}\n")

C_SUB=60; SEEDS=[42,1,7,2,3,4,5,6,8,9]
for sd in SEEDS:
    fp=f"{WORK}/ext_seeds/seed_{sd}.json"
    if os.path.exists(fp): print(f"seed {sd} done, skip"); continue
    print(f"=== SEED {sd} ===")
    m=load_model(sd)
    res={"seed":sd,"A":{},"B":{},"C":{},"A_records":{},"A_pids":{}}
    for ci,cls in enumerate(CLASSES):
        pos=sel[Yte[sel,ci]==1]
        if len(pos)<8: continue
        p2r={int(p):r for r,p in enumerate(pos)}
        for method in METHODS:
            attrs=np.stack([attribute(method,m,Xte[i],ci) for i in pos])
            ab=road_abpc(m,Xte[pos],attrs,ci,per_record=True)
            res["A"].setdefault(cls,{})[method]=float(ab.mean())
            res["A_records"].setdefault(cls,{})[method]=ab.tolist()
            if cls=="NORM":
                b=np.mean([norm_dispersion(attrs[j]) for j in range(len(pos))])
            else:
                vals=[axis_b_fraction(attrs[j],cls,DELN[int(pos[j])]) for j in range(len(pos))]
                vals=[v for v in vals if not np.isnan(v)]
                b=np.mean(vals) if vals else np.nan
            res["B"].setdefault(cls,{})[method]=float(b)
            # Axis C: randomize HALF once, signed spearman per record (matches locked stat)
            rngc=np.random.default_rng(1000+sd); ksub=min(C_SUB,len(pos))
            subi=rngc.choice(pos,ksub,replace=False)
            rm=randomize_half_model(m); cvals=[]
            for i in subi:
                a0=attrs[p2r[int(i)]].ravel()
                a1=attribute(method,rm,Xte[i],ci).ravel()
                if a0.std()>1e-9 and a1.std()>1e-9: cvals.append(spearmanr(a0,a1).correlation)
            res["C"].setdefault(cls,{})[method]=float(np.nanmean(cvals))
            del rm; gc.collect(); torch.cuda.empty_cache()
        res["A_pids"][cls]=pid[pos].tolist()
        print(f"  {cls}: done ({len(pos)} pos)")
    json.dump(res,open(fp,"w"),indent=2)
    print(f"  ✓ saved seed_{sd}.json\n")
print("✓ extension pass-1 done — run the summary cell next.")

# =====================================================================================
# PART C — shortcut injection, all 6 methods (verbatim + DeepLIFT/GradXInput)
# =====================================================================================
# ============ PART C: shortcut-injection detection, ALL 6 methods (adds DeepLIFT/GradXInput) ============
import os, json, numpy as np, torch, torch.nn as nn, pandas as pd, gc
from sklearn.metrics import roc_auc_score, roc_curve
from captum.attr import IntegratedGradients, GradientShap, Saliency, DeepLift, InputXGradient, LayerGradCam, LayerAttribution
if not hasattr(np,"trapz"): np.trapz=np.trapezoid
WORK="/kaggle/working"; dev="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
torch.cuda.empty_cache(); gc.collect()

class Block(nn.Module):
    def __init__(s,ci,co,k=7,st=1,down=None):
        super().__init__()
        s.c1=nn.Conv1d(ci,co,k,st,k//2,bias=False); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,1,k//2,bias=False);  s.b2=nn.BatchNorm1d(co)
        s.r1=nn.ReLU(False); s.r2=nn.ReLU(False); s.down=down
    def forward(s,x):
        i=x;o=s.r1(s.b1(s.c1(x)));o=s.b2(s.c2(o))
        if s.down is not None:i=s.down(i)
        return s.r2(o+i)
class ResNet1D(nn.Module):
    def __init__(s,ch=12,ncls=5,base=64,nblocks=(2,2,2,2)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(ch,base,15,2,7,bias=False),nn.BatchNorm1d(base),nn.ReLU(False),nn.MaxPool1d(3,2,1))
        layers=[];ci=base
        for li,nb in enumerate(nblocks):
            co=base*(2**li)
            for bi in range(nb):
                st=2 if(bi==0 and li>0)else 1;down=None
                if st!=1 or ci!=co:down=nn.Sequential(nn.Conv1d(ci,co,1,st,bias=False),nn.BatchNorm1d(co))
                layers.append(Block(ci,co,7,st,down));ci=co
        s.body=nn.Sequential(*layers)
        s.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(ci,ncls))
    def forward(s,x): return s.head(s.body(s.stem(x)))

X=np.load(f"{WORK}/X.npy"); Y=np.load(f"{WORK}/Y.npy"); idx=pd.read_csv(f"{WORK}/split_index.csv")
tr=(idx.split=="train").values; te=(idx.split=="test").values
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
TGT=1; LEAD=LEADS.index("aVR"); T0,T1=40,80; AMP=8.0
def inject(Xa,Ya):
    Xc=Xa.copy(); pos=np.where(Ya[:,TGT]==1)[0]; Xc[pos,LEAD,T0:T1]+=AMP; return Xc
Xtr=X[tr].transpose(0,2,1); Ytr=Y[tr]; Xte=X[te].transpose(0,2,1); Yte=Y[te]
Xtr_inj=inject(Xtr,Ytr); Xte_inj=inject(Xte,Yte)

def train(Xa,Ya,epochs=12):
    m=ResNet1D().to(dev); pos=Ya.sum(0)
    pw=torch.tensor((len(Ya)-pos)/np.clip(pos,1,None),dtype=torch.float32,device=dev)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-4)
    Xt=torch.tensor(Xa); Yt=torch.tensor(Ya)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),128):
            b=perm[i:i+128]; opt.zero_grad()
            loss=crit(m(Xt[b].to(dev)),Yt[b].to(dev)); loss.backward(); opt.step()
    m.eval(); return m
print("training shortcut model..."); m=train(Xtr_inj,Ytr)

pos_idx=np.where(Yte[:,TGT]==1)[0][:120]; recs=Xte_inj[pos_idx]
def frac_on_spike(attr):
    a=np.abs(attr); tot=a.reshape(len(a),-1).sum(1)+1e-9
    return a[:,LEAD,T0:T1].reshape(len(a),-1).sum(1)/tot

# ---- BATCHED attribution (small batches + low n_steps = memory safe) ----
def attr_batched(name, recs, bs=16):
    out=np.zeros((len(recs),12,1000),np.float32)
    for i in range(0,len(recs),bs):
        xb=torch.tensor(recs[i:i+bs],dtype=torch.float32,device=dev).requires_grad_(True)
        if name=="integrated_grad": a=IntegratedGradients(m).attribute(xb,target=TGT,baselines=xb*0,n_steps=16)
        elif name=="deepshap": a=GradientShap(m).attribute(xb,target=TGT,baselines=torch.cat([xb*0,xb*0+torch.randn_like(xb)*0.1]))
        elif name=="saliency": a=Saliency(m).attribute(xb,target=TGT)
        elif name=="deeplift": a=DeepLift(m).attribute(xb,target=TGT)
        elif name=="gradxinput": a=InputXGradient(m).attribute(xb,target=TGT)
        elif name=="gradcam":
            a=LayerGradCam(m,m.body[-1].c2).attribute(xb,target=TGT); a=LayerAttribution.interpolate(a,(1000,)); a=a.expand(-1,12,-1)
        out[i:i+bs]=a.detach().cpu().numpy()
        del xb,a; torch.cuda.empty_cache()
    return out

pos_scores=frac_on_spike(attr_batched("integrated_grad",recs))
neg_scores=frac_on_spike(np.abs(recs))                      # |input| reference
neg_scores2=frac_on_spike(np.random.default_rng(0).standard_normal(recs.shape))
print(f"\nPOSITIVE (IG)      mean={pos_scores.mean():.4f}")
print(f"NEGATIVE (|input|) mean={neg_scores.mean():.4f}")
print(f"NEGATIVE (random)  mean={neg_scores2.mean():.4f}")

y_true=np.concatenate([np.ones(len(pos_scores)),np.zeros(len(neg_scores)+len(neg_scores2))])
y_score=np.concatenate([pos_scores,neg_scores,neg_scores2])
auc=roc_auc_score(y_true,y_score); fpr,tpr,thr=roc_curve(y_true,y_score)
J=tpr-fpr; tau=thr[np.argmax(J)]
print(f"\nseparation AUROC={auc:.4f}  calibrated τ_audit={tau:.4f}")

print(f"\n=== detection per method (τ={tau:.4f}) ===")
det={}
for nm in ["integrated_grad","deepshap","deeplift","gradxinput","saliency","gradcam"]:
    s=frac_on_spike(attr_batched(nm,recs)); rate=(s>tau).mean()
    det[nm]={"mean_frac":float(s.mean()),"detect_rate":float(rate)}
    print(f"  {nm:16s} mean-frac={s.mean():.4f}  detect={rate:.1%}  {'DETECTS' if rate>0.5 else 'MISSES (false trust)'}")
    torch.cuda.empty_cache()

json.dump({"tau_audit":float(tau),"separation_auc":float(auc),"detection":det},
          open(f"{WORK}/step24_calibration.json","w"),indent=2)
print("\n✓ saved step24_calibration.json")

# =====================================================================================
# PARTS D & E — aggregation + nested mixed model (D verbatim-style; E NEW)
# =====================================================================================
# ============ PART D: aggregate 10-seed axes -> full tables (mean, 95% CI) ============
import os, json, glob, numpy as np
WORK="/kaggle/working"
CLASSES=["NORM","MI","STTC","CD","HYP"]
METHODS=["Saliency","GradXInput","DeepLIFT","IG","GradCAM","DeepSHAP","Random"]

seed_files=sorted(glob.glob(f"{WORK}/ext_seeds/seed_*.json"))
print(f"aggregating {len(seed_files)} seed files: {[os.path.basename(f) for f in seed_files]}\n")
allres=[json.load(open(f)) for f in seed_files]
nseed=len(allres)

def ci95(vals):
    v=np.array([x for x in vals if x is not None and not (isinstance(x,float) and np.isnan(x))],float)
    if len(v)==0: return (np.nan,np.nan,np.nan)
    m=v.mean()
    if len(v)<2: return (m,m,m)
    se=v.std(ddof=1)/np.sqrt(len(v)); return (m, m-1.96*se, m+1.96*se)

# Axis A: mean ABPC per method per class across seeds (vs Random subtracted) + CI
print("=== AXIS A (ABPC vs Random), mean [95% CI] over %d seeds ==="%nseed)
A_table={}; 
for cls in CLASSES:
    A_table[cls]={}
    rand=[r["A"].get(cls,{}).get("Random",0.0) for r in allres]
    for meth in METHODS:
        if meth=="Random": continue
        raw=[r["A"].get(cls,{}).get(meth,np.nan) for r in allres]
        adj=[ (raw[i]-rand[i]) if raw[i] is not None else np.nan for i in range(nseed)]
        m,lo,hi=ci95(adj); A_table[cls][meth]={"mean":m,"lo":lo,"hi":hi}
    line=" | ".join(f"{meth}:{A_table[cls][meth]['mean']:+.3f}[{A_table[cls][meth]['lo']:+.3f},{A_table[cls][meth]['hi']:+.3f}]"
                     for meth in ["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM"])
    print(f"  {cls:5s}: {line}")

# Axis B and C aggregated similarly
B_table={}; C_table={}
for cls in CLASSES:
    B_table[cls]={}; C_table[cls]={}
    for meth in METHODS:
        if meth=="Random": continue
        bm,_,_=ci95([r["B"].get(cls,{}).get(meth,np.nan) for r in allres])
        cm,_,_=ci95([abs(r["C"].get(cls,{}).get(meth,np.nan)) if r["C"].get(cls,{}).get(meth) is not None else np.nan for r in allres])
        B_table[cls][meth]={"mean":bm}; C_table[cls][meth]={"mean_abs":cm}

# Headline pooled MI faithfulness (matches Table 3 framing) with per-record across seeds
print("\n=== AXIS A on MI, pooled per-record across seeds (Table 3 headline) ===")
mi_pooled={}
for meth in ["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM"]:
    recs=[]
    randrecs=[]
    for r in allres:
        recs += r.get("A_records",{}).get("MI",{}).get(meth,[])
        randrecs += r.get("A_records",{}).get("MI",{}).get("Random",[])
    recs=np.array(recs,float); 
    if len(recs):
        m=recs.mean(); se=recs.std(ddof=1)/np.sqrt(len(recs))
        mi_pooled[meth]={"mean":float(m),"lo":float(m-1.96*se),"hi":float(m+1.96*se),"n":int(len(recs))}
        print(f"  {meth:10s} {m:+.3f} [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]  (n={len(recs)})")

json.dump({"A":A_table,"B":B_table,"C":C_table,"MI_pooled":mi_pooled,"nseed":nseed},
          open(f"{WORK}/axes_10seed_summary.json","w"),indent=2)
print("\n\u2713 saved axes_10seed_summary.json")

# ============ PART E: NESTED MIXED MODEL (NEW code - never existed in notebook) ============
# Honesty note: the notebook never contained a GLMM. This is NEW analysis code.
# Outcome ABPC can be negative -> a strict Beta GLM is inappropriate for ABPC; we fit a
# Gaussian nested mixed model on per-record ABPC: fixed method*class, crossed random
# intercepts for seed and class. (If the pre-registration's "Beta" referred to the bounded
# Axis-B in-region fraction in [0,1], the optional Beta block below fits that.)
print("\n=== PART E: nested mixed model (method x class; random intercepts seed, class) ===")
try:
    import pandas as pd, numpy as np
    import statsmodels.formula.api as smf
    rows=[]
    for si,r in enumerate(allres):
        sd=r.get("seed",si)
        for cls in CLASSES:
            randrec=r.get("A_records",{}).get(cls,{}).get("Random",[])
            rr=np.array(randrec,float) if randrec else None
            for meth in ["Saliency","GradXInput","DeepLIFT","IG","GradCAM","DeepSHAP"]:
                vals=r.get("A_records",{}).get(cls,{}).get(meth,[])
                for j,v in enumerate(vals):
                    base = rr[j] if (rr is not None and j<len(rr)) else 0.0
                    rows.append({"abpc":float(v)-float(base),"method":meth,"cls":cls,"seed":str(sd)})
    df=pd.DataFrame(rows)
    print(f"  long dataframe: {len(df)} rows, {df['method'].nunique()} methods, {df['cls'].nunique()} classes, {df['seed'].nunique()} seeds")
    # crossed random intercepts for seed and class via variance components
    df["grp"]=1
    vcf={"seed":"0 + C(seed)", "cls":"0 + C(cls)"}
    md=smf.mixedlm("abpc ~ C(method, Treatment(reference='DeepSHAP'))*C(cls)", df,
                   groups="grp", vc_formula=vcf, re_formula="0")
    mdf=md.fit(method="lbfgs", maxiter=200, disp=False)
    print(mdf.summary())
    with open(f"{WORK}/mixedmodel_abpc.txt","w") as fh: fh.write(str(mdf.summary()))
    # marginal method means (averaged over classes), with simple contrast vs Grad-CAM
    print("\n  marginal mean ABPC by method (averaged over classes):")
    for meth in ["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM"]:
        mm=df[df.method==meth]["abpc"].mean(); print(f"    {meth:10s} {mm:+.3f}")
    json.dump({"note":"Gaussian nested mixed model on per-record ABPC; fixed method*class; random intercepts seed,class",
               "n_obs":int(len(df))}, open(f"{WORK}/mixedmodel_meta.json","w"),indent=2)
    print("\n\u2713 saved mixedmodel_abpc.txt")
except Exception as e:
    print("  [mixed model] could not fit:", repr(e))
    print("  -> ensure statsmodels installed:  !pip -q install statsmodels")

# ---- OPTIONAL Beta regression for bounded Axis-B in-region fraction (in [0,1]) ----
try:
    from statsmodels.othermod.betareg import BetaModel
    import pandas as pd, numpy as np, patsy
    brows=[]
    for r in allres:
        for cls in ["MI","STTC","CD","HYP"]:
            for meth in ["Saliency","GradXInput","DeepLIFT","IG","GradCAM","DeepSHAP"]:
                v=r["B"].get(cls,{}).get(meth,np.nan)
                if v is not None and not np.isnan(v):
                    v=min(max(v,1e-4),1-1e-4); brows.append({"frac":v,"method":meth,"cls":cls})
    bdf=pd.DataFrame(brows)
    if len(bdf)>20:
        y,Xd=patsy.dmatrices("frac ~ C(method, Treatment(reference='DeepSHAP'))+C(cls)", bdf, return_type="dataframe")
        bm=BetaModel(y,Xd).fit(disp=False)
        with open(f"{WORK}/betareg_axisB.txt","w") as fh: fh.write(str(bm.summary()))
        print("\u2713 saved betareg_axisB.txt (bounded Axis-B in-region fraction)")
except Exception as e:
    print("  [beta Axis-B] skipped:", repr(e))

print("\n================ ALL PARTS DONE ================")
print("Outputs in /kaggle/working:")
print("  ext_seeds/seed_*.json         (per-seed 6x5 A/B/C, 10 seeds)")
print("  axes_10seed_summary.json      (full tables, mean + 95% CI)")
print("  step24_calibration.json       (injection detection, 6 methods)")
print("  mixedmodel_abpc.txt           (nested mixed model)")
print("  betareg_axisB.txt             (optional Beta on Axis-B)")
print("REMEMBER: press Save Version after this completes.")

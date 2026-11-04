# =====================================================================================
# Axes A, B and C for ResNet1D: six attribution methods + random map, five superclasses, ten seeds.
#   Axis A  leakage-controlled ROAD / ABPC (neighbourhood imputer, k = 8, guard band g = 15)
#   Axis B  share of positive attribution inside delineated diagnostic regions (NeuroKit2)
#   Axis C  signed Spearman similarity after re-initializing the upper half of the network
# Input : X.npy, Y.npy, split_index.csv and models/resnet1d_seed*.pt (see docs/REPRODUCE.md)
# Output: ext_seeds/seed_*.json (one file per seed; the run is resumable)
# =====================================================================================
# Requirements: see requirements.txt (captum, neurokit2, ...)
import os, gc, glob, time, json, copy, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import neurokit2 as nk
from scipy.stats import spearmanr
from captum.attr import (Saliency, InputXGradient, DeepLift, IntegratedGradients,
                         LayerGradCam, LayerAttribution, GradientShap)
warnings.filterwarnings("ignore")
if not hasattr(np,"trapz"): np.trapz=np.trapezoid
dev="cuda" if torch.cuda.is_available() else "cpu"
print("CUDA:", torch.cuda.is_available())

# ---- locate permanent dataset ----
def _find(name):
    h=glob.glob(f"/kaggle/input/**/{name}", recursive=True); return h[0] if h else None
SRC=os.path.dirname(_find("X.npy")); MODELS_DIR=os.path.dirname(_find("resnet1d_seed42.pt"))
WORK="/kaggle/working"; os.makedirs(f"{WORK}/ext_seeds", exist_ok=True)
print("SRC:",SRC,"| MODELS:",MODELS_DIR)

CLASSES=["NORM","MI","STTC","CD","HYP"]
METHODS=["Saliency","GradXInput","DeepLIFT","IG","GradCAM","DeepSHAP","Random"]
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
    m=ResNet1D().to(dev); m.load_state_dict(torch.load(f"{MODELS_DIR}/resnet1d_seed{seed}.pt",map_location=dev)); m.eval(); return m

def attribute(method, model, x, cls_idx):
    xt=torch.tensor(x.T[None],dtype=torch.float32,device=dev).requires_grad_(True)
    if method=="Saliency":      a=Saliency(model).attribute(xt,target=cls_idx)
    elif method=="GradXInput":  a=InputXGradient(model).attribute(xt,target=cls_idx)
    elif method=="DeepLIFT":    a=DeepLift(model).attribute(xt,target=cls_idx)
    elif method=="IG":          a=IntegratedGradients(model).attribute(xt,target=cls_idx,baselines=xt*0,n_steps=32)
    elif method=="DeepSHAP":    a=GradientShap(model).attribute(xt,target=cls_idx,baselines=torch.cat([xt*0,xt*0+torch.randn_like(xt)*0.1]))
    elif method=="GradCAM":
        a=LayerGradCam(model,model.body[-1].c2).attribute(xt,target=cls_idx)
        a=LayerAttribution.interpolate(a,(1000,)).expand(-1,12,-1)
    elif method=="Random":      a=torch.randn_like(xt)
    return a.detach().cpu().numpy()[0].T

def neigh_impute(sig, mask, k=8, gap=15):
    # Vectorized version of the verified gap=15 neighbourhood imputer (identical rule;
    # verified max abs diff 3e-6 vs the original loop over 300 random tests, ~230x faster).
    out=sig.copy(); T=sig.shape[1]; ia=np.arange(T); keep=~mask
    if keep.sum()==0: out[:,:]=0.0; return out
    ki=ia[keep]; ri=ia[mask]
    if len(ri)==0: return out
    n=len(ki); kv=sig[:,keep]
    cs=np.concatenate([np.zeros((sig.shape[0],1)),np.cumsum(kv,1)],1)
    j=np.searchsorted(ki,ri)
    lo=np.maximum(0,j-gap-k//2); a=np.maximum(0,j-gap); b=np.minimum(n,j+gap); hi=np.minimum(n,j+gap+k//2)
    a=np.maximum(a,lo); hi=np.maximum(hi,b)
    ssum=(cs[:,a]-cs[:,lo])+(cs[:,hi]-cs[:,b]); cnt=(a-lo)+(hi-b)
    out[:,ri]=np.where(cnt>0, ssum/np.maximum(cnt,1), kv.mean(1,keepdims=True))
    return out
STEPS=np.linspace(0.0,0.5,11)
@torch.no_grad()
def _prob(model,sig_LT,c):
    return torch.sigmoid(model(torch.tensor(sig_LT[None],dtype=torch.float32,device=dev)))[0,c].item()
def road_abpc(model, X_batch, attrs, cls_idx):
    N=X_batch.shape[0]; per=[]
    for n in range(N):
        sig=X_batch[n].T; a=attrs[n]; imp=np.abs(a).sum(1); rank=np.argsort(imp); T=len(imp)
        morf=np.zeros(len(STEPS)); lerf=np.zeros(len(STEPS))
        for si,frac in enumerate(STEPS):
            nrem=int(frac*T)
            if nrem==0: morf[si]=lerf[si]=_prob(model,sig,cls_idx); continue
            mk_m=np.zeros(T,bool); mk_m[rank[::-1][:nrem]]=True
            mk_l=np.zeros(T,bool); mk_l[rank[:nrem]]=True
            morf[si]=_prob(model,neigh_impute(sig,mk_m),cls_idx)
            lerf[si]=_prob(model,neigh_impute(sig,mk_l),cls_idx)
        per.append(float(np.trapz(lerf-morf,STEPS)))
    return np.array(per)

CLASS_REGION={"MI":{"segments":["QRS","ST"],"leads":"all"},"STTC":{"segments":["ST","T"],"leads":"all"},
              "CD":{"segments":["QRS"],"leads":"all"},"HYP":{"segments":["QRS","ST","T"],"leads":["V1","V2","V3","V4","V5","V6"]}}
def axis_b_fraction(attr, cls, delin):
    if not delin["ok"]: return np.nan
    T=attr.shape[0]; spec=CLASS_REGION[cls]
    lead_idx=range(len(LEADS)) if spec["leads"]=="all" else [LEADS.index(l) for l in spec["leads"]]
    m=np.zeros(T,bool)
    for seg in spec["segments"]:
        for (a,b) in delin["windows"].get(seg,[]): m[a:b]=True
    pos=np.clip(attr,0,None)
    inr=sum(pos[m,l].sum() for l in lead_idx); tot=sum(pos[:,l].sum() for l in lead_idx)+1e-12
    return float(inr/tot)
def norm_dispersion(attr):
    a=np.abs(attr); a=a/(a.sum(0,keepdims=True)+1e-12)
    ent=-(a*np.log(a+1e-12)).sum(0); return float(ent.mean()/np.log(attr.shape[0]))
def randomize_half_model(model):
    mods=[(nm,mo) for nm,mo in model.named_modules() if isinstance(mo,(nn.Conv1d,nn.Linear))]
    rm=copy.deepcopy(model); names={nm for nm,_ in mods[len(mods)//2:]}
    for nm,mo in rm.named_modules():
        if nm in names and isinstance(mo,(nn.Conv1d,nn.Linear)):
            nn.init.kaiming_normal_(mo.weight)
            if mo.bias is not None: nn.init.zeros_(mo.bias)
    rm.eval(); return rm
def delineate(x, lead=1):
    try:
        sig=nk.ecg_clean(x[:,lead],sampling_rate=SR)
        _,info=nk.ecg_peaks(sig,sampling_rate=SR); rp=info["ECG_R_Peaks"]
        if rp is None or len(rp)<2: return {"ok":False,"windows":{}}
        _,w=nk.ecg_delineate(sig,rp,sampling_rate=SR,method="dwt")
        W={"QRS":[],"ST":[],"T":[]}
        ron=w.get("ECG_R_Onsets",[]); roff=w.get("ECG_R_Offsets",[]); ton=w.get("ECG_T_Onsets",[]); toff=w.get("ECG_T_Offsets",[])
        for a,b in zip(ron,roff):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["QRS"].append((int(a),int(b)))
        for a,b in zip(roff,ton):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["ST"].append((int(a),int(b)))
        for a,b in zip(ton,toff):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=1000: W["T"].append((int(a),int(b)))
        return {"ok":len(W["QRS"])>0,"windows":W}
    except Exception:
        return {"ok":False,"windows":{}}
print("functions ready (gap=15 imputer)\n")

# ---- data + subsample ----
X=np.load(f"{SRC}/X.npy"); Y=np.load(f"{SRC}/Y.npy"); idx=pd.read_csv(f"{SRC}/split_index.csv")
te=(idx.split=="test").values; Xte=X[te]; Yte=Y[te]; pid=idx[te].patient_id.values
rng=np.random.default_rng(42); uniq=np.unique(pid); rng.shuffle(uniq); keep=set(uniq[:300])
sel=np.where([p in keep for p in pid])[0]

# ---- delineation cache (once) ----
dcache=f"{WORK}/ext_delineation.json"
if os.path.exists(dcache):
    DELN={int(k):v for k,v in json.load(open(dcache)).items()}; print("loaded delineation cache")
else:
    print("delineating subsample (once)...")
    DELN={}
    for k,i in enumerate(sel):
        DELN[int(i)]=delineate(Xte[i])
        if (k+1)%100==0: print(f"  {k+1}/{len(sel)}")
    json.dump({str(k):v for k,v in DELN.items()},open(dcache,"w"))
print(f"delineation ok-rate={np.mean([DELN[int(i)]['ok'] for i in sel]):.2f}\n")

C_SUB=60; SEEDS=[42,1,7,2,3,4,5,6,8,9]
for sd in SEEDS:
    fp=f"{WORK}/ext_seeds/seed_{sd}.json"
    if os.path.exists(fp): print(f"seed {sd} done, skip"); continue
    t0=time.time(); print(f"=== SEED {sd} ===", flush=True)
    m=load_model(sd)
    res={"seed":sd,"A":{},"B":{},"C":{},"A_records":{},"A_pids":{}}
    for ci,cls in enumerate(CLASSES):
        pos=sel[Yte[sel,ci]==1]
        if len(pos)<8: continue
        p2r={int(p):r for r,p in enumerate(pos)}
        # SPEEDUP: build the randomized model ONCE per class (shared by all methods)
        rm=randomize_half_model(m)
        rngc=np.random.default_rng(1000+sd); ksub=min(C_SUB,len(pos))
        subi=rngc.choice(pos,ksub,replace=False)
        for method in METHODS:
            attrs=np.stack([attribute(method,m,Xte[i],ci) for i in pos])
            ab=road_abpc(m,Xte[pos],attrs,ci)
            res["A"].setdefault(cls,{})[method]=float(ab.mean())
            res["A_records"].setdefault(cls,{})[method]=ab.tolist()
            if cls=="NORM":
                b=np.mean([norm_dispersion(attrs[j]) for j in range(len(pos))])
            else:
                vals=[axis_b_fraction(attrs[j],cls,DELN[int(pos[j])]) for j in range(len(pos))]
                vals=[v for v in vals if not np.isnan(v)]; b=np.mean(vals) if vals else np.nan
            res["B"].setdefault(cls,{})[method]=float(b)
            cvals=[]
            for i in subi:
                a0=attrs[p2r[int(i)]].ravel(); a1=attribute(method,rm,Xte[i],ci).ravel()
                if a0.std()>1e-9 and a1.std()>1e-9: cvals.append(spearmanr(a0,a1).correlation)
            res["C"].setdefault(cls,{})[method]=float(np.nanmean(cvals)) if cvals else float("nan")
        res["A_pids"][cls]=pid[pos].tolist()
        print(f"  {cls}: done ({len(pos)} pos)  [{time.time()-t0:.0f}s]", flush=True)
        del rm; gc.collect(); torch.cuda.empty_cache()
    json.dump(res,open(fp,"w"),indent=2)
    print(f"  SAVED seed_{sd}.json  (seed took {time.time()-t0:.0f}s)\n", flush=True)

print("================ PART 1 DONE — axes saved ================")
print("Axes complete; next: 05_injection_gradcam_controls.py")

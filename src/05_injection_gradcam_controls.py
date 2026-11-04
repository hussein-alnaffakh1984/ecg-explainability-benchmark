# =====================================================================================
# Shortcut injection, Grad-CAM variants, control maps, ROAD curves, runtime and examples (ResNet1D).
#   * eight artefact settings (amplitude, width, position, low-frequency, QRS-like, physiological electrode)
#   * reliance check: MI AUROC with the artefact on MI records, removed, shifted, added to all records
#   * threshold calibrated on 120 records, detection evaluated on 200 disjoint records (Wilson 95% CI)
#   * Grad-CAM at blocks 2, 3, 4 and Grad-CAM++; control maps |x|, smooth random, QRS mask
#   * measured ROAD curves, per-method runtime, attribution maps on two test records
# Output: part2/ (inj_*.json, axisA_seed_*.json, runtime.json, examples.npz); resumable
# =====================================================================================
# Requirements: see requirements.txt (captum, neurokit2, ...)
import os, gc, glob, time, json, copy, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import neurokit2 as nk
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve
from captum.attr import (Saliency, InputXGradient, DeepLift, IntegratedGradients,
                         LayerGradCam, LayerAttribution, GradientShap)
warnings.filterwarnings("ignore")
if not hasattr(np,"trapz"): np.trapz=np.trapezoid
dev="cuda" if torch.cuda.is_available() else "cpu"
print("CUDA:", torch.cuda.is_available(), flush=True)

# ---------------- knobs ----------------
EPOCHS=12            # shortcut-model training epochs (same as original injection test)
N_CAL=120            # MI-positive test records used ONLY to calibrate tau
N_EVAL=200           # disjoint MI-positive test records used ONLY to evaluate detection
SEEDS_A=[42,1,7]     # seeds for the extra Axis-A runs (baselines, Grad-CAM variants, curves)

def _find(name):
    h=glob.glob(f"/kaggle/input/**/{name}", recursive=True); return h[0] if h else None
SRC=os.path.dirname(_find("X.npy")); MODELS_DIR=os.path.dirname(_find("resnet1d_seed42.pt"))
WORK="/kaggle/working"; OUT=f"{WORK}/part2"; os.makedirs(OUT, exist_ok=True)
print("SRC:",SRC,"| MODELS:",MODELS_DIR, flush=True)

CLASSES=["NORM","MI","STTC","CD","HYP"]; MI=1
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; SR=100; T=1000

# ---------------- model ----------------
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
# Grad-CAM target layers: end of each residual stage (temporal size after the stride-2 stem+pool and stages)
CAM_LAYERS={"GradCAM_L4":7,"GradCAM_L3":5,"GradCAM_L2":3}

# ---------------- attribution (batched; x: (B,12,T) numpy) ----------------
def gradcampp(model, layer, xb, tgt):
    acts={}; grads={}
    h1=layer.register_forward_hook(lambda m,i,o: acts.__setitem__("a",o))
    h2=layer.register_full_backward_hook(lambda m,gi,go: grads.__setitem__("g",go[0]))
    model.zero_grad(); out=model(xb); out[:,tgt].sum().backward()
    h1.remove(); h2.remove()
    A=acts["a"].detach(); g=grads["g"].detach()                    # (B,K,t)
    g2=g**2; g3=g**3; sA=A.sum(2,keepdim=True)
    alpha=g2/(2*g2+sA*g3+1e-8); alpha=torch.where(g!=0,alpha,torch.zeros_like(alpha))
    w=(alpha*torch.relu(g)).sum(2,keepdim=True)
    cam=torch.relu((w*A).sum(1,keepdim=True))                      # (B,1,t)
    return LayerAttribution.interpolate(cam,(T,))                  # nearest upsampling, as Grad-CAM

def attr_batch(name, model, X, tgt, bs=16, delin=None, idxs=None, rng=None):
    out=np.zeros((len(X),12,T),np.float32)
    for i in range(0,len(X),bs):
        xb=torch.tensor(X[i:i+bs],dtype=torch.float32,device=dev).requires_grad_(True)
        if name=="Saliency":     a=Saliency(model).attribute(xb,target=tgt)
        elif name=="GradXInput": a=InputXGradient(model).attribute(xb,target=tgt)
        elif name=="DeepLIFT":   a=DeepLift(model).attribute(xb,target=tgt)
        elif name=="IG":         a=IntegratedGradients(model).attribute(xb,target=tgt,baselines=xb*0,n_steps=32,internal_batch_size=64)
        elif name=="DeepSHAP":   a=GradientShap(model).attribute(xb,target=tgt,baselines=torch.cat([xb*0,xb*0+torch.randn_like(xb)*0.1]))
        elif name in CAM_LAYERS:
            a=LayerGradCam(model,model.body[CAM_LAYERS[name]].c2).attribute(xb,target=tgt)
            a=LayerAttribution.interpolate(a,(T,)).expand(-1,12,-1)
        elif name=="GradCAMpp_L4":
            a=gradcampp(model,model.body[7].c2,xb,tgt).expand(-1,12,-1)
        elif name=="Random":     a=torch.randn_like(xb)
        elif name=="SmoothRandom":
            r=np.random.default_rng(i).standard_normal(xb.shape)
            a=torch.tensor(gaussian_filter1d(r,10,axis=2),device=dev)
        elif name=="AbsX":       a=xb.abs()
        elif name=="QRSmask":
            m=np.zeros(xb.shape,np.float32)
            for j in range(len(xb)):
                d=delin.get(int(idxs[i+j])) if delin is not None else None
                if d and d["ok"]:
                    for (s0,s1) in d["windows"]["QRS"]: m[j,:,s0:s1]=1.0
            m+=1e-3*np.random.default_rng(i).random(m.shape)       # tie-break only
            a=torch.tensor(m,device=dev)
        out[i:i+len(xb)]=a.detach().float().cpu().numpy()
        del xb,a
    if dev=="cuda": torch.cuda.empty_cache()
    return out

# ---------------- data ----------------
X=np.load(f"{SRC}/X.npy"); Y=np.load(f"{SRC}/Y.npy"); idx=pd.read_csv(f"{SRC}/split_index.csv")
tr=(idx.split=="train").values; te=(idx.split=="test").values
Xtr=np.ascontiguousarray(X[tr].transpose(0,2,1)); Ytr=Y[tr]
Xte=np.ascontiguousarray(X[te].transpose(0,2,1)); Yte=Y[te]; pid=idx[te].patient_id.values
ns=_find("norm_stats.npz")
SD=np.load(ns)["sd"].reshape(-1) if ns else np.ones(12)
print("per-lead sd (normalisation):", np.round(SD,3), flush=True)

# =====================================================================================
# PART 2A — SHORTCUT INJECTION
# =====================================================================================
def spike(shape, t0, t1, amp, kind="rect"):
    w=np.zeros(T,np.float32)
    if kind=="rect": w[t0:t1]=amp
    elif kind=="hann": w[t0:t1]=amp*np.hanning(t1-t0)
    elif kind=="gauss":
        c=(t0+t1)/2; s=(t1-t0)/6; tt=np.arange(T); w=(amp*np.exp(-0.5*((tt-c)/s)**2)).astype(np.float32)
    return w
# left-arm electrode artefact e(t): I=+e, II=0, III=-e, aVR=-e/2, aVL=+e, aVF=-e/2, V1..V6=-e/3
PHYS_COEF=np.array([1,0,-1,-0.5,1,-0.5]+[-1/3]*6,np.float32)
aVR=LEADS.index("aVR")
CONFIGS={
 "base_aVR_A8_W40":      dict(kind="rect", t0=40, t1=80,  amp=8.0, leads="aVR"),
 "lowamp_A2":            dict(kind="rect", t0=40, t1=80,  amp=2.0, leads="aVR"),
 "vlowamp_A0.5":         dict(kind="rect", t0=40, t1=80,  amp=0.5, leads="aVR"),
 "narrow_W10":           dict(kind="rect", t0=60, t1=70,  amp=8.0, leads="aVR"),
 "late_t600":            dict(kind="rect", t0=600,t1=640, amp=8.0, leads="aVR"),
 "wide_lowfreq_W200":    dict(kind="hann", t0=400,t1=600, amp=2.0, leads="aVR"),
 "morph_gauss_A2":       dict(kind="gauss",t0=51, t1=69,  amp=2.0, leads="aVR"),
 "physio_LAelectrode":   dict(kind="rect", t0=40, t1=80,  amp=8.0, leads="phys"),
}
def lead_profile(cfg):
    if cfg["leads"]=="aVR":
        p=np.zeros(12,np.float32); p[aVR]=1.0; return p
    # amplitude defined so that lead I receives `amp` normalised units; others follow the physics
    e_raw=cfg["amp"]*SD[0]
    return (PHYS_COEF*e_raw/SD).astype(np.float32)/cfg["amp"]
def add_spike(Xa, rows, cfg, shift=0):
    w=spike(None,cfg["t0"]+shift,min(T,cfg["t1"]+shift),cfg["amp"],cfg["kind"]) if shift else spike(None,cfg["t0"],cfg["t1"],cfg["amp"],cfg["kind"])
    p=lead_profile(cfg)
    Xa[rows]+= p[None,:,None]*w[None,None,:]
    return Xa
def window_mask(cfg):
    if cfg["kind"]=="hann": t0,t1=cfg["t0"],cfg["t1"]
    else: t0,t1=cfg["t0"],cfg["t1"]
    p=lead_profile(cfg); leads=np.where(np.abs(p)>0)[0]
    return t0,t1,leads

def train_shortcut(Xa,Ya,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m=ResNet1D().to(dev); pos=Ya.sum(0)
    pw=torch.tensor((len(Ya)-pos)/np.clip(pos,1,None),dtype=torch.float32,device=dev)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-4)
    Xt=torch.from_numpy(Xa); Yt=torch.from_numpy(Ya)
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),128):
            b=perm[i:i+128]; opt.zero_grad()
            loss=crit(m(Xt[b].to(dev)),Yt[b].to(dev)); loss.backward(); opt.step()
    m.eval(); return m
@torch.no_grad()
def predict(m,Xa,bs=256):
    P=[]
    for i in range(0,len(Xa),bs): P.append(torch.sigmoid(m(torch.from_numpy(Xa[i:i+bs]).to(dev))).cpu().numpy())
    return np.concatenate(P)
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-r)/d,(c+r)/d)
def boot_ci(v,B=1000,seed=0):
    r=np.random.default_rng(seed); v=np.asarray(v)
    b=[v[r.integers(0,len(v),len(v))].mean() for _ in range(B)]; return (float(np.percentile(b,2.5)),float(np.percentile(b,97.5)))

INJ_METHODS=["Saliency","GradXInput","DeepLIFT","IG","DeepSHAP",
             "GradCAM_L4","GradCAM_L3","GradCAM_L2","GradCAMpp_L4","Random","SmoothRandom","AbsX"]
rng0=np.random.default_rng(0)
mi_pos=np.where(Yte[:,MI]==1)[0]; rng0.shuffle(mi_pos)
cal_idx=mi_pos[:N_CAL]; ev_idx=mi_pos[N_CAL:N_CAL+N_EVAL]
print(f"MI-positive test records: {len(mi_pos)} -> calibration {len(cal_idx)}, evaluation {len(ev_idx)} (disjoint)\n", flush=True)

for name,cfg in CONFIGS.items():
    fp=f"{OUT}/inj_{name}.json"
    if os.path.exists(fp): print(f"[inject] {name} done, skip", flush=True); continue
    t0=time.time(); print(f"=== INJECTION CONFIG {name} ===", flush=True)
    Xtr_i=add_spike(Xtr.copy(), np.where(Ytr[:,MI]==1)[0], cfg)
    m=train_shortcut(Xtr_i,Ytr); del Xtr_i; gc.collect()
    # ---- reliance check (MI AUROC) ----
    mi_te=np.where(Yte[:,MI]==1)[0]
    Xa=add_spike(Xte.copy(), mi_te, cfg);                  au_spike=roc_auc_score(Yte[:,MI],predict(m,Xa)[:,MI])
    au_clean=roc_auc_score(Yte[:,MI],predict(m,Xte)[:,MI])
    sh=300 if cfg["t1"]+300<=T else -300
    Xs=add_spike(Xte.copy(), mi_te, cfg, shift=sh);        au_shift=roc_auc_score(Yte[:,MI],predict(m,Xs)[:,MI]); del Xs
    Xall=add_spike(Xte.copy(), np.arange(len(Xte)), cfg);  au_all=roc_auc_score(Yte[:,MI],predict(m,Xall)[:,MI]); del Xall
    print(f"  reliance  MI-AUROC: spike-on-MI={au_spike:.3f}  clean={au_clean:.3f}  shifted={au_shift:.3f}  spike-on-all={au_all:.3f}", flush=True)
    # ---- attribution fractions ----
    t0w,t1w,sleads=window_mask(cfg)
    chance_lead=(t1w-t0w)*len(sleads)/(T*12); chance_time=(t1w-t0w)/T
    def fracs(A):
        a=np.abs(A); tot=a.reshape(len(a),-1).sum(1)+1e-12
        lead=a[:,sleads,t0w:t1w].reshape(len(a),-1).sum(1)/tot
        tt=a.sum(1); time_=tt[:,t0w:t1w].sum(1)/(tt.sum(1)+1e-12)
        return lead,time_
    Xcal=Xa[cal_idx]; Xev=Xa[ev_idx]
    # calibration on the CALIBRATION records only: positive = IG maps, negatives = |x| and random maps
    ig_cal,_=fracs(attr_batch("IG",m,Xcal,MI))
    neg1,_=fracs(np.abs(Xcal)); neg2,_=fracs(np.random.default_rng(1).standard_normal(Xcal.shape))
    yt=np.r_[np.ones(len(ig_cal)),np.zeros(len(neg1)+len(neg2))]; ys=np.r_[ig_cal,neg1,neg2]
    cal_auc=roc_auc_score(yt,ys); fpr,tpr,thr=roc_curve(yt,ys); tau=float(thr[np.argmax(tpr-fpr)])
    cal_failed=(not np.isfinite(tau)) or cal_auc<=0.5          # shortcut not learned / not localisable
    if not np.isfinite(tau): tau=float(ys.max()+1e-9)            # finite stand-in -> zero detections
    tau_fixed=10*chance_lead                                   # a priori rule: 10x the lead-specific chance level
    res={"config":cfg|{"profile":lead_profile(cfg).tolist()},"n_cal":int(len(cal_idx)),"n_eval":int(len(ev_idx)),
         "reliance":{"auroc_spike_on_MI":au_spike,"auroc_clean":au_clean,"auroc_shifted":au_shift,"auroc_spike_on_all":au_all},
         "calibration":{"auroc_calibration_set":cal_auc,"tau_heldout":tau,"tau_fixed_10x_chance":tau_fixed,
                        "calibration_failed":bool(cal_failed)},
         "chance":{"lead_specific":chance_lead,"time_only":chance_time},"methods":{}}
    for meth in INJ_METHODS:
        A=attr_batch(meth,m,Xev,MI)
        lf,tf=fracs(A); n=len(lf)
        k1=int((lf>tau).sum()); k2=int((lf>tau_fixed).sum()); k3=int((tf>10*chance_time).sum())
        res["methods"][meth]={
          "lead_frac_mean":float(lf.mean()),"lead_frac_ci":boot_ci(lf),"lead_frac_x_chance":float(lf.mean()/chance_lead),
          "time_frac_mean":float(tf.mean()),"time_frac_ci":boot_ci(tf),"time_frac_x_chance":float(tf.mean()/chance_time),
          "detect_heldout_tau":k1/n,"detect_heldout_ci":wilson(k1,n),
          "detect_fixed10x":k2/n,"detect_fixed10x_ci":wilson(k2,n),
          "detect_timeonly_10x":k3/n,"detect_timeonly_ci":wilson(k3,n)}
        print(f"  {meth:13s} lead-frac={lf.mean():.3f} ({lf.mean()/chance_lead:6.1f}x chance)  "
              f"detect(held-out tau)={k1/n:5.1%}  detect(10x chance)={k2/n:5.1%}  time-only detect={k3/n:5.1%}", flush=True)
    json.dump(res,open(fp,"w"),indent=2)
    print(f"  SAVED inj_{name}.json  tau_heldout={tau:.4f} (calibration AUROC={cal_auc:.3f})  [{time.time()-t0:.0f}s]\n", flush=True)
    del m,Xa,Xcal,Xev; gc.collect()
    if dev=="cuda": torch.cuda.empty_cache()
print("PART 2A (injection) done\n", flush=True)

# =====================================================================================
# PART 2B — AXIS A: extra baselines + Grad-CAM variants + real ROAD curves + runtime
# =====================================================================================
def neigh_impute(sig, mask, k=8, gap=15):
    # vectorised verified gap=15 imputer (max abs diff 3e-6 vs the original loop)
    out=sig.copy(); ia=np.arange(sig.shape[1]); keep=~mask
    if keep.sum()==0: out[:,:]=0.0; return out
    ki=ia[keep]; ri=ia[mask]
    if len(ri)==0: return out
    n=len(ki); kv=sig[:,keep]
    cs=np.concatenate([np.zeros((sig.shape[0],1)),np.cumsum(kv,1)],1)
    j=np.searchsorted(ki,ri)
    lo=np.maximum(0,j-gap-k//2); a=np.maximum(0,j-gap); b=np.minimum(n,j+gap); hi=np.minimum(n,j+gap+k//2)
    a=np.maximum(a,lo); hi=np.maximum(hi,b)
    ssum=(cs[:,a]-cs[:,lo])+(cs[:,hi]-cs[:,b]); cnt=(a-lo)+(hi-b)
    out[:,ri]=np.where(cnt>0, ssum/np.maximum(cnt,1), kv.mean(1,keepdims=True)); return out
STEPS=np.linspace(0.0,0.5,11)
@torch.no_grad()
def _prob(model,sig,c): return torch.sigmoid(model(torch.tensor(sig[None],dtype=torch.float32,device=dev)))[0,c].item()
def road(model, Xb, A, c):
    per=[]; M=[]; L=[]
    for n in range(len(Xb)):
        sig=Xb[n]; imp=np.abs(A[n]).sum(0); rank=np.argsort(imp)
        mo=np.zeros(len(STEPS)); le=np.zeros(len(STEPS))
        for si,f in enumerate(STEPS):
            nr=int(f*T)
            if nr==0: mo[si]=le[si]=_prob(model,sig,c); continue
            km=np.zeros(T,bool); km[rank[::-1][:nr]]=True
            kl=np.zeros(T,bool); kl[rank[:nr]]=True
            mo[si]=_prob(model,neigh_impute(sig,km),c); le[si]=_prob(model,neigh_impute(sig,kl),c)
        per.append(float(np.trapz(le-mo,STEPS))); M.append(mo); L.append(le)
    return np.array(per),np.array(M),np.array(L)
def delineate(x, lead=1):
    try:
        sig=nk.ecg_clean(x[lead],sampling_rate=SR)
        _,info=nk.ecg_peaks(sig,sampling_rate=SR); rp=info["ECG_R_Peaks"]
        if rp is None or len(rp)<2: return {"ok":False,"windows":{}}
        _,w=nk.ecg_delineate(sig,rp,sampling_rate=SR,method="dwt")
        W={"QRS":[]}
        for a,b in zip(w.get("ECG_R_Onsets",[]),w.get("ECG_R_Offsets",[])):
            if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=T: W["QRS"].append((int(a),int(b)))
        return {"ok":len(W["QRS"])>0,"windows":W}
    except Exception: return {"ok":False,"windows":{}}

r=np.random.default_rng(42); uniq=np.unique(pid); r.shuffle(uniq); keep=set(uniq[:300])
sel=np.where([p in keep for p in pid])[0]
dfile=f"{OUT}/qrs_delineation.json"
if os.path.exists(dfile): DEL={int(k):v for k,v in json.load(open(dfile)).items()}
else:
    print("delineating QRS for the QRS-mask baseline...", flush=True)
    DEL={int(i):delineate(Xte[i]) for i in sel}; json.dump({str(k):v for k,v in DEL.items()},open(dfile,"w"))
print(f"QRS delineation ok-rate={np.mean([DEL[int(i)]['ok'] for i in sel]):.2f}", flush=True)

A_METHODS=["Saliency","GradXInput","DeepLIFT","IG","DeepSHAP","GradCAM_L4","GradCAM_L3","GradCAM_L2",
           "GradCAMpp_L4","Random","SmoothRandom","AbsX","QRSmask"]
for sd in SEEDS_A:
    fp=f"{OUT}/axisA_seed_{sd}.json"
    if os.path.exists(fp): print(f"[axisA] seed {sd} done, skip", flush=True); continue
    t0=time.time(); print(f"=== AXIS-A extras SEED {sd} ===", flush=True)
    m=load_model(sd); res={"seed":sd,"abpc":{},"records":{},"curves":{}}
    for ci,cls in enumerate(CLASSES):
        pos=sel[Yte[sel,ci]==1]
        if len(pos)<8: continue
        for meth in A_METHODS:
            A=attr_batch(meth,m,Xte[pos],ci,delin=DEL,idxs=pos)
            ab,Mo,Le=road(m,Xte[pos],A,ci)
            res["abpc"].setdefault(cls,{})[meth]=float(ab.mean())
            res["records"].setdefault(cls,{})[meth]=ab.tolist()
            res["curves"].setdefault(cls,{})[meth]={"morf_mean":Mo.mean(0).tolist(),"lerf_mean":Le.mean(0).tolist(),
                                                     "morf_sd":Mo.std(0).tolist(),"lerf_sd":Le.std(0).tolist(),"n":int(len(pos))}
        print(f"  {cls}: done ({len(pos)} pos) [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res,open(fp,"w"))
    print(f"  SAVED axisA_seed_{sd}.json  [{time.time()-t0:.0f}s]\n", flush=True)
    del m; gc.collect()

# ---- runtime per method (seed 42, 64 MI records, batch of 1 to mimic per-ECG use) ----
rfile=f"{OUT}/runtime.json"
if not os.path.exists(rfile):
    m=load_model(42); pos=sel[Yte[sel,MI]==1][:64]; rt={}
    for meth in ["Saliency","GradXInput","DeepLIFT","IG","DeepSHAP","GradCAM_L4","GradCAMpp_L4"]:
        attr_batch(meth,m,Xte[pos[:2]],MI,bs=1)                           # warm-up
        if dev=="cuda": torch.cuda.synchronize()
        t=time.time(); attr_batch(meth,m,Xte[pos],MI,bs=1)
        if dev=="cuda": torch.cuda.synchronize()
        rt[meth]=(time.time()-t)/len(pos)*1000
        print(f"  runtime {meth:13s} {rt[meth]:7.2f} ms / ECG", flush=True)
    json.dump({"ms_per_ecg":rt,"device":torch.cuda.get_device_name(0) if dev=="cuda" else "cpu","IG_steps":32,
               "DeepSHAP_baselines":"2 (zero, zero+N(0,0.1))"},open(rfile,"w"),indent=2)

# ---- two real ECGs with their attribution maps (qualitative figure) ----
efile=f"{OUT}/examples.npz"
if not os.path.exists(efile):
    m=load_model(42); ex={}
    for ci,cls in [(MI,"MI"),(0,"NORM")]:
        pos=sel[Yte[sel,ci]==1]; i=pos[0]
        ex[f"{cls}_signal"]=Xte[i]; ex[f"{cls}_record_index"]=np.array(i)
        for meth in ["Saliency","GradXInput","DeepLIFT","IG","DeepSHAP","GradCAM_L4"]:
            ex[f"{cls}_{meth}"]=attr_batch(meth,m,Xte[i:i+1],ci)[0]
    np.savez_compressed(efile,**ex)
print("\n================ PART 2 DONE ================", flush=True)
print("Files in /kaggle/working/part2/ : inj_*.json, axisA_seed_*.json, runtime.json, examples.npz, qrs_delineation.json")
print("Run complete; outputs are in the part2 folder.")

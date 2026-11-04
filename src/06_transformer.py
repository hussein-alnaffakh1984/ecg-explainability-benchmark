# =====================================================================================
# Transformer encoder: training (seeds 42, 1, 7), Axes A-C and shortcut injection (three settings).
# Grad-CAM is computed on the last convolution of the convolutional stem, the model's only
# convolutional layer. Captum's DeepLIFT falls back to gradients for attention/GELU operations,
# so DeepLIFT equals Grad x Input for this architecture.
# Output: part3/ (models, axes_seed_*.json, inj_*.json); resumable
# =====================================================================================
# Requirements: see requirements.txt (captum, neurokit2, ...)
import os, gc, glob, time, json, copy, random, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import neurokit2 as nk
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from captum.attr import (Saliency, InputXGradient, DeepLift, IntegratedGradients,
                         LayerGradCam, LayerAttribution, GradientShap)
warnings.filterwarnings("ignore")
if not hasattr(np,"trapz"): np.trapz=np.trapezoid
dev="cuda" if torch.cuda.is_available() else "cpu"
print("CUDA:", torch.cuda.is_available(), flush=True)

SEEDS=[42,1,7]; MAXEP=40; PATIENCE=8; INJ_EPOCHS=12; N_CAL=120; N_EVAL=200; C_SUB=60
def _find(name):
    h=glob.glob(f"/kaggle/input/**/{name}", recursive=True); return h[0] if h else None
SRC=os.path.dirname(_find("X.npy"))
WORK="/kaggle/working"; OUT=f"{WORK}/part3"; os.makedirs(f"{OUT}/models",exist_ok=True)
CLASSES=["NORM","MI","STTC","CD","HYP"]; MI=1
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; SR=100; T=1000

# ---------------- Transformer encoder (same architecture as the first-round analysis) ----------------
class ConvStem(nn.Module):
    def __init__(s, ci=12, d=128):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv1d(ci,64,15,4,7,bias=False), nn.BatchNorm1d(64), nn.GELU(),   # 1000->250
            nn.Conv1d(64,d,7,2,3,bias=False),   nn.BatchNorm1d(d),  nn.GELU(),   # 250->125
        )
    def forward(s,x): return s.net(x)
class PosEnc(nn.Module):
    def __init__(s, d, L=200):
        super().__init__()
        pe=torch.zeros(L,d); pos=torch.arange(L).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-np.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        s.register_buffer("pe",pe.unsqueeze(0))
    def forward(s,x): return x+s.pe[:,:x.size(1)]
class ECGTransformer(nn.Module):
    def __init__(s, ci=12, d=128, heads=4, layers=3, ncls=5):
        super().__init__()
        s.stem=ConvStem(ci,d); s.pos=PosEnc(d)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,0.2,batch_first=True,activation="gelu")
        s.tr=nn.TransformerEncoder(enc,layers)
        s.head=nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d,ncls))
    def forward(s,x):
        z=s.stem(x).transpose(1,2); z=s.pos(z); z=s.tr(z)
        return s.head(z.mean(1))

# ---------------- data ----------------
X=np.load(f"{SRC}/X.npy"); Y=np.load(f"{SRC}/Y.npy"); idx=pd.read_csv(f"{SRC}/split_index.csv")
tr,va,te=(idx.split=="train").values,(idx.split=="val").values,(idx.split=="test").values
Xtr=np.ascontiguousarray(X[tr].transpose(0,2,1)); Ytr=Y[tr]
Xva=np.ascontiguousarray(X[va].transpose(0,2,1)); Yva=Y[va]
Xte=np.ascontiguousarray(X[te].transpose(0,2,1)); Yte=Y[te]; pid=idx[te].patient_id.values
del X; gc.collect()

@torch.no_grad()
def predict(m,Xa,bs=256):
    m.eval(); P=[]
    for i in range(0,len(Xa),bs): P.append(torch.sigmoid(m(torch.from_numpy(Xa[i:i+bs]).to(dev))).cpu().numpy())
    return np.concatenate(P)

# ---------------- train (AdamW 1e-3, wd 1e-4, plateau sched, early stop on val AUROC) ----------------
def train_transformer(seed, Xa, Ya, early_stop=True, epochs=MAXEP):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m=ECGTransformer().to(dev); pos=Ya.sum(0)
    pw=torch.tensor((len(Ya)-pos)/np.clip(pos,1,None),dtype=torch.float32,device=dev)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.3,patience=3)
    Xt=torch.from_numpy(Xa); Yt=torch.from_numpy(Ya); best=0; best_state=None; bad=0
    for ep in range(1,epochs+1):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),128):
            b=perm[i:i+128]; opt.zero_grad(); loss=crit(m(Xt[b].to(dev)),Yt[b].to(dev)); loss.backward(); opt.step()
        if early_stop:
            va_auc=roc_auc_score(Yva,predict(m,Xva),average="macro"); sched.step(va_auc)
            if va_auc>best: best=va_auc; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=PATIENCE: break
    if early_stop: m.load_state_dict(best_state)
    m.eval(); return m, best, ep

metrics={}
for sd in SEEDS:
    fp=f"{OUT}/models/transformer_seed{sd}.pt"; mp=f"{OUT}/models/metrics_seed{sd}.json"
    if os.path.exists(fp) and os.path.exists(mp): metrics[sd]=json.load(open(mp)); print(f"transformer seed {sd} done, skip"); continue
    t0=time.time(); print(f"=== TRAIN Transformer seed {sd} ===", flush=True)
    m,best,ep=train_transformer(sd,Xtr,Ytr)
    P=predict(m,Xte); auc=roc_auc_score(Yte,P,average="macro")
    per={c:round(float(roc_auc_score(Yte[:,i],P[:,i])),4) for i,c in enumerate(CLASSES)}
    torch.save(m.state_dict(),fp)
    metrics[sd]={"seed":sd,"val_AUROC":round(float(best),4),"test_macro_AUROC":round(float(auc),4),"per_class_AUROC":per,
                 "epochs_run":ep,"params_M":round(sum(p.numel() for p in m.parameters())/1e6,3)}
    json.dump(metrics[sd],open(mp,"w"),indent=2)
    print(f"  seed {sd}: test macro-AUROC={auc:.4f}  ({ep} epochs, {time.time()-t0:.0f}s)  SAVED", flush=True)
    del m; gc.collect()
a=[metrics[s]["test_macro_AUROC"] for s in SEEDS]
print(f"Transformer test macro-AUROC: mean={np.mean(a):.4f} sd={np.std(a):.4f} range=[{min(a):.4f},{max(a):.4f}]\n", flush=True)
def load_tr(sd):
    m=ECGTransformer().to(dev); m.load_state_dict(torch.load(f"{OUT}/models/transformer_seed{sd}.pt",map_location=dev)); m.eval(); return m

# ---------------- attribution (batched; X: (B,12,T)) ----------------
METHODS=["Saliency","GradXInput","DeepLIFT","IG","DeepSHAP","GradCAM_stem","AbsX","Random"]
def attr_batch(name, model, Xa, tgt, bs=16):
    out=np.zeros((len(Xa),12,T),np.float32)
    for i in range(0,len(Xa),bs):
        xb=torch.tensor(Xa[i:i+bs],dtype=torch.float32,device=dev).requires_grad_(True)
        if name=="Saliency":     a=Saliency(model).attribute(xb,target=tgt)
        elif name=="GradXInput": a=InputXGradient(model).attribute(xb,target=tgt)
        elif name=="DeepLIFT":   a=DeepLift(model).attribute(xb,target=tgt)
        elif name=="IG":         a=IntegratedGradients(model).attribute(xb,target=tgt,baselines=xb*0,n_steps=32,internal_batch_size=64)
        elif name=="DeepSHAP":   a=GradientShap(model).attribute(xb,target=tgt,baselines=torch.cat([xb*0,xb*0+torch.randn_like(xb)*0.1]))
        elif name=="GradCAM_stem":
            a=LayerGradCam(model,model.stem.net[3]).attribute(xb,target=tgt)
            a=LayerAttribution.interpolate(a,(T,)).expand(-1,12,-1)
        elif name=="AbsX":       a=xb.abs()
        elif name=="Random":     a=torch.randn_like(xb)
        out[i:i+len(xb)]=a.detach().float().cpu().numpy(); del xb,a
    if dev=="cuda": torch.cuda.empty_cache()
    return out

# ---------------- verified pipeline pieces (vectorised gap=15 imputer; ROAD/ABPC; axis B; delineation) ----------------
def neigh_impute(sig, mask, k=8, gap=15):
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
def road_abpc(model, Xb, A, c):
    per=[]
    for n in range(len(Xb)):
        sig=Xb[n]; rank=np.argsort(np.abs(A[n]).sum(0))
        mo=np.zeros(len(STEPS)); le=np.zeros(len(STEPS))
        for si,f in enumerate(STEPS):
            nr=int(f*T)
            if nr==0: mo[si]=le[si]=_prob(model,sig,c); continue
            km=np.zeros(T,bool); km[rank[::-1][:nr]]=True; kl=np.zeros(T,bool); kl[rank[:nr]]=True
            mo[si]=_prob(model,neigh_impute(sig,km),c); le[si]=_prob(model,neigh_impute(sig,kl),c)
        per.append(float(np.trapz(le-mo,STEPS)))
    return np.array(per)
CLASS_REGION={"MI":{"segments":["QRS","ST"],"leads":"all"},"STTC":{"segments":["ST","T"],"leads":"all"},
              "CD":{"segments":["QRS"],"leads":"all"},"HYP":{"segments":["QRS","ST","T"],"leads":["V1","V2","V3","V4","V5","V6"]}}
def axis_b_fraction(A, cls, delin):          # A: (12,T)
    if not delin["ok"]: return np.nan
    spec=CLASS_REGION[cls]; li=range(12) if spec["leads"]=="all" else [LEADS.index(l) for l in spec["leads"]]
    msk=np.zeros(T,bool)
    for seg in spec["segments"]:
        for (a,b) in delin["windows"].get(seg,[]): msk[a:b]=True
    pos=np.clip(A,0,None); return float(sum(pos[l,msk].sum() for l in li)/(sum(pos[l].sum() for l in li)+1e-12))
def norm_dispersion(A):
    a=np.abs(A).T; a=a/(a.sum(0,keepdims=True)+1e-12)
    return float((-(a*np.log(a+1e-12)).sum(0)).mean()/np.log(a.shape[0]))
def delineate(x, lead=1):                   # x: (12,T)
    try:
        sig=nk.ecg_clean(x[lead],sampling_rate=SR); _,info=nk.ecg_peaks(sig,sampling_rate=SR); rp=info["ECG_R_Peaks"]
        if rp is None or len(rp)<2: return {"ok":False,"windows":{}}
        _,w=nk.ecg_delineate(sig,rp,sampling_rate=SR,method="dwt"); W={"QRS":[],"ST":[],"T":[]}
        ron=w.get("ECG_R_Onsets",[]); roff=w.get("ECG_R_Offsets",[]); ton=w.get("ECG_T_Onsets",[]); toff=w.get("ECG_T_Offsets",[])
        for key,(A_,B_) in {"QRS":(ron,roff),"ST":(roff,ton),"T":(ton,toff)}.items():
            for a,b in zip(A_,B_):
                if not(np.isnan(a) or np.isnan(b)) and 0<=int(a)<int(b)<=T: W[key].append((int(a),int(b)))
        return {"ok":len(W["QRS"])>0,"windows":W}
    except Exception: return {"ok":False,"windows":{}}
def randomize_top(model):
    # randomise the top half: encoder layers 2-3 and the classification head (all their parameters re-initialised)
    rm=copy.deepcopy(model); torch.manual_seed(1234)
    for mod in list(rm.tr.layers[1:])+[rm.head]:
        for sub in mod.modules():
            if hasattr(sub,"reset_parameters"): sub.reset_parameters()
            elif hasattr(sub,"_reset_parameters"): sub._reset_parameters()
    rm.eval(); return rm

r=np.random.default_rng(42); uniq=np.unique(pid); r.shuffle(uniq); keep=set(uniq[:300])
sel=np.where([p in keep for p in pid])[0]
dfile=f"{OUT}/delineation.json"
if os.path.exists(dfile): DEL={int(k):v for k,v in json.load(open(dfile)).items()}
else:
    print("delineating subsample (once)...", flush=True)
    DEL={int(i):delineate(Xte[i]) for i in sel}; json.dump({str(k):v for k,v in DEL.items()},open(dfile,"w"))
print(f"delineation ok-rate={np.mean([DEL[int(i)]['ok'] for i in sel]):.2f}\n", flush=True)

# =====================================================================================
# AXES A / B / C on the Transformer
# =====================================================================================
for sd in SEEDS:
    fp=f"{OUT}/axes_seed_{sd}.json"
    if os.path.exists(fp): print(f"[axes] seed {sd} done, skip"); continue
    t0=time.time(); print(f"=== AXES Transformer seed {sd} ===", flush=True)
    m=load_tr(sd); rm=randomize_top(m)
    res={"seed":sd,"A":{},"B":{},"C":{},"A_records":{},"failures":{}}
    for ci,cls in enumerate(CLASSES):
        pos=sel[Yte[sel,ci]==1]
        if len(pos)<8: continue
        subi=np.random.default_rng(1000+sd).choice(len(pos),min(C_SUB,len(pos)),replace=False)
        for meth in METHODS:
            try:
                A=attr_batch(meth,m,Xte[pos],ci)
            except Exception as e:
                res["failures"][f"{cls}/{meth}"]=repr(e)[:200]; print(f"  {cls}/{meth} FAILED: {repr(e)[:120]}", flush=True); continue
            ab=road_abpc(m,Xte[pos],A,ci)
            res["A"].setdefault(cls,{})[meth]=float(ab.mean()); res["A_records"].setdefault(cls,{})[meth]=ab.tolist()
            if cls=="NORM": b=np.mean([norm_dispersion(A[j]) for j in range(len(pos))])
            else:
                v=[axis_b_fraction(A[j],cls,DEL[int(pos[j])]) for j in range(len(pos))]; v=[x for x in v if not np.isnan(x)]
                b=np.mean(v) if v else np.nan
            res["B"].setdefault(cls,{})[meth]=float(b)
            if meth in ("AbsX","Random"):            # model-independent maps: randomisation is not defined
                res["C"].setdefault(cls,{})[meth]=None; continue
            Ar=attr_batch(meth,rm,Xte[pos[subi]],ci); cv=[]
            for j,k in enumerate(subi):
                a0=A[k].ravel(); a1=Ar[j].ravel()
                if a0.std()>1e-9 and a1.std()>1e-9: cv.append(spearmanr(a0,a1).correlation)
            res["C"].setdefault(cls,{})[meth]=float(np.nanmean(cv)) if cv else None
        print(f"  {cls}: done ({len(pos)} pos) [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res,open(fp,"w"),indent=2)
    print(f"  SAVED axes_seed_{sd}.json [{time.time()-t0:.0f}s]\n", flush=True)
    del m,rm; gc.collect()

# =====================================================================================
# SHORTCUT INJECTION on the Transformer (held-out calibration + reliance)
# =====================================================================================
aVR=LEADS.index("aVR")
CONFIGS={"base_aVR_A8_W40":dict(kind="rect",t0=40,t1=80,amp=8.0),
         "lowamp_A2":dict(kind="rect",t0=40,t1=80,amp=2.0),
         "morph_gauss_A2":dict(kind="gauss",t0=51,t1=69,amp=2.0)}
def wave(cfg,shift=0):
    t0,t1=cfg["t0"]+shift,cfg["t1"]+shift; w=np.zeros(T,np.float32)
    if cfg["kind"]=="rect": w[t0:t1]=cfg["amp"]
    else:
        c=(t0+t1)/2; s=(t1-t0)/6; tt=np.arange(T); w=(cfg["amp"]*np.exp(-0.5*((tt-c)/s)**2)).astype(np.float32)
    return w
def add_spike(Xa,rows,cfg,shift=0): Xa[rows,aVR,:]+=wave(cfg,shift); return Xa
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); rr=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-rr)/d,(c+rr)/d)
mi_pos=np.where(Yte[:,MI]==1)[0]; np.random.default_rng(0).shuffle(mi_pos)
cal_idx=mi_pos[:N_CAL]; ev_idx=mi_pos[N_CAL:N_CAL+N_EVAL]
for name,cfg in CONFIGS.items():
    fp=f"{OUT}/inj_{name}.json"
    if os.path.exists(fp): print(f"[inject] {name} done, skip"); continue
    t0=time.time(); print(f"=== Transformer INJECTION {name} ===", flush=True)
    Xi=add_spike(Xtr.copy(),np.where(Ytr[:,MI]==1)[0],cfg)
    m,_,_=train_transformer(0,Xi,Ytr,early_stop=False,epochs=INJ_EPOCHS); del Xi; gc.collect()
    mi_te=np.where(Yte[:,MI]==1)[0]
    Xa=add_spike(Xte.copy(),mi_te,cfg)
    rel={"auroc_spike_on_MI":roc_auc_score(Yte[:,MI],predict(m,Xa)[:,MI]),
         "auroc_clean":roc_auc_score(Yte[:,MI],predict(m,Xte)[:,MI]),
         "auroc_shifted":roc_auc_score(Yte[:,MI],predict(m,add_spike(Xte.copy(),mi_te,cfg,shift=300))[:,MI]),
         "auroc_spike_on_all":roc_auc_score(Yte[:,MI],predict(m,add_spike(Xte.copy(),np.arange(len(Xte)),cfg))[:,MI])}
    print("  reliance:",{k:round(float(v),3) for k,v in rel.items()}, flush=True)
    t0w,t1w=cfg["t0"],cfg["t1"]; chance=(t1w-t0w)/(T*12)
    def frac(A): a=np.abs(A); return a[:,aVR,t0w:t1w].reshape(len(a),-1).sum(1)/(a.reshape(len(a),-1).sum(1)+1e-12)
    ig=frac(attr_batch("IG",m,Xa[cal_idx],MI)); n1=frac(np.abs(Xa[cal_idx])); n2=frac(np.random.default_rng(1).standard_normal(Xa[cal_idx].shape))
    yt=np.r_[np.ones(len(ig)),np.zeros(len(n1)+len(n2))]; ys=np.r_[ig,n1,n2]
    cal_auc=roc_auc_score(yt,ys); fpr,tpr,thr=roc_curve(yt,ys); tau=float(thr[np.argmax(tpr-fpr)])
    failed=(not np.isfinite(tau)) or cal_auc<=0.5
    if not np.isfinite(tau): tau=float(ys.max()+1e-9)
    res={"config":cfg,"reliance":{k:float(v) for k,v in rel.items()},"chance_lead":chance,
         "calibration":{"auroc":float(cal_auc),"tau_heldout":tau,"tau_fixed_10x_chance":10*chance,"calibration_failed":bool(failed)},
         "n_cal":int(len(cal_idx)),"n_eval":int(len(ev_idx)),"methods":{}}
    for meth in METHODS:
        try: f=frac(attr_batch(meth,m,Xa[ev_idx],MI))
        except Exception as e: res["methods"][meth]={"failed":repr(e)[:200]}; continue
        n=len(f); k1=int((f>tau).sum()); k2=int((f>10*chance).sum())
        res["methods"][meth]={"frac_mean":float(f.mean()),"x_chance":float(f.mean()/chance),
                              "detect_heldout":k1/n,"detect_heldout_ci":wilson(k1,n),"detect_10x":k2/n,"detect_10x_ci":wilson(k2,n)}
        print(f"  {meth:13s} frac={f.mean():.3f} ({f.mean()/chance:6.1f}x)  detect held-out={k1/n:5.1%}  10x-chance={k2/n:5.1%}", flush=True)
    json.dump(res,open(fp,"w"),indent=2)
    print(f"  SAVED inj_{name}.json  tau={tau:.4f} [{time.time()-t0:.0f}s]\n", flush=True)
    del m,Xa; gc.collect()

print("================ PART 3 DONE ================", flush=True)
print("Run complete; outputs are in the part3 folder.")

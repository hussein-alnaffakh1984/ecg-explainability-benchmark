# =====================================================================================
# Reconstruction-leakage ablation for the ROAD imputer (Appendix Table A2).
# Leakage = Pearson correlation between imputed and true values at removed positions,
# averaged over 200 random test records with random removal masks. Lower = less leakage.
# Part 1: linear interpolation and the k = 8 neighbourhood fill without a guard band.
# Part 2: neighbourhood fill with k = 4/8 and guard bands g = 0/5/15, and a global-mean fill.
# Input : X.npy, split_index.csv (from 01_rebuild_and_train.py)
# =====================================================================================
import os, json, numpy as np, pandas as pd

WORK=os.environ.get("WORK","/kaggle/working")
X=np.load(f"{WORK}/X.npy"); idx=pd.read_csv(f"{WORK}/split_index.csv")
Xte=X[(idx.split=="test").values].transpose(0,2,1)      # (record, lead, time)

# ---------------- Part 1 ----------------
def linear_impute(sig, mask):
    """Baseline imputer: linear interpolation across masked positions, per lead.
       sig: (L, T) ; mask: (T,) bool, True = REMOVE/impute."""
    out=sig.copy(); T=sig.shape[1]; t=np.arange(T)
    for l in range(sig.shape[0]):
        m=mask
        if m.all():            # nothing to interpolate from
            out[l,:]=0.0; continue
        out[l,m]=np.interp(t[m], t[~m], sig[l,~m])
    return out

def neighbourhood_impute(sig, mask, k=8):
    """Proposal imputer: fill each removed position by the mean of its k nearest
       NON-removed neighbours in time, per lead (in-distribution, local)."""
    out=sig.copy(); T=sig.shape[1]; idx_all=np.arange(T)
    for l in range(sig.shape[0]):
        keep=~mask
        if keep.sum()==0: out[l,:]=0.0; continue
        kept_idx=idx_all[keep]; kept_val=sig[l,keep]
        rem_idx=idx_all[mask]
        # nearest k kept neighbours by time distance
        for ri in rem_idx:
            j=np.searchsorted(kept_idx, ri)
            lo=max(0,j-k//2); hi=min(len(kept_idx),lo+k)
            out[l,ri]=kept_val[lo:hi].mean()
    return out

# -------- reconstruction-leakage test --------
# Randomly remove a fraction p of positions, impute, and measure how well the
# imputed value matches the TRUE removed value. High recovery = high leakage.
def leakage_report(Xsub, imputer, fracs=(0.1,0.2,0.3,0.5), seed=42):
    rng=np.random.default_rng(seed); rep={}
    L,T=Xsub.shape[1],Xsub.shape[2]
    for p in fracs:
        rec_corr=[]; rec_rmse=[]; sig_var=[]
        for n in range(Xsub.shape[0]):
            sig=Xsub[n]
            mask=rng.random(T)<p
            if mask.sum()<2: continue
            imp=imputer(sig, mask)
            true_removed=sig[:,mask].ravel()
            imp_removed =imp[:,mask].ravel()
            # correlation between imputed and true removed values = leakage proxy
            if true_removed.std()>1e-6 and imp_removed.std()>1e-6:
                rec_corr.append(np.corrcoef(true_removed, imp_removed)[0,1])
            rec_rmse.append(np.sqrt(((true_removed-imp_removed)**2).mean()))
            sig_var.append(true_removed.std())
        rep[p]={"recon_corr_mean":float(np.mean(rec_corr)),
                "recon_rmse_mean":float(np.mean(rec_rmse)),
                "signal_std_mean":float(np.mean(sig_var))}
    return rep


rng=np.random.default_rng(0)
sub=Xte[rng.choice(Xte.shape[0], size=200, replace=False)]
part1={n:leakage_report(sub,imp) for n,imp in [("linear",linear_impute),("neighbourhood_k8",neighbourhood_impute)]}
for n,rep in part1.items():
    print(n, {p:round(v["recon_corr_mean"],3) for p,v in rep.items()})

# ---------------- Part 2 ----------------
rng=np.random.default_rng(0); sub=Xte[rng.choice(Xte.shape[0],200,replace=False)]

def neigh_impute(sig, mask, k=8, gap=0):
    """Fill removed positions by mean of k kept neighbours, EXCLUDING the nearest
       'gap' kept samples on each side (larger gap -> less local reconstruction)."""
    out=sig.copy(); T=sig.shape[1]; idx_all=np.arange(T)
    for l in range(sig.shape[0]):
        keep=~mask
        if keep.sum()==0: out[l,:]=0.0; continue
        kept_idx=idx_all[keep]; kept_val=sig[l,keep]
        for ri in idx_all[mask]:
            j=np.searchsorted(kept_idx, ri)
            lo=max(0,j-gap-k//2); hi_excl_lo=max(0,j-gap)
            hi=min(len(kept_idx), j+gap+k//2); lo_excl_hi=min(len(kept_idx), j+gap)
            pool=np.concatenate([kept_val[lo:hi_excl_lo], kept_val[lo_excl_hi:hi]])
            out[l,ri]= pool.mean() if len(pool)>0 else kept_val.mean()
    return out

def global_mean_impute(sig, mask):
    """Fill with per-lead global mean of kept samples (most off-local, least leakage)."""
    out=sig.copy()
    for l in range(sig.shape[0]):
        keep=~mask
        out[l,mask]= sig[l,keep].mean() if keep.sum()>0 else 0.0
    return out

def leak(imputer, fracs=(0.1,0.2,0.3)):
    rep={}
    for p in fracs:
        cs=[]
        for n in range(sub.shape[0]):
            sig=sub[n]; mask=rng.random(sig.shape[1])<p
            if mask.sum()<2: continue
            imp=imputer(sig,mask)
            tr=sig[:,mask].ravel(); im=imp[:,mask].ravel()
            if tr.std()>1e-6 and im.std()>1e-6: cs.append(np.corrcoef(tr,im)[0,1])
        rep[p]=round(float(np.mean(cs)),3)
    return rep

print("recon_corr by config (lower = less leakage):")
for cfg,fn in [
    ("neigh k=4 gap=0", lambda s,m: neigh_impute(s,m,4,0)),
    ("neigh k=8 gap=5", lambda s,m: neigh_impute(s,m,8,5)),
    ("neigh k=8 gap=15",lambda s,m: neigh_impute(s,m,8,15)),
    ("global_mean",     global_mean_impute),
]:
    print(f"  {cfg:18s} -> {leak(fn)}")

# Collects every number reported in the manuscript into results/master_results.json.
import json, glob, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy import stats
warnings.filterwarnings("ignore")
import os
B=os.environ.get("RESULTS_DIR","results/raw")  # folder holding resnet_axes/, injection_controls/, transformer/
R1=f"{B}/resnet_axes"; R2=f"{B}/injection_controls"; R3=f"{B}/transformer"
CL=["NORM","MI","STTC","CD","HYP"]; FAM=["DeepSHAP","IG","DeepLIFT","GradXInput"]; M6=FAM+["Saliency","GradCAM"]
S=[json.load(open(f)) for f in sorted(glob.glob(f"{R1}/seed_*.json"))]; assert len(S)==10
OUT={"source":{"resnet_axes":"part1 10 seeds","injection_baselines":"part2","transformer":"part3 3 seeds"}}
def ci(v):
    v=np.asarray(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return [float(m),float(m-1.96*se),float(m+1.96*se)]
def seedA(c,m): return np.array([s["A"][c][m]-s["A"][c]["Random"] for s in S])
# ---- Axis A (ResNet, 10 seeds, seed-level CI) ----
OUT["resnet_axisA"]={c:{m:ci(seedA(c,m)) for m in M6} for c in CL}
# ---- MI pooled per record ----
mi={}
for m in M6:
    r=np.concatenate([np.array(s["A_records"]["MI"][m])-np.array(s["A_records"]["MI"]["Random"]) for s in S])
    se=r.std(ddof=1)/np.sqrt(len(r)); mi[m]=[float(r.mean()),float(r.mean()-1.96*se),float(r.mean()+1.96*se),int(len(r))]
OUT["resnet_MI_pooled"]=mi
# ---- TOST (margin 0.02) ----
def tost(x,mg=0.02):
    n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n)
    p=max(stats.t.sf((m+mg)/se,n-1),stats.t.cdf((m-mg)/se,n-1)); return {"mean":float(m),"p":float(p),"equivalent":bool(p<0.05)}
OUT["tost_gradcam_vs_random"]={c:tost(seedA(c,"GradCAM")) for c in CL}
fam={}
for m in ["IG","DeepLIFT","GradXInput"]:
    d=np.array([s["A"][c][m]-s["A"][c]["DeepSHAP"] for s in S for c in ["NORM","HYP","STTC"]]); fam[m]=tost(d)
OUT["tost_family_vs_deepshap_powered"]=fam
# ---- Holm on method vs random (paired t, 30 tests) ----
lab=[];p=[]
for c in CL:
    for m in M6: lab.append(f"{c}/{m}"); p.append(stats.ttest_1samp(seedA(c,m),0).pvalue)
o=np.argsort(p); h=np.empty(len(p)); run=0
for k,i in enumerate(o): run=max(run,min(1,(len(p)-k)*p[i])); h[i]=run
OUT["holm"]={lab[i]:{"p_raw":float(p[i]),"p_holm":float(h[i])} for i in range(len(p))}
OUT["holm_n_significant"]=int((h<0.05).sum())
# ---- hierarchical bootstrap ----
rng=np.random.default_rng(0); hb={}
for c in CL:
    hb[c]={}
    for m in M6:
        ps=[np.array(s["A_records"][c][m])-np.array(s["A_records"][c]["Random"]) for s in S]; bs=[]
        for _ in range(2000):
            bs.append(np.mean([a[rng.integers(0,len(a),len(a))].mean() for a in [ps[i] for i in rng.integers(0,10,10)]]))
        hb[c][m]=[float(np.mean(bs)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
OUT["resnet_axisA_hier_bootstrap"]=hb
# ---- crossed random effects (record + seed): contrasts vs DeepSHAP ----
cre={}
for c in CL:
    rows=[]
    for s in S:
        rd=np.array(s["A_records"][c]["Random"])
        for m in ["DeepSHAP","GradCAM","Saliency"]:
            for j,v in enumerate(np.array(s["A_records"][c][m])-rd): rows.append(dict(y=v,method=m,rec=f"r{j}",seed=str(s["seed"])))
    df=pd.DataFrame(rows); df["g"]=1
    f=smf.mixedlm("y ~ C(method, Treatment('DeepSHAP'))",df,groups="g",vc_formula={"rec":"0+C(rec)","seed":"0+C(seed)"},re_formula="0").fit(method="lbfgs",disp=False)
    cre[c]={}
    for k in f.params.index:
        if "T." in k:
            lo,hi=f.conf_int().loc[k]; cre[c][k.split("T.")[1][:-1]+"-DeepSHAP"]=[float(f.params[k]),float(lo),float(hi)]
OUT["resnet_crossedRE_vs_DeepSHAP"]=cre
# ---- Axis B (10 seeds) ----
OUT["resnet_axisB"]={c:{m:ci([s["B"][c][m] for s in S]) for m in M6+["Random"]} for c in CL}
# ---- Axis C ----
OUT["resnet_axisC_mean_abs"]={m:{c:float(np.mean([abs(s["C"][c][m]) for s in S])) for c in CL} for m in M6+["Random"]}
OUT["resnet_axisC_signed_range_MI"]={m:[float(min(s["C"]["MI"][m] for s in S)),float(max(s["C"]["MI"][m] for s in S)),float(np.std([s["C"]["MI"][m] for s in S],ddof=1))] for m in M6}
# ---- orthogonality ----
a=[];b=[];cc=[]
for c in CL:
    for m in M6:
        a.append(seedA(c,m).mean()); b.append(np.mean([s["B"][c][m] for s in S])); cc.append(np.mean([abs(s["C"][c][m]) for s in S]))
OUT["axis_correlation_spearman"]={"A_B":list(map(float,stats.spearmanr(a,b))),"A_C":list(map(float,stats.spearmanr(a,cc))),"B_C":list(map(float,stats.spearmanr(b,cc))),"n_cells":len(a)}
# ---- part2: injection, extra baselines, Grad-CAM variants, runtime ----
inj={}
for f in sorted(glob.glob(f"{R2}/inj_*.json")):
    d=json.load(open(f)); n=f.split("inj_")[1][:-5]
    inj[n]={"reliance":d["reliance"],"calibration":d["calibration"],"chance":d["chance"],"n_eval":d["n_eval"],
            "methods":{m:{k:d["methods"][m][k] for k in ["lead_frac_mean","lead_frac_x_chance","detect_heldout_tau","detect_heldout_ci","detect_fixed10x","detect_fixed10x_ci"]} for m in d["methods"]}}
OUT["resnet_injection"]=inj
X=[json.load(open(f"{R2}/axisA_seed_{s}.json")) for s in [42,1,7]]
OUT["resnet_axisA_extra_3seeds"]={c:{m:[float(np.mean([x["abpc"][c][m]-x["abpc"][c]["Random"] for x in X])),
                                         float(min(x["abpc"][c][m]-x["abpc"][c]["Random"] for x in X)),
                                         float(max(x["abpc"][c][m]-x["abpc"][c]["Random"] for x in X))]
                                    for m in X[0]["abpc"][c] if m!="Random"} for c in CL}
OUT["runtime_ms_per_ecg_T4"]=json.load(open(f"{R2}/runtime.json"))["ms_per_ecg"]
# ---- part3: Transformer ----
T=[json.load(open(f"{R3}/axes_seed_{s}.json")) for s in [42,1,7]]
OUT["transformer_classifier"]={s:json.load(open(f"{R3}/models/metrics_seed{s}.json")) for s in [42,1,7]}
TM=["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM_stem","AbsX"]
OUT["transformer_axisA_3seeds"]={c:{m:[float(np.mean([t["A"][c][m]-t["A"][c]["Random"] for t in T])),
                                       float(min(t["A"][c][m]-t["A"][c]["Random"] for t in T)),
                                       float(max(t["A"][c][m]-t["A"][c]["Random"] for t in T))] for m in TM} for c in CL}
OUT["transformer_axisB_3seeds"]={c:{m:float(np.mean([t["B"][c][m] for t in T])) for m in TM+["Random"]} for c in CL}
OUT["transformer_axisC_mean_abs"]={m:{c:float(np.mean([abs(t["C"][c][m]) for t in T])) for c in CL} for m in TM[:6]}
tinj={}
for f in sorted(glob.glob(f"{R3}/inj_*.json")):
    d=json.load(open(f)); tinj[f.split("inj_")[1][:-5]]={"reliance":d["reliance"],"calibration":d["calibration"],"methods":d["methods"]}
OUT["transformer_injection"]=tinj
# ---- TOST sensitivity to the equivalence margin (Grad-CAM minus random, paired over seeds) ----
def tost_p(d,m):
    n=len(d); se=d.std(ddof=1)/np.sqrt(n)
    return float(max(1-stats.t.cdf((d.mean()+m)/se,n-1), stats.t.cdf((d.mean()-m)/se,n-1)))
OUT["tost_gradcam_vs_random_margin_sensitivity"]={"margins":[0.01,0.02,0.03],
    **{c:[round(tost_p(seedA(c,"GradCAM"),m),4) for m in (0.01,0.02,0.03)] for c in CL},
    "note":"TOST p, paired t over 10 seeds, GradCAM minus Random"}
json.dump(OUT,open("results/master_results.json","w"),indent=1)
print("written master_results.json; keys:",list(OUT.keys()))

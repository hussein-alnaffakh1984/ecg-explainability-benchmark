# Draws Figures 1-8 (600 dpi) from results/master_results.json and the raw run outputs.
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
B=os.environ.get("RESULTS_DIR","results/raw")  # folder holding resnet_axes/, injection_controls/, transformer/
OUT="figures"; import os; os.makedirs(OUT,exist_ok=True)
D=json.load(open("results/master_results.json"))
R1=sorted(glob.glob(f"{B}/resnet_axes/seed_*.json")); S=[json.load(open(f)) for f in R1]
X=[json.load(open(f"{B}/injection_controls/axisA_seed_{s}.json")) for s in [42,1,7]]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":9.5,"axes.titleweight":"bold",
    "axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,"axes.spines.top":False,
    "axes.spines.right":False,"axes.linewidth":0.8,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e",
    "axes.grid":True,"grid.color":"#e4e3df","grid.linewidth":0.6,"axes.axisbelow":True,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.facecolor":"white","figure.facecolor":"white"})
INK="#0b0b0b"; INK2="#52514e"; GRAY="#8a8985"
COL={"DeepSHAP":"#2a78d6","IG":"#eb6834","DeepLIFT":"#1baf7a","GradXInput":"#eda100","Saliency":"#e87ba4","GradCAM":"#008300","AbsX":"#6250d6"}
NICE={"DeepSHAP":"GradientSHAP","IG":"Integrated Gradients","DeepLIFT":"DeepLIFT","GradXInput":"Grad×Input","Saliency":"Saliency",
      "GradCAM":"Grad-CAM","AbsX":"|x| amplitude (model-agnostic)","GradCAM_stem":"Grad-CAM"}
MK={"DeepSHAP":"o","IG":"s","DeepLIFT":"^","GradXInput":"v","Saliency":"D","GradCAM":"X","AbsX":"P"}
CL=["NORM","MI","STTC","CD","HYP"]; M6=["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM"]
def save(fig,name): fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved",name)

# ---------------- Figure 1: pipeline ----------------
fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.set_xlim(0,100); ax.set_ylim(0,52); ax.axis("off")
def box(x,y,w,h,title,body,fc="#f3f2ee",ec="#52514e"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h-3.0,title,ha="center",va="top",fontsize=8.2,fontweight="bold",color=INK)
    ax.text(x+w/2,y+h-7.6,body,ha="center",va="top",fontsize=6.9,color=INK2,linespacing=1.25)
def arrow(x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#52514e",lw=0.9,mutation_scale=9))
box(1,33,21,17,"PTB-XL","21,430 records\n12-lead, 100 Hz\npatient-disjoint folds")
box(26,33,22,17,"Classifiers","ResNet1D × 10 seeds\nTransformer × 3 seeds")
box(52,33,22,17,"Attribution","6 methods\n+ |x|, smooth-random,\nQRS-mask, random")
box(78,33,21,17,"Statistics","seed-level CIs, TOST\nHolm, crossed RE\nhierarchical bootstrap")
arrow(22.4,41.5,25.6,41.5); arrow(48.4,41.5,51.6,41.5); arrow(74.4,41.5,77.6,41.5)
box(1,4,23,24,"Axis A","leakage-controlled\nROAD / ABPC\n(gap = 15 imputer)\nper class")
box(26.5,4,23,24,"Shortcut injection","8 artefact settings\nheld-out calibration\nreliance check\n(decisive test)",fc="#e3eefb",ec="#2a78d6")
box(52,4,22,24,"Axis B","lead-aware clinical\nregions (QRS/ST/T)\nreported as a flag")
box(76.5,4,22.5,24,"Axis C","weight randomisation\nmean over seeds |C|\ndiagnostic only")
for xc in [12.5,38,63,87.7]: arrow(63,32.6,xc,28.6)
ax.text(25.2,1.0,"Decision rule = Axis A (where powered)  +  shortcut injection",ha="center",fontsize=7.4,color="#2a78d6",fontweight="bold")
save(fig,"fig1_pipeline")

# ---------------- Figure 2: Axis A, ResNet (10 seeds) + |x| (3 seeds) and Transformer (3 seeds) ----------------
fig,axs=plt.subplots(2,1,figsize=(7.2,6.4),sharex=True)
meths=M6+["AbsX"]; off=np.linspace(-0.33,0.33,len(meths))
ax=axs[0]
for k,m in enumerate(meths):
    for i,c in enumerate(CL):
        if m=="AbsX": v,lo,hi=D["resnet_axisA_extra_3seeds"][c]["AbsX"]
        else: v,lo,hi=D["resnet_axisA"][c][m]
        ax.errorbar(i+off[k],v,yerr=[[v-lo],[hi-v]],fmt=MK[m],ms=5,mfc=COL[m] if m!="AbsX" else "white",mec=COL[m],mew=1.1,
                    ecolor=COL[m],elinewidth=1.0,capsize=0,label=NICE[m] if i==0 else None)
ax.axhline(0,color=GRAY,lw=0.9,ls="--"); ax.text(-0.45,0.006,"random",fontsize=7,color=INK2,ha="left",va="bottom")
ax.set_ylabel("ABPC minus random"); ax.set_title("a  ResNet1D: family methods (10 seeds, 95% CI) vs |x| amplitude (3 seeds, range)",loc="left")
ax.legend(ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.02)) if False else None
ax=axs[1]
tm=["DeepSHAP","IG","DeepLIFT","GradXInput","Saliency","GradCAM_stem","AbsX"]
for k,m in enumerate(tm):
    key="GradCAM" if m=="GradCAM_stem" else m
    for i,c in enumerate(CL):
        v,lo,hi=D["transformer_axisA_3seeds"][c][m]
        ax.errorbar(i+off[k],v,yerr=[[v-lo],[hi-v]],fmt=MK[key],ms=5,mfc=COL[key] if key!="AbsX" else "white",mec=COL[key],mew=1.1,
                    ecolor=COL[key],elinewidth=1.0,capsize=0,label=NICE[key] if i==0 else None)
ax.axhline(0,color=GRAY,lw=0.9,ls="--")
ax.set_ylabel("ABPC minus random"); ax.set_title("b  Transformer (3 seeds, range); DeepLIFT coincides with Grad×Input",loc="left")
ax.set_xticks(range(5)); ax.set_xticklabels(CL)
h,l=axs[1].get_legend_handles_labels()
fig.legend(h,l,ncol=4,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=(0,0.05,1,1)); save(fig,"fig2_axisA_10seeds")

# ---------------- Figure 3: real ROAD curves (ResNet, 3 seeds pooled, mean ± 95% CI over records) ----------------
steps=np.linspace(0,50,11)
show=["DeepSHAP","Saliency","GradCAM_L4","AbsX"]; lab={"GradCAM_L4":"Grad-CAM","DeepSHAP":"GradientSHAP","Saliency":"Saliency","AbsX":"|x| amplitude"}
ck={"GradCAM_L4":"GradCAM"}
fig,axs=plt.subplots(2,4,figsize=(7.2,3.9),sharex=True,sharey=True)
for r,c in enumerate(["NORM","HYP"]):
    for j,m in enumerate(show):
        ax=axs[r,j]; col=COL[ck.get(m,m)]
        mo=np.mean([x["curves"][c][m]["morf_mean"] for x in X],0); le=np.mean([x["curves"][c][m]["lerf_mean"] for x in X],0)
        n=X[0]["curves"][c][m]["n"]*3
        mos=np.mean([x["curves"][c][m]["morf_sd"] for x in X],0)/np.sqrt(n)*1.96
        les=np.mean([x["curves"][c][m]["lerf_sd"] for x in X],0)/np.sqrt(n)*1.96
        ax.fill_between(steps,mo,le,color=col,alpha=0.14,lw=0)
        ax.plot(steps,le,color=col,lw=2,label="LeRF"); ax.fill_between(steps,le-les,le+les,color=col,alpha=0.25,lw=0)
        ax.plot(steps,mo,color=INK2,lw=2,ls="--",label="MoRF"); ax.fill_between(steps,mo-mos,mo+mos,color=INK2,alpha=0.2,lw=0)
        ab=np.mean([x["abpc"][c][m]-x["abpc"][c]["Random"] for x in X])
        ax.set_title(f"{lab[m]}\n{c}  ABPC {ab:+.3f}",fontsize=8); ax.set_ylim(0,1)
        if j==0: ax.set_ylabel("predicted probability")
        if r==1: ax.set_xlabel("% removed")
axs[0,0].legend(frameon=False,loc="lower left",fontsize=7)
fig.tight_layout(); save(fig,"fig3_road_curves")

# ---------------- Figure 4: MI profile (faithfulness vs clinical validity), ResNet 10 seeds ----------------
fig,ax=plt.subplots(figsize=(4.6,3.6))
rb=D["resnet_axisB"]["MI"]["Random"][0]
for m in M6:
    x,xl,xh=D["resnet_axisA"]["MI"][m]; y,yl,yh=D["resnet_axisB"]["MI"][m]
    ax.errorbar(x,y-rb,xerr=[[x-xl],[xh-x]],yerr=[[y-yl],[yh-y]],fmt=MK[m],ms=6,mfc=COL[m],mec="white",mew=0.8,ecolor=COL[m],elinewidth=1)
dx={"DeepSHAP":(-0.0225,-0.013),"IG":(-0.0125,0.016),"DeepLIFT":(0.0012,-0.024),"GradXInput":(-0.0085,-0.026),"Saliency":(0.0015,0.01),"GradCAM":(0.0015,0.012)}
for m in M6:
    x=D["resnet_axisA"]["MI"][m][0]; y=D["resnet_axisB"]["MI"][m][0]-rb
    ax.text(x+dx[m][0],y+dx[m][1],NICE[m],fontsize=7.2,color=INK)
ax.axhline(0,color=GRAY,lw=0.9,ls="--"); ax.axvline(0,color=GRAY,lw=0.9,ls="--")
ax.set_xlabel("Faithfulness on MI (ABPC minus random; seed-level 95% CI)"); ax.set_ylabel("Clinical validity on MI\n(in-region share minus random)")
ax.set_title("Faithfulness × clinical validity on MI (ResNet1D, 10 seeds)",loc="left",fontsize=8.5)
fig.tight_layout(); save(fig,"fig4_profile")

# ---------------- Figure 5: Axis C per-seed signed similarity (ResNet, MI) ----------------
fig,ax=plt.subplots(figsize=(5.4,3.4))
for k,m in enumerate(M6):
    v=np.array([s["C"]["MI"][m] for s in S]); jit=np.random.default_rng(k).uniform(-0.12,0.12,len(v))
    ax.scatter(k+jit,v,s=22,color=COL[m],edgecolor="white",lw=0.6,zorder=3)
    ma=D["resnet_axisC_mean_abs"][m]["MI"]
    ax.plot([k-0.28,k+0.28],[ma,ma],color=INK,lw=1.6,zorder=4)
rv=np.array([s["C"]["MI"]["Random"] for s in S]); ax.scatter(6+np.random.default_rng(9).uniform(-0.12,0.12,10),rv,s=22,color=GRAY,edgecolor="white",lw=0.6)
ax.plot([5.72,6.28],[D["resnet_axisC_mean_abs"]["Random"]["MI"]]*2,color=INK,lw=1.6)
ax.axhline(0.30,color="#e34948",lw=1,ls=":"); ax.text(6.45,0.31,"τ_C = 0.30",fontsize=7,color=INK2,ha="right",va="bottom")
ax.axhline(0,color=GRAY,lw=0.7)
ax.set_xticks(range(7)); ax.set_xticklabels([NICE[m] if m!="IG" else "IG" for m in M6]+["Random"],rotation=20,ha="right")
ax.set_ylabel("post-randomisation similarity\n(signed Spearman, per seed)")
ax.set_title("Axis C on MI: dots = seeds; bar = mean over seeds of |C|",loc="left",fontsize=8.5)
fig.tight_layout(); save(fig,"fig5_axisC_seeds")

# ---------------- Figure 6: injection detection heatmaps ----------------
cfgs=["base_aVR_A8_W40","lowamp_A2","vlowamp_A0.5","narrow_W10","late_t600","wide_lowfreq_W200","morph_gauss_A2","physio_LAelectrode"]
cfg_lab=["base\nA=8, W=40","low amp\nA=2","very low\nA=0.5","narrow\nW=10","late\nt=600","wide\nW=200","QRS-like\nGaussian","physiological\nLA electrode"]
rows=["DeepLIFT","GradXInput","IG","DeepSHAP","Saliency","GradCAM_L4","GradCAM_L3","GradCAM_L2","GradCAMpp_L4","AbsX","SmoothRandom","Random"]
rlab=["DeepLIFT","Grad×Input","Integrated Gradients","GradientSHAP","Saliency","Grad-CAM (layer 4)","Grad-CAM (layer 3)","Grad-CAM (layer 2)","Grad-CAM++ (layer 4)","|x| amplitude","smooth random","random"]
Mx=np.array([[D["resnet_injection"][c]["methods"][r]["detect_heldout_tau"]*100 for c in cfgs] for r in rows])
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list("blues",["#f5f8fd","#a9c8ef","#2a78d6","#0d3f7d"])
fig,axs=plt.subplots(1,2,figsize=(7.2,4.6),gridspec_kw={"width_ratios":[8,3.0],"wspace":0.08})
ax=axs[0]; im=ax.imshow(Mx,cmap=cmap,vmin=0,vmax=100,aspect="auto")
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax.text(j,i,str(int(Mx[i,j]+0.5+1e-9)),ha="center",va="center",fontsize=6.6,color="white" if Mx[i,j]>55 else INK)
ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([c.replace("\n"," ") for c in cfg_lab],fontsize=6.4,rotation=40,ha="right"); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab,fontsize=7)
ax.grid(False); ax.set_title("a  ResNet1D: detection rate (%), held-out τ, 200 records",loc="left",fontsize=8.5)
trows=["DeepLIFT","GradXInput","IG","DeepSHAP","Saliency","GradCAM_stem","AbsX","Random"]
tcfg=["base_aVR_A8_W40","lowamp_A2","morph_gauss_A2"]
Mt=np.array([[D["transformer_injection"][c]["methods"][r]["detect_heldout"]*100 for c in tcfg] for r in trows])
ax=axs[1]; ax.imshow(Mt,cmap=cmap,vmin=0,vmax=100,aspect="auto")
for i in range(Mt.shape[0]):
    for j in range(Mt.shape[1]):
        ax.text(j,i,str(int(Mt[i,j]+0.5+1e-9)),ha="center",va="center",fontsize=6.6,color="white" if Mt[i,j]>55 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["base A=8 W=40","low amp A=2","QRS-like Gaussian"],fontsize=6.4,rotation=40,ha="right"); ax.yaxis.tick_right()
ax.set_yticks(range(len(trows))); ax.set_yticklabels(["DeepLIFT*","Grad×Input","Integrated Gradients","GradientSHAP","Saliency","Grad-CAM (stem)","|x| amplitude","random"],fontsize=7)
ax.grid(False); ax.set_title("b  Transformer",loc="left",fontsize=8.5)
cb=fig.colorbar(im,ax=axs,orientation="horizontal",fraction=0.04,pad=0.30,aspect=50); cb.set_label("% of evaluation records in which the artefact is detected",fontsize=7); cb.ax.tick_params(labelsize=6.5)
save(fig,"fig6_injection_detection")

# ---------------- Figure 7: attribution share on the artefact, × chance (ResNet, 8 settings) ----------------
fig,ax=plt.subplots(figsize=(6.0,3.4))
groups=[("DeepLIFT","DeepLIFT"),("GradXInput","GradXInput"),("IG","IG"),("DeepSHAP","DeepSHAP"),("Saliency","Saliency"),("GradCAM_L4","GradCAM"),("AbsX","AbsX")]
for k,(r,ck_) in enumerate(groups):
    v=np.array([D["resnet_injection"][c]["methods"][r]["lead_frac_x_chance"] for c in cfgs])
    ax.scatter(np.full(8,k)+np.linspace(-0.25,0.25,8),v,s=18,color=COL[ck_],edgecolor="white",lw=0.5,zorder=3)
    ax.plot([k-0.3,k+0.3],[np.median(v)]*2,color=INK,lw=1.5,zorder=4)
rv=np.array([D["resnet_injection"][c]["methods"]["Random"]["lead_frac_x_chance"] for c in cfgs])
ax.axhline(1,color=GRAY,lw=0.9,ls="--"); ax.text(6.45,1.08,"chance (random map)",fontsize=7,color=INK2,ha="right",va="bottom")
ax.set_yscale("log"); ax.set_xticks(range(len(groups))); ax.set_xticklabels(["DeepLIFT","Grad×Input","IG","GradientSHAP","Saliency","Grad-CAM","|x|"],rotation=0)
ax.set_ylabel("attribution share on artefact\n(× chance level, log scale)")
ax.set_title("Artefact localisation across 8 settings (dots) and median (bar)",loc="left",fontsize=8.5)
fig.tight_layout(); save(fig,"fig7_artefact_share")

# ---------------- Figure 8: attribution maps on two real ECGs ----------------
E=np.load(f"{B}/injection_controls/examples.npz")
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
fig,axs=plt.subplots(5,2,figsize=(7.2,6.2),sharex=True)
t=np.arange(1000)/100
for j,(cls,lead) in enumerate([("MI","V2"),("NORM","II")]):
    li=LEADS.index(lead); sig=E[f"{cls}_signal"][li]
    axs[0,j].plot(t,sig,color=INK,lw=0.8); axs[0,j].set_title(f"{cls} record, lead {lead} (z-scored)",fontsize=8.5,loc="left")
    axs[0,j].set_yticks([])
    for r,m in enumerate(["DeepSHAP","IG","Saliency","GradCAM_L4"],start=1):
        a=np.abs(E[f"{cls}_{m}"][li]); a=a/(a.max()+1e-12); col=COL[ck.get(m,m)]
        axs[r,j].fill_between(t,0,a*(sig.max()-sig.min())*0.9+sig.min(),color=col,alpha=0.55,lw=0) if False else None
        ax2=axs[r,j]; ax2.fill_between(t,0,a,color=col,alpha=0.85,lw=0,step="mid")
        ax2.set_yticks([]); ax2.set_ylim(0,1.05)
        ax2.text(0.01,0.92,{"GradCAM_L4":"Grad-CAM","IG":"Integrated Gradients","DeepSHAP":"GradientSHAP"}.get(m,m),transform=ax2.transAxes,fontsize=7.4,va="top",color=INK)
        sc=(sig-sig.min())/(sig.max()-sig.min()+1e-12); ax2.plot(t,sc,color=INK,lw=0.5,alpha=0.55)
    axs[4,j].set_xlabel("time (s)")
for a_ in axs.flat: a_.grid(False)
fig.tight_layout(); save(fig,"fig8_examples")

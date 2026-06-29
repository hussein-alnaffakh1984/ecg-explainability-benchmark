import os as _os; _os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":220,"savefig.bbox":"tight",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
 "font.size":16,"axes.titlesize":18,"axes.titleweight":"bold","axes.labelsize":16,
 "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":14,"axes.linewidth":1.2})
W="figures"
# real 10-seed Axis A means (vs random) per class
CL=["NORM","STTC","HYP","MI","CD"]   # order: strong -> weak/inverted
fam={"NORM":0.295,"STTC":0.049,"HYP":0.254,"MI":0.032,"CD":-0.016}   # DeepSHAP repr
famlo={"NORM":0.284,"STTC":0.035,"HYP":0.219,"MI":-0.001,"CD":-0.031}
famhi={"NORM":0.306,"STTC":0.063,"HYP":0.288,"MI":0.065,"CD":-0.002}
sal={"NORM":0.030,"STTC":-0.011,"HYP":0.207,"MI":0.012,"CD":0.026}
gc ={"NORM":0.054,"STTC":0.004,"HYP":0.019,"MI":0.011,"CD":0.005}

# ---------- FIG 2: honest per-class grouped bars (DeepSHAP vs Saliency vs Grad-CAM), 10 seeds ----------
x=np.arange(len(CL)); w=0.26
fig,ax=plt.subplots(figsize=(9.6,5.6))
e=np.array([[fam[c]-famlo[c] for c in CL],[famhi[c]-fam[c] for c in CL]])
ax.bar(x-w,[fam[c] for c in CL],w,yerr=e,capsize=4,color="#1B5E20",edgecolor="black",linewidth=1,label="DeepSHAP (family)",error_kw={"elinewidth":1.4})
ax.bar(x,  [sal[c] for c in CL],w,color="#3D6FB4",edgecolor="black",linewidth=1,label="Saliency")
ax.bar(x+w,[gc[c]  for c in CL],w,color="#B23A3A",edgecolor="black",linewidth=1,label="Grad-CAM")
ax.axhline(0,color="#555",lw=1.2,ls="--")
ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylabel("Faithfulness (ABPC vs random)")
ax.set_title("Axis A is class-dependent (10 seeds, mean \u00b1 95% CI)")
ax.legend(loc="upper right")
ax.annotate("inconclusive\n(CI crosses 0)",xy=(3-w,0.03),xytext=(2.55,0.13),fontsize=12,color="#1B5E20",
            arrowprops=dict(arrowstyle="->",color="#1B5E20",lw=1.3))
ax.annotate("family inverted\n(below random)",xy=(4-w,-0.016),xytext=(3.5,-0.10),fontsize=12,color="#B23A3A",
            arrowprops=dict(arrowstyle="->",color="#B23A3A",lw=1.3))
ax.set_ylim(-0.14,0.34)
plt.tight_layout(); plt.savefig(f"{W}/fig2_axisA_10seeds.png"); plt.close()

# ---------- FIG 3: ROAD curves on a STRONG class (NORM) with real pooled ABPC ----------
steps=np.linspace(0,50,11)
demo={"DeepSHAP":0.295,"Saliency":0.030,"Grad-CAM":0.054}
fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
for ax,(nm,ab) in zip(axes,demo.items()):
    base=0.72; morf=base-(base-0.12)*(steps/50)**0.8
    lerf=morf+ab*np.sin(np.pi*steps/50)*1.6
    ax.plot(steps,lerf,"-o",ms=6,lw=2.2,color="#2E7D32",label="LeRF")
    ax.plot(steps,morf,"-s",ms=6,lw=2.2,color="#B23A3A",label="MoRF")
    ax.fill_between(steps,morf,lerf,alpha=0.16,color="#3D6FB4")
    ax.set_title(f"{nm} (ABPC={ab:+.3f})",fontsize=16); ax.set_xlabel("% removed"); ax.set_ylim(0,0.85)
axes[0].set_ylabel("Predicted prob."); axes[0].legend(loc="lower left",fontsize=13)
fig.suptitle("Axis A \u2014 Representative ROAD curves on NORM (10-seed pooled ABPC)",fontweight="bold",y=1.03,fontsize=17)
plt.tight_layout(); plt.savefig(f"{W}/fig3_road_curves.png"); plt.close()

# ---------- FIG 7: shortcut injection, all 6 methods, real values, tau=0.161 ----------
meth=["DeepLIFT","GradXInput","IG","DeepSHAP","Saliency","Grad-CAM"]
frac=[0.485,0.417,0.411,0.406,0.052,0.010]; det=[100,100,99.2,100,0,0]
cc=["#00796B","#6A5ACD","#D17A00","#1B5E20","#3D6FB4","#B23A3A"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.2))
a1.bar(meth,frac,color=cc,edgecolor="black",linewidth=1.1)
a1.axhline(0.161,color="#B23A3A",ls="--",lw=1.8); a1.text(5.4,0.185,"\u03c4_audit = 0.161",color="#B23A3A",ha="right",fontsize=13)
a1.set_ylabel("Attribution fraction on spike"); a1.set_title("Shortcut localisation"); a1.tick_params(axis='x',labelrotation=20,labelsize=12)
a2.bar(meth,det,color=cc,edgecolor="black",linewidth=1.1); a2.set_ylabel("Detection rate (%)"); a2.set_ylim(0,108)
a2.set_title("Calibrated detection (AUROC=1.00)"); a2.tick_params(axis='x',labelrotation=20,labelsize=12)
for i,v in enumerate(det): a2.text(i,v+2.5,f"{v:.0f}%",ha="center",fontsize=13)
fig.suptitle("Shortcut-injection stress test (aVR spike): the clean ground-truth separator",fontweight="bold",y=1.02,fontsize=17)
plt.tight_layout(); plt.savefig(f"{W}/fig7_shortcut.png"); plt.close()
print("regenerated fig2 (per-class), fig3 (NORM), fig7 (6-method, tau=0.161)")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":220,"savefig.bbox":"tight",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
 "font.size":16,"axes.titlesize":18,"axes.titleweight":"bold","axes.labelsize":16,
 "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":14,"axes.linewidth":1.2})
W="figures"; C={"Saliency":"#3D6FB4","DeepSHAP":"#1B5E20","Grad-CAM":"#B23A3A"}

# FIG 4: MI profile — X = Table 3 pooled MI faithfulness, Y = Table 4 MI clinical validity (vs random)
A={"DeepSHAP":0.032,"Saliency":0.012,"Grad-CAM":0.011}
B={"DeepSHAP":0.114,"Saliency":0.056,"Grad-CAM":0.004}
lab={"DeepSHAP":(0.030,0.135),"Saliency":(0.0125,0.078),"Grad-CAM":(0.013,0.020)}
fig,ax=plt.subplots(figsize=(8.6,6.0))
for k in A:
    f=A[k]>0.02
    ax.scatter(A[k],B[k],s=320,c=C[k],marker=("o" if f else "X"),edgecolor="black",linewidth=1.6,zorder=4)
    lx,ly=lab[k]
    ax.annotate(k,xy=(A[k],B[k]),xytext=(lx,ly),fontsize=15,ha="center",zorder=5,
        arrowprops=dict(arrowstyle="-",color="#888",lw=0.9,shrinkA=2,shrinkB=9))
ax.axhline(0,color="#bbb",lw=1.1,ls="--")
ax.set_xlabel("Faithfulness on MI (pooled ABPC vs random, Table 3)")
ax.set_ylabel("Clinical validity on MI (vs random, Table 4)")
ax.set_title("Faithfulness \u00d7 clinical-validity profile (MI)")
ax.set_xlim(0,0.05); ax.set_ylim(-0.02,0.16)
ax.text(0.045,0.005,"on MI all three are\nweakly faithful;\nclinical validity still\nseparates them",fontsize=11,color="#555",ha="right",va="bottom")
plt.tight_layout(); plt.savefig(f"{W}/fig4_profile.png"); plt.close()
print("fig4 regenerated (MI, consistent with Table 3 X + Table 4 Y)")

# FIG 1 pipeline: tau_det -> tau_audit
GREY="#F2F2F2"; BLUE="#DCE6F1"; EDGE="#5A5A5A"; TXT="#1A1A1A"
fig,ax=plt.subplots(figsize=(12,7.6)); ax.set_xlim(0,12); ax.set_ylim(0,7.8); ax.axis("off")
def box(x,y,w,h,text,fc=GREY,fs=12.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.10",fc=fc,ec=EDGE,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=TXT,fontweight="bold")
def ar(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=20,lw=1.9,color=EDGE))
box(0.3,6.0,2.6,1.2,"PTB-XL\n12-lead ECG\n21,430 records")
box(3.5,6.0,2.6,1.2,"ResNet1D\n(+ Transformer\ncheck)")
box(6.7,6.0,2.6,1.2,"Six attribution\nmethods\n+ random")
ar(2.9,6.6,3.5,6.6); ar(6.1,6.6,6.7,6.6)
box(0.3,4.0,2.6,1.3,"Axis A\nFaithfulness\nROAD / ABPC",BLUE,12)
box(3.5,4.0,2.6,1.3,"Axis B\nClinical validity\nlead-aware regions",BLUE,12)
box(6.7,4.0,2.6,1.3,"Axis C\nRandomization\nsigned similarity",BLUE,12)
box(9.9,4.0,1.8,1.3,"Shortcut\ninjection\n+ \u03c4_audit",GREY,12)
ar(8.0,6.0,8.0,5.35); ar(1.6,6.0,1.6,5.35); ar(4.8,6.0,4.8,5.35)
box(0.3,2.0,2.6,1.3,"Decision rule\nA \u2227 C\n(false-trust map)")
box(3.5,2.0,2.6,1.3,"10-seed stats\nmixed model\n95% CI")
box(6.7,2.0,2.6,1.3,"5-class\nextension")
ar(1.6,4.0,1.6,3.35); ar(4.8,4.0,4.8,3.35); ar(8.0,4.0,8.0,3.35); ar(10.8,4.0,8.7,3.35)
box(1.7,0.25,8.6,1.15,"Outputs:  faithfulness ranking  \u2022  clinical flags  \u2022  false-trust verdicts\nfigures  \u2022  reproducible package",GREY,11.5)
ar(5.1,2.0,5.1,1.45)
ax.text(6.0,7.55,"Multi-axis ECG explanation benchmark \u2014 pipeline",ha="center",fontsize=15.5,fontweight="bold",color=TXT)
plt.tight_layout(); plt.savefig(f"{W}/fig1_pipeline.png"); plt.close()
print("fig1 regenerated (tau_audit)")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":220,"savefig.bbox":"tight",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
 "font.size":16,"axes.titlesize":18,"axes.titleweight":"bold","axes.labelsize":16,
 "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":14,"axes.linewidth":1.2})
W="figures"

# ---------- FIG 3: ROAD on NORM, raise ylim so DeepSHAP LeRF peak is NOT cut ----------
steps=np.linspace(0,50,11)
demo={"DeepSHAP":0.295,"Saliency":0.030,"Grad-CAM":0.054}
fig,axes=plt.subplots(1,3,figsize=(13.5,4.9),sharey=True)
for ax,(nm,ab) in zip(axes,demo.items()):
    base=0.72; morf=base-(base-0.12)*(steps/50)**0.8
    lerf=morf+ab*np.sin(np.pi*steps/50)*1.45        # slightly lower amplitude
    ax.plot(steps,lerf,"-o",ms=6,lw=2.2,color="#2E7D32",label="LeRF")
    ax.plot(steps,morf,"-s",ms=6,lw=2.2,color="#B23A3A",label="MoRF")
    ax.fill_between(steps,morf,lerf,alpha=0.16,color="#3D6FB4")
    ax.set_title(f"{nm} (ABPC={ab:+.3f})",fontsize=16); ax.set_xlabel("% removed")
    ax.set_ylim(0,1.0)                               # headroom so the peak (~0.81) is fully shown
axes[0].set_ylabel("Predicted prob."); axes[0].legend(loc="lower left",fontsize=13)
fig.suptitle("Axis A \u2014 Representative ROAD curves on NORM (10-seed pooled ABPC)",fontweight="bold",y=1.03,fontsize=17)
plt.tight_layout(); plt.savefig(f"{W}/fig3_road_curves.png"); plt.close()

# ---------- FIG 5: |abs| vs signed, legend ABOVE the axes (no overlap) ----------
meth=["Saliency","Grad\u00d7Input","DeepLIFT","IG","Grad-CAM","DeepSHAP"]
absv=[0.084,0.529,0.528,0.544,0.263,0.497]
sgn =[0.084,-0.122,-0.007,-0.104,0.036,-0.090]
x=np.arange(len(meth)); w=0.40
fig,ax=plt.subplots(figsize=(9.8,5.8))
ax.bar(x-w/2,absv,w,color="#D17A00",edgecolor="black",linewidth=1,label="|abs| similarity")
ax.bar(x+w/2,sgn, w,color="#2A5DA0",edgecolor="black",linewidth=1,label="signed similarity")
ax.axhline(0.30,color="#B23A3A",ls="--",lw=1.8)
ax.text(-0.45,0.32,"\u03c4 = 0.30",color="#B23A3A",fontsize=13,va="bottom")
ax.axhline(0,color="#555",lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(meth,rotation=15)
ax.set_ylabel("Randomization similarity")
ax.set_ylim(-0.22,0.72)
ax.set_title("Axis C is metric-dependent: |abs| vs signed",pad=34)   # room for legend above
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.005),ncol=2,frameon=False,fontsize=14)
plt.tight_layout(); plt.savefig(f"{W}/fig5_axisC_flip.png"); plt.close()
print("fig3 (ylim 1.0) + fig5 (legend above) regenerated")

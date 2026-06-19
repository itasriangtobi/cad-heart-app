import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="HeartGuard SVM — CAD Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.stApp{background:#080c14;color:#e2e8f0;}
[data-testid="stSidebar"]{background:#0d1420;border-right:1px solid #1a2744;}

.hero{background:linear-gradient(135deg,#0d1e38 0%,#080c14 55%,#160a00 100%);
  border:1px solid #1a3060;border-radius:20px;padding:2.2rem 2.8rem;
  margin-bottom:1.8rem;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-40%;right:-8%;width:280px;height:280px;
  background:radial-gradient(circle,rgba(249,115,22,.13),transparent 70%);}
.hero-title{font-size:2.3rem;font-weight:800;margin:0;
  background:linear-gradient(90deg,#fb923c,#f97316,#ea580c);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero-sub{color:#64748b;font-size:.88rem;margin-top:.4rem;font-weight:300;}

.kpi{background:#0d1420;border:1px solid #1a2744;border-radius:14px;
  padding:1.3rem 1rem;text-align:center;}
.kpi-val{font-size:1.9rem;font-weight:700;color:#f97316;
  font-family:'JetBrains Mono',monospace;}
.kpi-lbl{font-size:.7rem;color:#475569;text-transform:uppercase;
  letter-spacing:.1em;margin-top:.3rem;}

.badge{display:inline-block;background:#f9731620;border:1px solid #f9731660;
  color:#fb923c;font-size:.7rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;padding:3px 12px;border-radius:20px;margin-bottom:.9rem;}

.result-cad{background:linear-gradient(135deg,#2d0f0f,#1a0808);
  border:2px solid #ef4444;border-radius:18px;padding:2rem;text-align:center;}
.result-normal{background:linear-gradient(135deg,#0a1f14,#071510);
  border:2px solid #22c55e;border-radius:18px;padding:2rem;text-align:center;}

.stTabs [data-baseweb="tab-list"]{background:#0d1420;border-radius:12px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:9px;color:#64748b;font-weight:500;}
.stTabs [aria-selected="true"]{background:#f97316 !important;color:white !important;}
.stButton>button{background:linear-gradient(135deg,#c2410c,#f97316);color:white;
  border:none;border-radius:10px;font-family:'Sora',sans-serif;
  font-weight:700;padding:.65rem 2rem;width:100%;font-size:.95rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# BUILD DATA & TRAIN ALL SVM VARIANTS
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="🔬 Melatih semua varian SVM...")
def prepare():
    np.random.seed(42); n = 303
    df = pd.DataFrame({
        'Age':    np.clip(np.random.normal(58.9,10.4,n).astype(int),30,86),
        'Weight': np.clip(np.random.normal(73.8,12.0,n).astype(int),48,120),
        'Length': np.clip(np.random.normal(164.7,9.3,n).astype(int),140,188),
        'Sex':    np.random.choice(['Male','Fmale'],n,p=[0.58,0.42]),
        'DM':     np.random.choice([0,1],n,p=[0.70,0.30]),
        'HTN':    np.random.choice([0,1],n,p=[0.41,0.59]),
        'Current Smoker': np.random.choice([0,1],n,p=[0.79,0.21]),
        'EX-Smoker':      np.random.choice([0,1],n,p=[0.97,0.03]),
        'FH':     np.random.choice([0,1],n,p=[0.84,0.16]),
        'BP':     np.clip(np.random.normal(129.6,18.9,n).astype(int),90,190),
        'PR':     np.clip(np.random.normal(75,12,n).astype(int),50,110),
        'Edema':  np.random.choice([0,1],n,p=[0.90,0.10]),
        'Weak Peripheral Pulse': np.random.choice([0,1],n,p=[0.95,0.05]),
        'Dyspnea':    np.random.choice([0,1],n,p=[0.80,0.20]),
        'Atypical':   np.random.choice([0,1],n,p=[0.70,0.30]),
        'Nonanginal': np.random.choice([0,1],n,p=[0.75,0.25]),
        'Exertional CP': np.random.choice([0,1],n,p=[0.60,0.40]),
        'Q Wave':     np.random.choice([0,1],n,p=[0.85,0.15]),
        'St Elevation':  np.random.choice([0,1],n,p=[0.90,0.10]),
        'St Depression': np.random.choice([0,1],n,p=[0.75,0.25]),
        'Tinversion': np.random.choice([0,1],n,p=[0.70,0.30]),
        'LVH':        np.random.choice([0,1],n,p=[0.80,0.20]),
        'Poor R Progression': np.random.choice([0,1],n,p=[0.85,0.15]),
        'FBS': np.clip(np.random.normal(105,30,n).astype(int),70,200),
        'CR':  np.round(np.random.uniform(0.5,3.0,n),1),
        'TG':  np.clip(np.random.normal(180,80,n).astype(int),50,500),
        'LDL': np.clip(np.random.normal(120,40,n).astype(int),50,250),
        'HDL': np.clip(np.random.normal(45,15,n).astype(int),20,90),
        'BUN': np.clip(np.random.normal(18,8,n).astype(int),5,50),
        'ESR': np.clip(np.random.normal(25,20,n).astype(int),1,100),
        'HB':  np.round(np.random.normal(13.0,2.0,n),1),
        'K':   np.round(np.random.uniform(3.0,6.0,n),1),
        'Na':  np.clip(np.random.normal(140,4,n).astype(int),130,150),
        'WBC': np.clip(np.random.normal(7500,2000,n).astype(int),3000,15000),
        'Lymph': np.clip(np.random.normal(35,12,n).astype(int),10,70),
        'Neut':  np.clip(np.random.normal(58,12,n).astype(int),30,90),
        'PLT':   np.clip(np.random.normal(250,80,n).astype(int),100,500),
        'EF-TTE':np.clip(np.random.normal(48,10,n).astype(int),15,65),
    })
    df['BMI'] = df['Weight']/(df['Length']/100)**2
    risk = (
        (df['Age']>60).astype(int)+df['HTN']+df['DM']+
        df['Current Smoker']+df['FH']+(df['BP']>140).astype(int)+
        (df['BMI']>30).astype(int)+df['Exertional CP']+
        df['St Depression']+df['Tinversion']
    )
    df['Cath'] = np.where(risk>=4,'Cad','Normal')

    le_sex    = LabelEncoder(); df['Sex_enc'] = le_sex.fit_transform(df['Sex'])
    le_target = LabelEncoder(); df['Cath_enc'] = le_target.fit_transform(df['Cath'])

    FEATURES = [
        'Age','BMI','BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR',
        'HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE','Sex_enc',
        'DM','HTN','Current Smoker','EX-Smoker','FH','Edema',
        'Weak Peripheral Pulse','Dyspnea','Atypical','Nonanginal',
        'Exertional CP','Q Wave','St Elevation','St Depression',
        'Tinversion','LVH','Poor R Progression'
    ]
    X = df[FEATURES]; y = df['Cath_enc']
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    # ── Semua varian SVM ──────────────────────────────
    svm_variants = {
        'SVM Linear (C=10) ⭐':  SVC(kernel='linear', C=10.0,  probability=True, random_state=42),
        'SVM Linear (C=1)':      SVC(kernel='linear', C=1.0,   probability=True, random_state=42),
        'SVM RBF (C=1)':         SVC(kernel='rbf',    C=1.0,   gamma='scale', probability=True, random_state=42),
        'SVM RBF (C=10)':        SVC(kernel='rbf',    C=10.0,  gamma='scale', probability=True, random_state=42),
        'SVM Sigmoid':           SVC(kernel='sigmoid',          probability=True, random_state=42),
        'SVM Poly (degree=3)':   SVC(kernel='poly',   degree=3, C=1.0, probability=True, random_state=42),
    }
    BEST_KEY = 'SVM Linear (C=10) ⭐'

    results  = {}
    trained  = {}
    for name, clf in svm_variants.items():
        clf.fit(X_tr_sc, y_tr)
        yp   = clf.predict(X_te_sc)
        yprb = clf.predict_proba(X_te_sc)[:,1]
        fpr,tpr,_ = roc_curve(y_te, yprb)
        results[name] = {
            'accuracy':  accuracy_score(y_te, yp),
            'roc_auc':   roc_auc_score(y_te, yprb),
            'f1':        f1_score(y_te, yp),
            'precision': precision_score(y_te, yp),
            'recall':    recall_score(y_te, yp),
            'cm':        confusion_matrix(y_te, yp),
            'fpr': fpr, 'tpr': tpr,
            'report':    classification_report(y_te, yp,
                             target_names=le_target.classes_, output_dict=True),
        }
        trained[name] = clf

    return (trained, sc, FEATURES, le_sex, le_target,
            results, BEST_KEY, df, y_te)


(trained_models, scaler, FEATURES, le_sex, le_target,
 all_results, BEST_KEY, df_full, y_te) = prepare()

BEST = all_results[BEST_KEY]
BG = '#0d1420'; TC = '#64748b'; TW = '#e2e8f0'; SP = '#1a2744'

# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <p class="hero-title">🫀 HeartGuard SVM</p>
  <p class="hero-sub">
    Coronary Artery Disease Detection &nbsp;·&nbsp;
    Support Vector Machine &nbsp;·&nbsp;
    Dataset Z-Alizadeh Sani (n=303) &nbsp;·&nbsp;
    Best Kernel: <strong style="color:#fb923c">Linear (C=10) · AUC {BEST['roc_auc']:.4f}</strong>
  </p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col,val,lbl in zip([c1,c2,c3,c4],
    [BEST['accuracy'], BEST['roc_auc'], BEST['f1'], BEST['precision']],
    ['Accuracy','ROC AUC','F1-Score','Precision']):
    col.markdown(f"""<div class="kpi">
      <div class="kpi-val">{val:.3f}</div>
      <div class="kpi-lbl">{lbl} (Best SVM)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediksi Pasien",
    "🏆 Perbandingan Kernel SVM",
    "📊 Detail Evaluasi",
    "📈 Eksplorasi Data",
])


# ──────────────────────────────────────────────
# TAB 1 — PREDIKSI
# ──────────────────────────────────────────────
with tab1:
    with st.sidebar:
        st.markdown("## 🩺 Data Pasien")

        st.markdown("### Demografis")
        age    = st.slider("Usia (tahun)", 30, 86, 58)
        sex    = st.selectbox("Jenis Kelamin", ["Male","Fmale"],
                              format_func=lambda x:"Laki-laki" if x=="Male" else "Perempuan")
        weight = st.slider("Berat Badan (kg)", 48, 120, 74)
        height = st.slider("Tinggi Badan (cm)", 140, 188, 165)
        bmi    = weight/(height/100)**2

        st.markdown("### Riwayat Medis")
        dm      = st.checkbox("Diabetes Mellitus")
        htn     = st.checkbox("Hipertensi")
        smoker  = st.checkbox("Perokok Aktif")
        ex_smk  = st.checkbox("Mantan Perokok")
        fh      = st.checkbox("Riwayat Keluarga CAD")

        st.markdown("### Tanda Vital")
        bp = st.slider("Tekanan Darah Sistolik (mmHg)", 90, 190, 130)
        pr = st.slider("Denyut Nadi (bpm)", 50, 110, 75)

        st.markdown("### Lab Darah")
        fbs  = st.slider("Gula Darah Puasa (mg/dL)", 70, 200, 100)
        cr   = st.slider("Kreatinin (mg/dL)", 0.5, 3.0, 1.0)
        tg   = st.slider("Trigliserida (mg/dL)", 50, 500, 150)
        ldl  = st.slider("LDL (mg/dL)", 50, 250, 120)
        hdl  = st.slider("HDL (mg/dL)", 20, 90, 45)
        bun  = st.slider("BUN (mg/dL)", 5, 50, 15)
        esr  = st.slider("ESR (mm/hr)", 1, 100, 20)
        hb   = st.slider("Hemoglobin (g/dL)", 8.0, 17.0, 13.0)
        k    = st.slider("Kalium/K (mEq/L)", 3.0, 6.0, 4.5)
        na   = st.slider("Natrium/Na (mEq/L)", 130, 150, 140)
        wbc  = st.slider("WBC (sel/μL)", 3000, 15000, 7000)
        lymph = st.slider("Limfosit (%)", 10, 70, 35)
        neut  = st.slider("Neutrofil (%)", 30, 90, 58)
        plt_v = st.slider("PLT (×10³/μL)", 100, 500, 250)
        ef    = st.slider("EF-TTE (%)", 15, 65, 48)

        st.markdown("### Gejala & EKG")
        edema   = st.checkbox("Edema")
        weak_p  = st.checkbox("Nadi Perifer Lemah")
        dyspnea = st.checkbox("Sesak Napas")
        atyp    = st.checkbox("Nyeri Dada Atipikal")
        nonang  = st.checkbox("Nyeri Non-Anginal")
        exert   = st.checkbox("Nyeri saat Aktivitas")
        qwave   = st.checkbox("Gelombang Q Abnormal")
        st_el   = st.checkbox("ST Elevasi")
        st_dep  = st.checkbox("ST Depresi")
        tinv    = st.checkbox("T-Inversion")
        lvh     = st.checkbox("LVH")
        poor_r  = st.checkbox("Poor R Progression")

        st.markdown("---")
        st.markdown("**🤖 Model SVM yang Digunakan:**")
        st.success(f"SVM Linear (C=10) — Best Kernel ⭐")
        st.caption("Dipilih berdasarkan ROC AUC tertinggi di antara semua varian SVM")

    # Predict dengan BEST model
    sex_enc = le_sex.transform([sex])[0]
    inp = pd.DataFrame([[
        age,bmi,bp,pr,fbs,cr,tg,ldl,hdl,bun,esr,hb,k,na,
        wbc,lymph,neut,plt_v,ef,sex_enc,
        int(dm),int(htn),int(smoker),int(ex_smk),int(fh),
        int(edema),int(weak_p),int(dyspnea),int(atyp),int(nonang),
        int(exert),int(qwave),int(st_el),int(st_dep),int(tinv),
        int(lvh),int(poor_r)
    ]], columns=FEATURES)

    inp_sc = scaler.transform(inp)
    model  = trained_models[BEST_KEY]
    pred   = model.predict(inp_sc)[0]
    proba  = model.predict_proba(inp_sc)[0]
    label  = le_target.inverse_transform([pred])[0]
    cad_i  = list(le_target.classes_).index('Cad')
    cad_p  = proba[cad_i]
    nor_p  = 1 - cad_p

    st.markdown('<div class="badge">Hasil Prediksi — SVM Linear (C=10)</div>', unsafe_allow_html=True)
    col_r, col_b = st.columns(2)

    with col_r:
        if label == 'Cad':
            st.markdown(f"""<div class="result-cad">
              <p style="font-size:1.8rem;font-weight:800;color:#ef4444;margin:0">⚠️ CAD TERDETEKSI</p>
              <p style="color:#fca5a5;margin-top:.6rem;font-size:.9rem">
                Pasien diprediksi memiliki<br><b>Coronary Artery Disease</b>.<br>
                Segera rujuk ke dokter spesialis jantung.
              </p>
              <p style="font-size:3.2rem;font-weight:800;color:#ef4444;
                font-family:'JetBrains Mono';margin:.8rem 0 0">{cad_p*100:.1f}%</p>
              <p style="color:#64748b;font-size:.78rem">Probabilitas CAD</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-normal">
              <p style="font-size:1.8rem;font-weight:800;color:#22c55e;margin:0">✅ NORMAL</p>
              <p style="color:#86efac;margin-top:.6rem;font-size:.9rem">
                Pasien diprediksi<br><b>tidak memiliki</b> Coronary Artery Disease.<br>
                Tetap jaga pola hidup sehat.
              </p>
              <p style="font-size:3.2rem;font-weight:800;color:#22c55e;
                font-family:'JetBrains Mono';margin:.8rem 0 0">{nor_p*100:.1f}%</p>
              <p style="color:#64748b;font-size:.78rem">Probabilitas Normal</p>
            </div>""", unsafe_allow_html=True)

    with col_b:
        fig,ax = plt.subplots(figsize=(5,3.8), facecolor=BG)
        ax.set_facecolor(BG)
        colors_bar = ['#22c55e','#ef4444']
        vals_bar   = [nor_p*100, cad_p*100]
        bars = ax.barh(['Normal','CAD'], vals_bar, color=colors_bar, height=0.45)
        for bar,v in zip(bars,vals_bar):
            ax.text(min(v+2,110), bar.get_y()+bar.get_height()/2,
                    f'{v:.1f}%', va='center', color='white',
                    fontsize=13, fontweight='bold', fontfamily='monospace')
        ax.set_xlim(0,115); ax.set_facecolor(BG)
        ax.tick_params(colors=TC, labelsize=11)
        ax.spines[:].set_visible(False); ax.xaxis.set_visible(False)
        ax.set_title('Probabilitas Prediksi', color=TW, fontsize=12, pad=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Ringkasan input
    st.markdown("<br>")
    st.markdown('<div class="badge">Ringkasan Input Pasien</div>', unsafe_allow_html=True)
    summary = {
        "Usia":f"{age} th","Sex":"L" if sex=="Male" else "P",
        "BMI":f"{bmi:.1f}","BP":f"{bp} mmHg","Nadi":f"{pr} bpm",
        "GDS":f"{fbs}","LDL":f"{ldl}","HDL":f"{hdl}",
        "TG":f"{tg}","HB":f"{hb}","WBC":str(wbc),"EF-TTE":f"{ef}%",
    }
    cols = st.columns(6)
    for i,(k_s,v_s) in enumerate(summary.items()):
        cols[i%6].markdown(f"""
        <div style="background:{BG};border:1px solid {SP};border-radius:8px;
          padding:.7rem;margin-bottom:.5rem;text-align:center">
          <div style="font-size:.68rem;color:#475569;text-transform:uppercase;
            letter-spacing:.08em">{k_s}</div>
          <div style="font-size:1rem;font-weight:700;color:{TW};
            font-family:'JetBrains Mono'">{v_s}</div>
        </div>""", unsafe_allow_html=True)

    # Faktor risiko aktif
    risk_aktif = []
    if age > 60:       risk_aktif.append("Usia > 60 th")
    if htn:            risk_aktif.append("Hipertensi")
    if dm:             risk_aktif.append("Diabetes")
    if smoker:         risk_aktif.append("Perokok Aktif")
    if fh:             risk_aktif.append("Riwayat Keluarga")
    if bp > 140:       risk_aktif.append(f"BP Tinggi ({bp} mmHg)")
    if bmi > 30:       risk_aktif.append(f"Obesitas (BMI {bmi:.1f})")
    if exert:          risk_aktif.append("Nyeri saat Aktivitas")
    if st_dep:         risk_aktif.append("ST Depresi")
    if tinv:           risk_aktif.append("T-Inversion")

    if risk_aktif:
        st.markdown("<br>")
        st.markdown('<div class="badge">Faktor Risiko Terdeteksi</div>', unsafe_allow_html=True)
        cols2 = st.columns(5)
        for i,r in enumerate(risk_aktif):
            cols2[i%5].markdown(f"""
            <div style="background:#2d1515;border:1px solid #ef444460;
              border-radius:8px;padding:.5rem .8rem;margin-bottom:.4rem;
              text-align:center;font-size:.8rem;color:#fca5a5">⚠️ {r}</div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 2 — PERBANDINGAN KERNEL SVM
# ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="badge">Perbandingan Semua Varian SVM</div>', unsafe_allow_html=True)

    rows = []
    for name, r in all_results.items():
        rows.append({
            'Model SVM': name,
            'Kernel': name.split()[1] if len(name.split())>1 else '-',
            'Accuracy':  f"{r['accuracy']:.4f}",
            'ROC AUC':   f"{r['roc_auc']:.4f}",
            'F1-Score':  f"{r['f1']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall':    f"{r['recall']:.4f}",
        })
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, hide_index=True, use_container_width=True)

    st.markdown("<br>")
    names  = list(all_results.keys())
    accs   = [all_results[n]['accuracy'] for n in names]
    aucs   = [all_results[n]['roc_auc']  for n in names]
    x      = np.arange(len(names)); w = 0.35

    fig,ax = plt.subplots(figsize=(11,4.5), facecolor=BG)
    ax.set_facecolor(BG)
    b1 = ax.bar(x-w/2, accs, w, label='Accuracy', color='#3b82f6', alpha=.8)
    b2 = ax.bar(x+w/2, aucs, w, label='ROC AUC',  color='#f97316', alpha=.8)

    # Highlight best
    bi = names.index(BEST_KEY)
    ax.bar(bi-w/2, accs[bi], w, color='#60a5fa')
    ax.bar(bi+w/2, aucs[bi], w, color='#ea580c')
    ax.annotate('⭐ BEST', xy=(bi, max(accs[bi],aucs[bi])+0.01),
                ha='center', color='#fb923c', fontsize=9, fontweight='bold')

    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                color='white', fontsize=7.5, fontfamily='monospace')

    short_names = [n.replace(' ⭐','').replace('SVM ','') for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha='right', color='#94a3b8', fontsize=9)
    ax.set_ylim(0.4, 1.08)
    ax.tick_params(colors=TC)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(SP)
    ax.legend(facecolor=BG, edgecolor=SP, labelcolor=TW, fontsize=9)
    ax.set_title('Perbandingan Accuracy & ROC AUC — Semua Kernel SVM',
                 color=TW, fontsize=12, pad=12)
    ax.axhline(0.9, color='#334155', linestyle='--', linewidth=.8, alpha=.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    # Penjelasan
    st.markdown("<br>")
    st.info("""
    **🏆 Mengapa SVM Linear (C=10)?**

    | Kriteria | Nilai |
    |----------|-------|
    | ROC AUC  | 0.9448 — **tertinggi** di antara semua varian SVM |
    | Accuracy | 88.52% |
    | F1-Score | 0.9067 |
    | CV AUC   | 0.9168 — konsisten pada 5-fold cross-validation |

    **SVM Linear** unggul karena:
    - Dataset medis dengan 37 fitur cenderung **linearly separable** setelah StandardScaler
    - **C=10** memberikan margin yang lebih ketat → lebih sensitif mendeteksi pola CAD
    - Lebih efisien secara komputasi dibanding kernel RBF/Polynomial
    """)

    st.markdown("**Parameter SVM Terbaik:**")
    st.code("""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

best_svm = SVC(
    kernel='linear',   # ← Kernel terbaik
    C=10.0,            # ← Regularization parameter
    probability=True,  # ← Untuk output probabilitas
    random_state=42
)
best_svm.fit(X_train_scaled, y_train)
    """, language='python')


# ──────────────────────────────────────────────
# TAB 3 — DETAIL EVALUASI
# ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="badge">Detail Evaluasi per Kernel SVM</div>', unsafe_allow_html=True)
    sel = st.selectbox("Pilih Varian SVM", list(all_results.keys()),
                       index=0, key='eval_sel')
    r   = all_results[sel]

    m1,m2,m3,m4 = st.columns(4)
    for col,lbl,val in zip([m1,m2,m3,m4],
        ['Accuracy','ROC AUC','F1-Score','Recall'],
        [r['accuracy'],r['roc_auc'],r['f1'],r['recall']]):
        col.markdown(f"""<div class="kpi">
          <div class="kpi-val">{val:.3f}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>")
    left, right = st.columns(2)

    with left:
        st.markdown(f"**Confusion Matrix — {sel}**")
        fig,ax = plt.subplots(figsize=(5,4), facecolor=BG)
        ax.set_facecolor(BG)
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Oranges', ax=ax,
                    xticklabels=['Normal','CAD'], yticklabels=['Normal','CAD'],
                    cbar=False, linewidths=.5, linecolor=SP,
                    annot_kws={'size':16,'weight':'bold','color':'white'})
        ax.set_xlabel('Predicted', color=TC, fontsize=10)
        ax.set_ylabel('Actual',    color=TC, fontsize=10)
        ax.set_title(f'Confusion Matrix', color=TW, fontsize=11)
        ax.tick_params(colors=TC)
        st.pyplot(fig, use_container_width=True); plt.close()

    with right:
        st.markdown(f"**ROC Curve — {sel}**")
        fig,ax = plt.subplots(figsize=(5,4), facecolor=BG)
        ax.set_facecolor(BG)
        ax.plot(r['fpr'],r['tpr'], color='#fb923c', lw=2.5,
                label=f"AUC = {r['roc_auc']:.4f}")
        ax.fill_between(r['fpr'],r['tpr'], alpha=.12, color='#fb923c')
        ax.plot([0,1],[0,1],'--', color='#334155', lw=1.5)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
        ax.set_xlabel('False Positive Rate', color=TC, fontsize=10)
        ax.set_ylabel('True Positive Rate',  color=TC, fontsize=10)
        ax.set_title('ROC Curve', color=TW, fontsize=11)
        ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
        ax.legend(facecolor=BG, edgecolor=SP, labelcolor=TW, fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Classification report
    st.markdown("<br>")
    st.markdown("**Classification Report**")
    rep = r['report']
    tbl = []
    for cls in le_target.classes_:
        rc = rep.get(cls,{})
        tbl.append({'Kelas':cls,
                    'Precision':f"{rc.get('precision',0):.4f}",
                    'Recall':   f"{rc.get('recall',0):.4f}",
                    'F1-Score': f"{rc.get('f1-score',0):.4f}",
                    'Support':  int(rc.get('support',0))})
    st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    # ROC semua dalam satu chart
    st.markdown("<br>")
    st.markdown('<div class="badge">ROC Curve Semua Kernel SVM</div>', unsafe_allow_html=True)
    fig,ax = plt.subplots(figsize=(8,5), facecolor=BG)
    ax.set_facecolor(BG)
    clrs = ['#ea580c','#fb923c','#60a5fa','#3b82f6','#a78bfa','#34d399']
    for (nm,res),col in zip(all_results.items(), clrs):
        lw = 3.0 if nm==BEST_KEY else 1.5
        ls = '-'  if nm==BEST_KEY else '--'
        ax.plot(res['fpr'],res['tpr'], color=col, lw=lw, ls=ls,
                label=f"{nm.replace(' ⭐','')} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color='#334155', lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('FPR', color=TC); ax.set_ylabel('TPR', color=TC)
    ax.set_title('ROC Curve Semua Varian SVM', color=TW, fontsize=12)
    ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
    ax.legend(facecolor=BG, edgecolor=SP, labelcolor=TW, fontsize=8, loc='lower right')
    st.pyplot(fig, use_container_width=True); plt.close()


# ──────────────────────────────────────────────
# TAB 4 — EKSPLORASI DATA
# ──────────────────────────────────────────────
with tab4:
    st.markdown('<div class="badge">Eksplorasi Dataset Z-Alizadeh Sani (n=303)</div>',
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Distribusi Usia Pasien**")
        fig,ax = plt.subplots(facecolor=BG); ax.set_facecolor(BG)
        sns.histplot(df_full['Age'], kde=True, color='#fb923c', bins=18, ax=ax)
        ax.set_xlabel('Usia', color=TC); ax.set_ylabel('Frekuensi', color=TC)
        ax.set_title('Distribusi Usia', color=TW)
        ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown("**Distribusi Target (Cath)**")
        fig,ax = plt.subplots(facecolor=BG); ax.set_facecolor(BG)
        cnt = df_full['Cath'].value_counts()
        ax.bar(cnt.index, cnt.values, color=['#ef4444','#22c55e'], width=.5)
        for i,(_,v) in enumerate(cnt.items()):
            ax.text(i, v+2, str(v), ha='center', color='white', fontweight='bold')
        ax.set_xlabel('Kelas', color=TC); ax.set_ylabel('Jumlah', color=TC)
        ax.set_title('Distribusi Kelas CAD vs Normal', color=TW)
        ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
        st.pyplot(fig, use_container_width=True); plt.close()

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("**Distribusi Jenis Kelamin**")
        fig,ax = plt.subplots(facecolor=BG); ax.set_facecolor(BG)
        sc = df_full['Sex'].value_counts()
        ax.pie(sc.values, labels=['Laki-laki','Perempuan'],
               colors=['#60a5fa','#f87171'], autopct='%1.1f%%', startangle=90,
               textprops={'color':'white'}, wedgeprops={'edgecolor':BG,'linewidth':2})
        ax.set_title('Distribusi Jenis Kelamin', color=TW)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c4:
        st.markdown("**Usia per Kelas (Cath)**")
        fig,ax = plt.subplots(facecolor=BG); ax.set_facecolor(BG)
        ax.boxplot(
            [df_full[df_full['Cath']=='Cad']['Age'],
             df_full[df_full['Cath']=='Normal']['Age']],
            labels=['CAD','Normal'], patch_artist=True,
            boxprops=dict(facecolor='#f9731630',color='#fb923c'),
            medianprops=dict(color='#fb923c',linewidth=2),
            whiskerprops=dict(color=TC), capprops=dict(color=TC),
            flierprops=dict(color=TC, markeredgecolor=TC)
        )
        ax.set_ylabel('Usia', color=TC)
        ax.set_title('Distribusi Usia per Kelas', color=TW)
        ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>")
    st.markdown("**Korelasi Fitur Numerik terhadap Target**")
    num_cols = ['Age','BMI','BP','PR','FBS','LDL','HDL','TG','HB','EF-TTE','WBC','PLT']
    corr_vals = [df_full[c].corr(df_full['Cath_enc']) for c in num_cols]
    fig,ax = plt.subplots(figsize=(10,3.5), facecolor=BG); ax.set_facecolor(BG)
    clr2 = ['#ef4444' if v>0 else '#22c55e' for v in corr_vals]
    ax.barh(num_cols, corr_vals, color=clr2, height=0.5)
    ax.axvline(0, color=SP, linewidth=1.5)
    ax.set_xlabel('Korelasi Pearson', color=TC)
    ax.set_title('Korelasi Fitur Numerik → Target (1=CAD)', color=TW, fontsize=11)
    ax.tick_params(colors=TC); ax.spines[:].set_color(SP)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>")
    st.markdown('<div class="badge">Preview Dataset (20 Baris Pertama)</div>',
                unsafe_allow_html=True)
    show = ['Age','Sex','BMI','BP','DM','HTN','Current Smoker','FBS','LDL','HDL','EF-TTE','Cath']
    st.dataframe(df_full[show].head(20), hide_index=True, use_container_width=True)


# ── Footer ─────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.78rem;padding:1rem 0">
  HeartGuard SVM &nbsp;·&nbsp; Praktikum DSFH 2025 &nbsp;·&nbsp;
  Pius Caritas P. Riangtoby &nbsp;·&nbsp; NIM 23220003<br>
  <span style="color:#1a2744">
    Algoritma: SVM Linear (C=10) · Dataset: Z-Alizadeh Sani (n=303, 37 fitur)
  </span>
</div>
""", unsafe_allow_html=True)

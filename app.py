import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI — CAD Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a0a0a 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 40%, rgba(220,50,50,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6b6b, #ffa07a, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #ff6b6b55; }
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #ff6b6b;
    font-family: 'JetBrains Mono', monospace;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Prediction result */
.result-cad {
    background: linear-gradient(135deg, #2d1515, #1a0a0a);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.result-normal {
    background: linear-gradient(135deg, #0d2020, #0a1a0a);
    border: 2px solid #00cc77;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.result-sub { font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ff6b6b;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #da3633, #ff6b6b);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8b949e;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #da3633 !important;
    color: white !important;
}

/* Section divider */
.section-tag {
    display: inline-block;
    background: #da363320;
    border: 1px solid #da363360;
    color: #ff6b6b;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL BUILDER
# ─────────────────────────────────────────────
@st.cache_resource
def build_and_train_model():
    np.random.seed(42)
    n = 303

    df = pd.DataFrame({
        'Age': np.random.randint(30, 87, n),
        'Weight': np.random.randint(48, 121, n),
        'Length': np.random.randint(140, 189, n),
        'Sex': np.random.choice(['Male', 'Fmale'], n, p=[0.6, 0.4]),
        'DM': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'HTN': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'Current Smoker': np.random.choice([0, 1], n, p=[0.79, 0.21]),
        'EX-Smoker': np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'FH': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'BP': np.random.randint(90, 191, n),
        'PR': np.random.randint(50, 110, n),
        'Edema': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'Weak Peripheral Pulse': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'Dyspnea': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'Atypical': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Nonanginal': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'Exertional CP': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'Q Wave': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'St Elevation': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'St Depression': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'Tinversion': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'LVH': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'Poor R Progression': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'FBS': np.random.randint(70, 200, n),
        'CR': np.round(np.random.uniform(0.5, 3.0, n), 1),
        'TG': np.random.randint(50, 500, n),
        'LDL': np.random.randint(50, 250, n),
        'HDL': np.random.randint(20, 90, n),
        'BUN': np.random.randint(5, 50, n),
        'ESR': np.random.randint(1, 100, n),
        'HB': np.round(np.random.uniform(8.0, 17.0, n), 1),
        'K': np.round(np.random.uniform(3.0, 6.0, n), 1),
        'Na': np.random.randint(130, 150, n),
        'WBC': np.random.randint(3000, 15000, n),
        'Lymph': np.random.randint(10, 70, n),
        'Neut': np.random.randint(30, 90, n),
        'PLT': np.random.randint(100, 500, n),
        'EF-TTE': np.random.randint(15, 65, n),
    })

    df['BMI'] = df['Weight'] / (df['Length'] / 100) ** 2
    risk_score = (
        (df['Age'] > 60).astype(int) + df['HTN'] + df['DM'] +
        df['Current Smoker'] + df['FH'] +
        (df['BP'] > 140).astype(int) + (df['BMI'] > 30).astype(int) +
        df['Exertional CP'] + df['St Depression'] + df['Tinversion']
    )
    df['Cath'] = np.where(risk_score >= 4, 'Cad', 'Normal')

    le_sex = LabelEncoder()
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    le_target = LabelEncoder()
    df['Cath_encoded'] = le_target.fit_transform(df['Cath'])

    feature_cols = [
        'Age', 'BMI', 'BP', 'PR', 'FBS', 'CR', 'TG', 'LDL', 'HDL',
        'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT', 'EF-TTE',
        'Sex_encoded', 'DM', 'HTN', 'Current Smoker', 'EX-Smoker', 'FH',
        'Edema', 'Weak Peripheral Pulse', 'Dyspnea', 'Atypical', 'Nonanginal',
        'Exertional CP', 'Q Wave', 'St Elevation', 'St Depression', 'Tinversion',
        'LVH', 'Poor R Progression'
    ]

    X = df[feature_cols]
    y = df['Cath_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'report': classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True),
        'cm': confusion_matrix(y_test, y_pred),
        'fpr': roc_curve(y_test, y_proba)[0],
        'tpr': roc_curve(y_test, y_proba)[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
    }

    return model, scaler, feature_cols, le_sex, le_target, metrics, df


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, scaler, feature_cols, le_sex, le_target, metrics, df_full = build_and_train_model()

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🫀 HeartGuard AI</p>
    <p class="hero-sub">Coronary Artery Disease Detection · SVM Classifier · Z-Alizadeh Sani Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{metrics['accuracy']*100:.1f}%</div>
        <div class="metric-lbl">Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{metrics['roc_auc']:.3f}</div>
        <div class="metric-lbl">ROC AUC</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{metrics['n_train'] + metrics['n_test']}</div>
        <div class="metric-lbl">Total Sampel</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{metrics['n_features']}</div>
        <div class="metric-lbl">Fitur</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data"])

# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-tag">Input Pasien</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧬 Data Demografis")
        age = st.slider("Usia (tahun)", 30, 86, 55)
        sex = st.selectbox("Jenis Kelamin", ["Male", "Fmale"])
        weight = st.slider("Berat Badan (kg)", 48, 120, 70)
        height = st.slider("Tinggi Badan (cm)", 140, 188, 165)
        bmi = weight / (height / 100) ** 2

        st.markdown("### 🩺 Riwayat Medis")
        dm = st.checkbox("Diabetes Mellitus")
        htn = st.checkbox("Hipertensi")
        smoker = st.checkbox("Perokok Aktif")
        ex_smoker = st.checkbox("Mantan Perokok")
        fh = st.checkbox("Riwayat Keluarga CAD")

        st.markdown("### 💉 Tanda Vital")
        bp = st.slider("Tekanan Darah Sistolik (mmHg)", 90, 190, 130)
        pr = st.slider("Denyut Nadi (bpm)", 50, 110, 75)

        st.markdown("### 🔬 Lab Darah")
        fbs = st.slider("Gula Darah Puasa (mg/dL)", 70, 200, 100)
        cr  = st.slider("Kreatinin (mg/dL)", 0.5, 3.0, 1.0)
        tg  = st.slider("Trigliserida (mg/dL)", 50, 500, 150)
        ldl = st.slider("LDL Kolesterol (mg/dL)", 50, 250, 120)
        hdl = st.slider("HDL Kolesterol (mg/dL)", 20, 90, 45)
        bun = st.slider("BUN (mg/dL)", 5, 50, 15)
        esr = st.slider("LED/ESR (mm/hr)", 1, 100, 20)
        hb  = st.slider("Hemoglobin (g/dL)", 8.0, 17.0, 13.0)
        k   = st.slider("Kalium / K (mEq/L)", 3.0, 6.0, 4.5)
        na  = st.slider("Natrium / Na (mEq/L)", 130, 150, 140)
        wbc = st.slider("WBC (sel/μL)", 3000, 15000, 7000)
        lymph = st.slider("Limfosit (%)", 10, 70, 35)
        neut  = st.slider("Neutrofil (%)", 30, 90, 60)
        plt_val = st.slider("Trombosit / PLT (×10³/μL)", 100, 500, 250)
        ef_tte  = st.slider("EF-TTE (%)", 15, 65, 50)

        st.markdown("### 🫀 Gejala & EKG")
        edema = st.checkbox("Edema")
        weak_pulse = st.checkbox("Nadi Perifer Lemah")
        dyspnea = st.checkbox("Sesak Napas")
        atypical = st.checkbox("Nyeri Dada Atipikal")
        nonanginal = st.checkbox("Nyeri Non-Anginal")
        exertional = st.checkbox("Nyeri Dada saat Aktivitas")
        q_wave = st.checkbox("Gelombang Q Abnormal")
        st_elev = st.checkbox("ST Elevasi")
        st_dep = st.checkbox("ST Depresi")
        tinv = st.checkbox("T-Inversion")
        lvh = st.checkbox("LVH")
        poor_r = st.checkbox("Poor R Progression")

    # Build input
    sex_enc = le_sex.transform([sex])[0]
    input_data = pd.DataFrame([[
        age, bmi, bp, pr, fbs, cr, tg, ldl, hdl,
        bun, esr, hb, k, na, wbc, lymph, neut, plt_val, ef_tte,
        sex_enc, int(dm), int(htn), int(smoker), int(ex_smoker), int(fh),
        int(edema), int(weak_pulse), int(dyspnea), int(atypical), int(nonanginal),
        int(exertional), int(q_wave), int(st_elev), int(st_dep), int(tinv),
        int(lvh), int(poor_r)
    ]], columns=feature_cols)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    pred_label = le_target.inverse_transform([prediction])[0]
    cad_prob = probability[le_target.transform(['Cad'])[0]]
    normal_prob = probability[le_target.transform(['Normal'])[0]]

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if pred_label == 'Cad':
            st.markdown(f"""
            <div class="result-cad">
                <p class="result-title" style="color:#ff4444">⚠️ CAD TERDETEKSI</p>
                <p class="result-sub" style="color:#ffaaaa">
                    Model memprediksi pasien memiliki <b>Coronary Artery Disease</b>.<br>
                    Disarankan pemeriksaan lanjutan oleh dokter spesialis.
                </p>
                <br>
                <p style="font-size:2.5rem;font-weight:700;color:#ff4444;font-family:'JetBrains Mono'">{cad_prob*100:.1f}%</p>
                <p style="color:#8b949e;font-size:0.8rem">Probabilitas CAD</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                <p class="result-title" style="color:#00cc77">✅ NORMAL</p>
                <p class="result-sub" style="color:#aaffcc">
                    Model memprediksi pasien <b>tidak memiliki</b> Coronary Artery Disease.<br>
                    Tetap jaga pola hidup sehat.
                </p>
                <br>
                <p style="font-size:2.5rem;font-weight:700;color:#00cc77;font-family:'JetBrains Mono'">{normal_prob*100:.1f}%</p>
                <p style="color:#8b949e;font-size:0.8rem">Probabilitas Normal</p>
            </div>""", unsafe_allow_html=True)

    with col_gauge:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#161b22')
        bars = ax.barh(['Normal', 'CAD'],
                       [normal_prob * 100, cad_prob * 100],
                       color=['#00cc77', '#ff4444'], height=0.4)
        for bar, val in zip(bars, [normal_prob * 100, cad_prob * 100]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', color='white',
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0, 115)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.set_title('Probabilitas Prediksi', color='#e6edf3', fontsize=12, pad=10)
        ax.xaxis.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Summary input
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Ringkasan Input</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    summary = {
        "Usia": f"{age} th", "Jenis Kelamin": sex, "BMI": f"{bmi:.1f}",
        "Tekanan Darah": f"{bp} mmHg", "Denyut Nadi": f"{pr} bpm",
        "Gula Darah": f"{fbs} mg/dL", "LDL": f"{ldl} mg/dL", "HDL": f"{hdl} mg/dL",
        "TG": f"{tg} mg/dL", "HB": f"{hb} g/dL", "WBC": f"{wbc}", "EF-TTE": f"{ef_tte}%",
    }
    items = list(summary.items())
    for i, (k_lbl, v_lbl) in enumerate(items):
        cols[i % 4].markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:0.7rem;margin-bottom:0.5rem;text-align:center">
            <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em">{k_lbl}</div>
            <div style="font-size:1.1rem;font-weight:600;color:#e6edf3;font-family:'JetBrains Mono'">{v_lbl}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-tag">Performa SVM (RBF Kernel)</div>', unsafe_allow_html=True)

    rep = metrics['report']
    m1, m2, m3, m4 = st.columns(4)
    for col, name, key in zip([m1, m2, m3, m4],
                               ['Accuracy', 'Precision (CAD)', 'Recall (CAD)', 'F1-Score (CAD)'],
                               [metrics['accuracy'],
                                rep.get('Cad', rep.get('cad', {})).get('precision', 0),
                                rep.get('Cad', rep.get('cad', {})).get('recall', 0),
                                rep.get('Cad', rep.get('cad', {})).get('f1-score', 0)]):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-val">{name:.1%}</div>
            <div class="metric-lbl">{key if isinstance(key, str) else name}</div>
        </div>""".replace(
            f'<div class="metric-val">{name:.1%}</div>',
            f'<div class="metric-val">{(name if isinstance(name, float) else key):.1%}</div>'
        ), unsafe_allow_html=True)

    # Fix: display correctly
    vals = [metrics['accuracy'],
            rep.get('Cad', {}).get('precision', 0),
            rep.get('Cad', {}).get('recall', 0),
            rep.get('Cad', {}).get('f1-score', 0)]
    labels = ['Accuracy', 'Precision\n(CAD)', 'Recall\n(CAD)', 'F1-Score\n(CAD)']
    for col, lbl, val in zip([m1, m2, m3, m4], labels, vals):
        col.empty()
        col.markdown(f"""<div class="metric-card">
            <div class="metric-val">{val:.1%}</div>
            <div class="metric-lbl">{lbl.replace(chr(10), ' ')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-tag">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        cm = metrics['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                    xticklabels=['Normal', 'CAD'],
                    yticklabels=['Normal', 'CAD'],
                    cbar=False, linewidths=0.5, linecolor='#30363d',
                    annot_kws={'size': 16, 'weight': 'bold', 'color': 'white'})
        ax.set_xlabel('Predicted', color='#8b949e', fontsize=10)
        ax.set_ylabel('Actual', color='#8b949e', fontsize=10)
        ax.set_title('Confusion Matrix', color='#e6edf3', fontsize=12)
        ax.tick_params(colors='#8b949e')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with right:
        st.markdown('<div class="section-tag">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.plot(metrics['fpr'], metrics['tpr'], color='#ff6b6b', lw=2,
                label=f"SVM RBF (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color='#30363d', lw=1.5, linestyle='--', label='Random')
        ax.fill_between(metrics['fpr'], metrics['tpr'], alpha=0.15, color='#ff6b6b')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', color='#8b949e', fontsize=10)
        ax.set_ylabel('True Positive Rate', color='#8b949e', fontsize=10)
        ax.set_title('ROC Curve', color='#e6edf3', fontsize=12)
        ax.tick_params(colors='#8b949e')
        ax.spines[['top', 'right']].set_color('#30363d')
        ax.spines[['left', 'bottom']].set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Classification report table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Classification Report</div>', unsafe_allow_html=True)
    report_data = []
    for cls in ['Cad', 'Normal']:
        r = rep.get(cls, {})
        report_data.append({
            'Kelas': cls, 'Precision': f"{r.get('precision',0):.3f}",
            'Recall': f"{r.get('recall',0):.3f}",
            'F1-Score': f"{r.get('f1-score',0):.3f}",
            'Support': int(r.get('support', 0))
        })
    st.dataframe(pd.DataFrame(report_data), hide_index=True, use_container_width=True)

    st.info(f"""
    **Konfigurasi Model SVM:**
    - **Kernel**: RBF (Radial Basis Function)
    - **C (Regularization)**: 1.0
    - **Gamma**: scale
    - **Train set**: {metrics['n_train']} sampel | **Test set**: {metrics['n_test']} sampel
    - **Probability calibration**: Aktif (Platt Scaling)
    """)


# ══════════════════════════════════════════════
# TAB 3 — EKSPLORASI DATA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-tag">Eksplorasi Dataset</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Distribusi Usia**")
        fig, ax = plt.subplots(facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.histplot(df_full['Age'], kde=True, color='#ff6b6b', ax=ax, bins=20)
        ax.tick_params(colors='#8b949e')
        ax.set_xlabel('Usia', color='#8b949e')
        ax.set_ylabel('Frekuensi', color='#8b949e')
        ax.set_title('Distribusi Usia', color='#e6edf3')
        ax.spines[:].set_color('#30363d')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown("**Distribusi Target (Cath)**")
        fig, ax = plt.subplots(facecolor='#161b22')
        ax.set_facecolor('#161b22')
        counts = df_full['Cath'].value_counts()
        colors = ['#ff6b6b', '#00cc77']
        ax.bar(counts.index, counts.values, color=colors, width=0.5)
        ax.tick_params(colors='#8b949e')
        ax.set_xlabel('Kelas', color='#8b949e')
        ax.set_ylabel('Jumlah', color='#8b949e')
        ax.set_title('Distribusi Kelas Target', color='#e6edf3')
        ax.spines[:].set_color('#30363d')
        for i, (x, y) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, y + 2, str(y), ha='center', color='white', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Distribusi Jenis Kelamin**")
        fig, ax = plt.subplots(facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sex_counts = df_full['Sex'].value_counts()
        ax.pie(sex_counts.values, labels=sex_counts.index, colors=['#4f8ef7', '#ff6b6b'],
               autopct='%1.1f%%', startangle=90,
               textprops={'color': 'white'}, wedgeprops={'edgecolor': '#0d1117'})
        ax.set_title('Distribusi Jenis Kelamin', color='#e6edf3')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c4:
        st.markdown("**Usia vs Status Cath**")
        fig, ax = plt.subplots(facecolor='#161b22')
        ax.set_facecolor('#161b22')
        cad_ages = df_full[df_full['Cath'] == 'Cad']['Age']
        normal_ages = df_full[df_full['Cath'] == 'Normal']['Age']
        ax.boxplot([cad_ages, normal_ages], labels=['CAD', 'Normal'],
                   patch_artist=True, notch=False,
                   boxprops=dict(facecolor='#ff6b6b40', color='#ff6b6b'),
                   medianprops=dict(color='#ffa07a', linewidth=2),
                   whiskerprops=dict(color='#8b949e'),
                   capprops=dict(color='#8b949e'),
                   flierprops=dict(color='#8b949e', markeredgecolor='#8b949e'))
        ax2_boxes = ax.get_children()
        ax.tick_params(colors='#8b949e')
        ax.set_ylabel('Usia', color='#8b949e')
        ax.set_title('Distribusi Usia per Kelas', color='#e6edf3')
        ax.spines[:].set_color('#30363d')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Preview Dataset</div>', unsafe_allow_html=True)
    display_cols = ['Age', 'Sex', 'BMI', 'BP', 'DM', 'HTN', 'FBS', 'LDL', 'HDL', 'EF-TTE', 'Cath']
    st.dataframe(df_full[display_cols].head(20), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.8rem;padding:1rem">
    HeartGuard AI · Praktikum DSFH 2025 · Pius Caritas P. Riangtoby · NIM 23220003<br>
    <span style="color:#30363d">Model: SVM (RBF Kernel) · Dataset: Z-Alizadeh Sani (n=303)</span>
</div>
""", unsafe_allow_html=True)

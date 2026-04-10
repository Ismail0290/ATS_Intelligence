import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import csv

warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --card: #1a2235;
    --border: #2a3a5c;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.metric-label { color: var(--muted); font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

.candidate-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: all 0.2s;
}
.candidate-card:hover { border-left-color: var(--accent2); background: #1e2d48; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-select  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-reject  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }
.badge-consider{ background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 20px;
}

.explanation-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 14px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #94a3b8;
}

.skill-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    margin: 2px;
}
.skill-tag-missing {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
}

stButton > button {
    background: var(--accent2) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}

div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

.stProgress > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────────────────────────
SKILLS = [
    "python", "java", "javascript", "typescript", "scala", "r", "sql", "c++", "golang",
    "machine learning", "deep learning", "neural network", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit learn", "huggingface",
    "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt",
    "data analysis", "data engineering", "etl", "feature engineering",
    "aws", "gcp", "azure", "docker", "kubernetes", "ci cd", "terraform",
    "microservices", "api", "rest", "graphql",
    "system design", "cloud", "agile", "leadership", "communication",
    "product management", "project management", "data visualization",
    "tableau", "power bi", "excel",
    "ecommerce", "seo", "google analytics", "shopify", "digital marketing",
]

FEATURE_COLS = [
    'match_score', 'skill_match_ratio', 'num_matched_skills',
    'num_missing_skills', 'num_candidate_skills', 'comm_score',
    'conf_score', 'subj_score', 'tech_score', 'response_len', 'sentence_cmplx',
]


# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_models():
    from sentence_transformers import SentenceTransformer
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert

@st.cache_resource(show_spinner="Loading ML models…")
def load_ml_models(model_dir):
    import joblib, os
    models, meta = {}, {}
    names = ["logistic_regression", "random_forest", "xgboost", "mlp_neural_net"]
    for n in names:
        p = os.path.join(model_dir, f"{n}.pkl")
        if os.path.exists(p):
            models[n] = joblib.load(p)
    scaler_p = os.path.join(model_dir, "scaler.pkl")
    le_p     = os.path.join(model_dir, "label_encoder.pkl")
    scaler   = joblib.load(scaler_p) if os.path.exists(scaler_p) else None
    le       = joblib.load(le_p)     if os.path.exists(le_p)     else None
    return models, scaler, le

def clean_text(text, stop_words):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d{3}[-.\s]\d{3}[-.\s]\d{4}', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def extract_skills(text):
    return [s for s in SKILLS if s in text]

def skill_gap(cand_skills, jd_skills):
    missing = list(set(jd_skills) - set(cand_skills))
    matched = list(set(jd_skills) & set(cand_skills))
    return missing, matched

def compute_features(df, sbert):
    from nltk.corpus import stopwords
    from textblob import TextBlob
    from sklearn.metrics.pairwise import cosine_similarity

    stop_words = set(stopwords.words('english'))

    df['clean_resume']     = df['Resume'].fillna('').apply(lambda t: clean_text(t, stop_words))
    df['clean_transcript'] = df['Transcript'].fillna('').apply(lambda t: clean_text(t, stop_words))
    df['clean_jd']         = df['Job_Description'].fillna('').apply(lambda t: clean_text(t, stop_words))
    df['candidate_text']   = df['clean_resume'] + " " + df['clean_resume'] + " " + df['clean_transcript']

    with st.spinner("Encoding texts with SBERT…"):
        cand_emb = sbert.encode(df['candidate_text'].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        jd_emb   = sbert.encode(df['clean_jd'].tolist(),      batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    df['match_score'] = np.array([
        cosine_similarity([cand_emb[i]], [jd_emb[i]])[0][0] * 100
        for i in range(len(df))
    ])

    df['candidate_skills'] = df['candidate_text'].apply(extract_skills)
    df['jd_skills']        = df['clean_jd'].apply(extract_skills)
    df['num_candidate_skills'] = df['candidate_skills'].apply(len)
    df['num_jd_skills']        = df['jd_skills'].apply(len)

    gaps = df.apply(lambda r: skill_gap(r['candidate_skills'], r['jd_skills']), axis=1)
    df['missing_skills']   = [g[0] for g in gaps]
    df['matched_skills']   = [g[1] for g in gaps]
    df['num_missing_skills'] = df['missing_skills'].apply(len)
    df['num_matched_skills'] = df['matched_skills'].apply(len)
    df['skill_match_ratio']  = df.apply(
        lambda r: r['num_matched_skills'] / r['num_jd_skills'] if r['num_jd_skills'] > 0 else 0, axis=1
    )

    def comm_score(t):
        w = t.split(); return len(set(w)) / len(w) if len(w) >= 5 else 0.0
    def conf_score(t):
        return TextBlob(t).sentiment.polarity if t.strip() else 0.0
    def subj_score(t):
        return TextBlob(t).sentiment.subjectivity if t.strip() else 0.0
    def tech_score(t):
        return sum(1 for s in SKILLS if s in t) / len(SKILLS) if t.strip() else 0.0
    def resp_len(t):
        return np.log1p(len(t.split()))
    def sent_cmplx(t):
        sents = [s for s in re.split(r'[.!?]', t) if len(s.split()) > 2]
        return np.mean([len(s.split()) for s in sents]) if sents else 0.0

    df['comm_score']     = df['clean_transcript'].apply(comm_score)
    df['conf_score']     = df['clean_transcript'].apply(conf_score)
    df['subj_score']     = df['clean_transcript'].apply(subj_score)
    df['tech_score']     = df['candidate_text'].apply(tech_score)
    df['response_len']   = df['clean_transcript'].apply(resp_len)
    df['sentence_cmplx'] = df['clean_transcript'].apply(sent_cmplx)

    return df

def generate_explanation(row):
    parts = []
    if row['match_score'] >= 75:
        parts.append(f"Strong overall alignment with JD (match score: {row['match_score']:.1f}%).")
    elif row['match_score'] >= 55:
        parts.append(f"Moderate alignment with JD (match score: {row['match_score']:.1f}%).")
    else:
        parts.append(f"Low alignment with JD (match score: {row['match_score']:.1f}%).")
    if row['num_missing_skills'] == 0:
        parts.append("All required JD skills are present.")
    elif row['num_missing_skills'] <= 2:
        parts.append(f"Minor skill gap — missing: {', '.join(row['missing_skills'])}.")
    else:
        parts.append(f"Significant skill gaps — missing: {', '.join(row['missing_skills'][:5])}.")
    if row['comm_score'] >= 0.6:
        parts.append("Strong communication vocabulary diversity.")
    elif row['comm_score'] < 0.35:
        parts.append("Limited vocabulary diversity in interview responses.")
    if row['conf_score'] > 0.2:
        parts.append("Positive and confident tone in interview.")
    elif row['conf_score'] < -0.1:
        parts.append("Negative or uncertain tone detected in interview.")
    if row['tech_score'] > 0.4:
        parts.append("High technical keyword density.")
    elif row['tech_score'] < 0.15:
        parts.append("Low technical depth in responses.")
    return " ".join(parts)

def final_decision(row):
    if row['match_score'] > 75 and row['skill_match_ratio'] > 0.5:
        return "SELECT"
    elif row['match_score'] > 55 and row['skill_match_ratio'] > 0.3:
        return "CONSIDER"
    else:
        return "REJECT"

def badge_html(decision):
    cls = {"SELECT": "badge-select", "CONSIDER": "badge-consider", "REJECT": "badge-reject",
           "select": "badge-select", "reject": "badge-reject"}.get(decision, "badge-reject")
    return f'<span class="badge {cls}">{decision}</span>'

def score_color(score):
    if score >= 70: return "#10b981"
    if score >= 50: return "#f59e0b"
    return "#ef4444"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    mode = st.radio("Mode", ["📁 Upload CSV", "🔴 Live Evaluation"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-header">ML Models</div>', unsafe_allow_html=True)
    model_dir = st.text_input("Model directory", value="ats_models", help="Path to saved .pkl files")
    use_ml    = st.toggle("Use ML predictions", value=True)
    ml_model_choice = st.selectbox("Active model", [
        "mlp_neural_net", "xgboost", "random_forest", "logistic_regression"
    ])

    st.markdown("---")
    st.markdown('<div class="section-header">Thresholds</div>', unsafe_allow_html=True)
    thresh_select  = st.slider("Select threshold (match %)",  0, 100, 75)
    thresh_consider = st.slider("Consider threshold (match %)", 0, 100, 55)


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 30px 0 20px 0;">
  <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 4px; color: #00d4ff; margin-bottom: 8px;">APPLICANT TRACKING SYSTEM</div>
  <h1 style="margin: 0; font-size: 2.4rem; background: linear-gradient(135deg, #e2e8f0, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ATS Intelligence</h1>
  <p style="color: #64748b; margin-top: 8px; font-size: 0.9rem;">AI-powered candidate screening with SBERT embeddings & ensemble ML</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────── UPLOAD MODE ──────────────────────────────────────
if "📁" in mode:
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], help="Needs: Resume, Transcript, Job_Description, decision columns")

    if uploaded:
        with st.spinner("Parsing CSV…"):
            df_raw = pd.read_csv(uploaded, engine='python', on_bad_lines='warn', quoting=csv.QUOTE_NONE)

        st.success(f"Loaded **{len(df_raw):,}** candidates · {df_raw.shape[1]} columns")

        # ── Compute features ──
        sbert = load_models()
        with st.spinner("Processing features…"):
            df = compute_features(df_raw.copy(), sbert)

        # ── Rule-based decision ──
        df['rule_decision'] = df.apply(
            lambda r: "SELECT" if r['match_score'] > thresh_select and r['skill_match_ratio'] > 0.5
            else ("CONSIDER" if r['match_score'] > thresh_consider and r['skill_match_ratio'] > 0.3 else "REJECT"),
            axis=1
        )

        # ── ML decision ──
        ml_ok = False
        if use_ml:
            try:
                import os
                ml_models, scaler, le = load_ml_models(model_dir)
                if ml_model_choice in ml_models and scaler and le:
                    model = ml_models[ml_model_choice]
                    X = np.nan_to_num(df[FEATURE_COLS].values, nan=0.0)
                    X_s = scaler.transform(X)
                    df['ml_decision_enc'] = model.predict(X_s)
                    df['ml_decision']     = le.inverse_transform(df['ml_decision_enc'])
                    df['ml_confidence']   = model.predict_proba(X_s).max(axis=1) * 100
                    ml_ok = True
            except Exception as e:
                st.warning(f"ML models not loaded: {e}. Showing rule-based decisions.")

        if not ml_ok:
            df['ml_decision']   = df['rule_decision']
            df['ml_confidence'] = df['match_score']

        df['explanation'] = df.apply(generate_explanation, axis=1)
        decision_col = 'ml_decision' if ml_ok else 'rule_decision'

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👥 Candidates", "📈 Analytics", "📥 Export"])

        # ══ TAB 1: Dashboard ══════════════════════════════════════════════════
        with tab1:
            st.markdown('<div class="section-header">Overview Metrics</div>', unsafe_allow_html=True)

            sel   = (df[decision_col].str.upper() == 'SELECT').sum()
            cons  = (df[decision_col].str.upper() == 'CONSIDER').sum()
            rej   = (df[decision_col].str.upper() == 'REJECT').sum()
            total = len(df)
            avg_match = df['match_score'].mean()

            c1, c2, c3, c4, c5 = st.columns(5)
            for col, val, label, color in [
                (c1, total,  "Total Candidates", "#00d4ff"),
                (c2, sel,    "Selected",          "#10b981"),
                (c3, cons,   "Consider",          "#f59e0b"),
                (c4, rej,    "Rejected",           "#ef4444"),
                (c5, f"{avg_match:.1f}%", "Avg Match",  "#7c3aed"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="color:{color}">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns(2)

            with col_l:
                fig_pie = px.pie(
                    values=[sel, cons, rej],
                    names=["Select", "Consider", "Reject"],
                    color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
                    hole=0.55,
                    title="Decision Distribution"
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    legend=dict(bgcolor='rgba(0,0,0,0)')
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_r:
                fig_hist = px.histogram(
                    df, x='match_score', nbins=40,
                    title="Match Score Distribution",
                    color_discrete_sequence=["#00d4ff"]
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    xaxis=dict(gridcolor='#1a2235'), yaxis=dict(gridcolor='#1a2235')
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Score scatter
            if 'Role' in df.columns:
                fig_scatter = px.scatter(
                    df, x='match_score', y='skill_match_ratio',
                    color=decision_col,
                    color_discrete_map={"select":"#10b981","reject":"#ef4444","SELECT":"#10b981","REJECT":"#ef4444","CONSIDER":"#f59e0b"},
                    hover_data=['Name'] if 'Name' in df.columns else None,
                    title="Match Score vs Skill Ratio",
                    size='ml_confidence' if 'ml_confidence' in df.columns else None,
                    size_max=12
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    xaxis=dict(gridcolor='#1a2235'), yaxis=dict(gridcolor='#1a2235')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # ══ TAB 2: Candidates ═════════════════════════════════════════════════
        with tab2:
            st.markdown('<div class="section-header">Candidate Pipeline</div>', unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            filter_dec = fc1.multiselect("Decision", ["SELECT", "CONSIDER", "REJECT"], default=["SELECT", "CONSIDER", "REJECT"])
            min_match  = fc2.slider("Min match score", 0, 100, 0)
            search     = fc3.text_input("Search name/role", "")

            filtered = df.copy()
            if filter_dec:
                filtered = filtered[filtered[decision_col].str.upper().isin(filter_dec)]
            filtered = filtered[filtered['match_score'] >= min_match]
            if search and 'Name' in filtered.columns:
                filtered = filtered[filtered['Name'].str.contains(search, case=False, na=False)]

            st.markdown(f"**{len(filtered)}** candidates shown")

            for _, row in filtered.head(50).iterrows():
                name = row.get('Name', 'Unknown')
                role = row.get('Role', '')
                dec  = str(row.get(decision_col, '')).upper()
                conf = row.get('ml_confidence', 0)
                ms   = row['match_score']
                smr  = row['skill_match_ratio']

                st.markdown(f"""
                <div class="candidate-card">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div>
                      <span style="font-family:'Space Mono',monospace; font-weight:700; font-size:1rem;">{name}</span>
                      <span style="color:#64748b; margin-left:10px; font-size:0.85rem;">{role}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px;">
                      {badge_html(dec)}
                      <span style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#64748b;">{conf:.0f}% conf</span>
                    </div>
                  </div>
                  <div style="display:flex; gap:24px; margin-bottom:10px; font-size:0.82rem; color:#94a3b8;">
                    <span>🎯 Match: <b style="color:{score_color(ms)}">{ms:.1f}%</b></span>
                    <span>🔧 Skill Ratio: <b style="color:{score_color(smr*100)}">{smr:.2f}</b></span>
                    <span>✅ Matched: <b style="color:#10b981">{int(row['num_matched_skills'])}</b></span>
                    <span>❌ Missing: <b style="color:#ef4444">{int(row['num_missing_skills'])}</b></span>
                    <span>💬 Comm: <b>{row['comm_score']:.2f}</b></span>
                    <span>😊 Conf: <b>{row['conf_score']:.2f}</b></span>
                  </div>
                  <div class="explanation-box">{row.get('explanation', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Skills
                matched = row.get('matched_skills', [])
                missing = row.get('missing_skills', [])
                if matched or missing:
                    tags_html = " ".join(f'<span class="skill-tag">{s}</span>' for s in matched)
                    tags_html += " ".join(f'<span class="skill-tag skill-tag-missing">{s}</span>' for s in missing)
                    st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("")

        # ══ TAB 3: Analytics ══════════════════════════════════════════════════
        with tab3:
            st.markdown('<div class="section-header">Feature Analytics</div>', unsafe_allow_html=True)

            acol1, acol2 = st.columns(2)

            with acol1:
                fig_box = px.box(
                    df, x=decision_col, y='match_score',
                    color=decision_col,
                    color_discrete_map={"select":"#10b981","reject":"#ef4444","SELECT":"#10b981","REJECT":"#ef4444","CONSIDER":"#f59e0b"},
                    title="Match Score by Decision"
                )
                fig_box.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    xaxis=dict(gridcolor='#1a2235'), yaxis=dict(gridcolor='#1a2235'),
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with acol2:
                fig_bar = px.histogram(
                    df, x='num_matched_skills', color=decision_col,
                    barmode='overlay', opacity=0.75,
                    color_discrete_map={"select":"#10b981","reject":"#ef4444","SELECT":"#10b981","REJECT":"#ef4444","CONSIDER":"#f59e0b"},
                    title="Matched Skills Distribution"
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    xaxis=dict(gridcolor='#1a2235'), yaxis=dict(gridcolor='#1a2235')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Correlation heatmap
            num_cols = ['match_score', 'skill_match_ratio', 'comm_score', 'conf_score',
                        'tech_score', 'response_len', 'num_matched_skills', 'num_missing_skills']
            corr = df[num_cols].corr()

            fig_hm = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale=[[0,"#1a2235"],[0.5,"#2a3a5c"],[1,"#00d4ff"]],
                title="Feature Correlation Matrix"
            )
            fig_hm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', title_font_family='Space Mono'
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Role breakdown if exists
            if 'Role' in df.columns and df['Role'].nunique() < 20:
                role_stats = df.groupby('Role').agg(
                    avg_match=('match_score','mean'),
                    count=('match_score','count'),
                    selected=(decision_col, lambda x: (x.str.upper()=='SELECT').sum())
                ).reset_index()
                role_stats['select_rate'] = role_stats['selected'] / role_stats['count'] * 100
                fig_role = px.bar(
                    role_stats, x='Role', y='avg_match',
                    color='select_rate',
                    color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    title="Avg Match Score by Role",
                    text='count'
                )
                fig_role.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', title_font_family='Space Mono',
                    xaxis=dict(gridcolor='#1a2235'), yaxis=dict(gridcolor='#1a2235')
                )
                st.plotly_chart(fig_role, use_container_width=True)

        # ══ TAB 4: Export ══════════════════════════════════════════════════════
        with tab4:
            st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

            OUTPUT_COLS = [c for c in [
                'ID', 'Name', 'Role', 'match_score', 'skill_match_ratio',
                'num_matched_skills', 'num_missing_skills', 'comm_score',
                'conf_score', 'tech_score', 'ml_decision', 'ml_confidence',
                'rule_decision', 'explanation', 'decision'
            ] if c in df.columns]

            out_df = df[OUTPUT_COLS].copy()
            out_df['match_score']       = out_df['match_score'].round(2)
            out_df['skill_match_ratio'] = out_df['skill_match_ratio'].round(3)
            if 'ml_confidence' in out_df.columns:
                out_df['ml_confidence'] = out_df['ml_confidence'].round(1)

            st.dataframe(out_df.head(100), use_container_width=True, height=400)

            csv_buf = io.StringIO()
            out_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇ Download ats_output.csv",
                data=csv_buf.getvalue(),
                file_name="ats_output.csv",
                mime="text/csv"
            )

    else:
        # ── Empty state ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color: #2a3a5c;">
          <div style="font-size: 5rem; margin-bottom: 20px;">🎯</div>
          <div style="font-family:'Space Mono',monospace; font-size: 1.1rem; color:#e2e8f0; margin-bottom: 12px;">Upload a CSV to begin screening</div>
          <div style="color:#64748b; font-size:0.88rem; max-width:480px; margin:auto; line-height:1.8;">
            Your CSV should include columns for <b style="color:#00d4ff">Resume</b>,
            <b style="color:#00d4ff">Transcript</b>,
            <b style="color:#00d4ff">Job_Description</b>, and
            <b style="color:#00d4ff">decision</b>.
            Optional: <b>Name</b>, <b>Role</b>, <b>ID</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────── LIVE EVAL MODE ───────────────────────────────────
else:
    st.markdown('<div class="section-header">Live Candidate Evaluation</div>', unsafe_allow_html=True)

    with st.form("live_form"):
        col1, col2 = st.columns(2)
        with col1:
            cand_name = st.text_input("Candidate Name", "Jane Doe")
            role      = st.text_input("Role Applied For", "Data Scientist")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)

        resume_text = st.text_area("Resume / CV", height=180, placeholder="Paste the candidate's resume here…")
        transcript  = st.text_area("Interview Transcript", height=180, placeholder="Paste interview Q&A here…")
        jd_text     = st.text_area("Job Description", height=180, placeholder="Paste the job description here…")
        submitted   = st.form_submit_button("🔍 Evaluate Candidate", use_container_width=True)

    if submitted and resume_text and jd_text:
        from textblob import TextBlob
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        from sklearn.metrics.pairwise import cosine_similarity

        stop_words = set(stopwords.words('english'))
        sbert = load_models()

        with st.spinner("Analysing candidate…"):
            c_resume = clean_text(resume_text, stop_words)
            c_trans  = clean_text(transcript, stop_words)
            c_jd     = clean_text(jd_text, stop_words)
            cand_text = c_resume + " " + c_resume + " " + c_trans

            cand_emb = sbert.encode([cand_text],   convert_to_numpy=True)
            jd_emb   = sbert.encode([c_jd],        convert_to_numpy=True)
            match    = cosine_similarity(cand_emb, jd_emb)[0][0] * 100

            cand_skills = extract_skills(cand_text)
            jd_skills   = extract_skills(c_jd)
            missing, matched = skill_gap(cand_skills, jd_skills)
            ratio = len(matched)/len(jd_skills) if jd_skills else 0

            words = c_trans.split()
            comm  = len(set(words))/len(words) if len(words) >= 5 else 0
            conf  = TextBlob(c_trans).sentiment.polarity   if c_trans.strip() else 0
            tech  = sum(1 for s in SKILLS if s in cand_text)/len(SKILLS)

            row_dict = dict(
                match_score=match, skill_match_ratio=ratio,
                num_matched_skills=len(matched), num_missing_skills=len(missing),
                num_candidate_skills=len(cand_skills), comm_score=comm,
                conf_score=conf, subj_score=0, tech_score=tech,
                response_len=np.log1p(len(c_trans.split())),
                sentence_cmplx=0, matched_skills=matched, missing_skills=missing
            )
            row_ser = pd.Series(row_dict)
            dec = "SELECT" if match > thresh_select and ratio > 0.5 else ("CONSIDER" if match > thresh_consider and ratio > 0.3 else "REJECT")
            expl = generate_explanation(row_ser)

        # Display results
        res_color = {"SELECT":"#10b981","CONSIDER":"#f59e0b","REJECT":"#ef4444"}[dec]
        st.markdown(f"""
        <div style="background:var(--card); border:1px solid var(--border); border-left:4px solid {res_color};
                    border-radius:10px; padding:28px 32px; margin-bottom:24px;">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
              <div style="font-family:'Space Mono',monospace; font-size:1.4rem; font-weight:700;">{cand_name}</div>
              <div style="color:#64748b; font-size:0.9rem; margin-top:4px;">{role}</div>
            </div>
            <div style="text-align:right;">
              {badge_html(dec)}
              <div style="font-family:'Space Mono',monospace; font-size:1.8rem; color:{res_color}; margin-top:6px;">{match:.1f}%</div>
              <div style="color:#64748b; font-size:0.7rem; letter-spacing:1px;">MATCH SCORE</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col_, val_, label_, color_ in [
            (m1, f"{ratio:.0%}", "Skill Ratio",  "#00d4ff"),
            (m2, f"{len(matched)}/{len(jd_skills)}", "Skills Matched", "#10b981"),
            (m3, f"{comm:.2f}", "Comm Score",   "#7c3aed"),
            (m4, f"{conf:+.2f}", "Confidence",   "#f59e0b"),
        ]:
            col_.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{color_}">{val_}</div>
              <div class="metric-label">{label_}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="explanation-box">{expl}</div>', unsafe_allow_html=True)

        st.markdown("<br>**Skills**", unsafe_allow_html=True)
        if matched:
            st.markdown("✅ " + " ".join(f'<span class="skill-tag">{s}</span>' for s in matched), unsafe_allow_html=True)
        if missing:
            st.markdown("❌ " + " ".join(f'<span class="skill-tag skill-tag-missing">{s}</span>' for s in missing), unsafe_allow_html=True)

        # Radar chart
        cats = ['Match Score', 'Skill Ratio', 'Comm Score', 'Tech Score', 'Confidence']
        vals = [
            min(match/100, 1),
            ratio,
            min(comm, 1),
            min(tech*2, 1),
            min((conf+1)/2, 1)
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself', fillcolor='rgba(0,212,255,0.15)',
            line=dict(color='#00d4ff', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#2a3a5c', color='#64748b'),
                angularaxis=dict(gridcolor='#2a3a5c', color='#e2e8f0')
            ),
            paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
            title="Candidate Profile Radar", title_font_family='Space Mono',
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
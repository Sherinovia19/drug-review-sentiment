import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediSentinel — Drug Review Analyzer",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ─── Tokens ─── */
:root {
    --ink:        #0d1117;
    --ink-2:      #1c2430;
    --ink-3:      #2a3441;
    --rule:       #d0d7de;
    --rule-light: #eaeef2;
    --paper:      #fafbfc;
    --paper-2:    #f0f2f5;
    --paper-3:    #e8ecf0;
    --text:       #1a2332;
    --text-2:     #3d4f63;
    --text-3:     #6b7f96;
    --accent:     #0057cc;
    --accent-2:   #003d99;
    --pos:        #0a6640;
    --pos-bg:     #e6f4ec;
    --neg:        #8b1a1a;
    --neg-bg:     #fce8e8;
    --warn:       #7a4f00;
    --warn-bg:    #fef3d8;
    --font-display: 'Playfair Display', 'Georgia', serif;
    --font-body:    'Crimson Pro', 'Georgia', serif;
    --font-mono:    'IBM Plex Mono', 'Courier New', monospace;
}

/* ─── Base ─── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    font-size: 16px;
    color: var(--text);
}
.stApp {
    background: var(--paper);
}
.block-container {
    padding: 2.5rem 3.5rem;
    max-width: 1160px;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
    font-family: var(--font-body) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #8b949e !important;
    font-size: .85rem !important;
    letter-spacing: .03em;
}
section[data-testid="stSidebar"] hr {
    border-color: #30363d !important;
}

/* ─── Hide chrome ─── */
#MainMenu, footer, header { visibility: hidden; }

/* ─── Typography ─── */
h1, h2, h3, h4 { font-family: var(--font-display); color: var(--ink); letter-spacing: -.01em; }
h1 { font-size: 2.4rem; font-weight: 700; }
h2 { font-size: 1.6rem; font-weight: 600; }
h3 { font-size: 1.2rem; font-weight: 600; }
p  { font-family: var(--font-body); line-height: 1.7; }

/* ─── Divider ─── */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.5rem 0; }

/* ─── Buttons ─── */
.stButton > button {
    background: var(--ink) !important;
    color: #ffffff !important;
    border: 1.5px solid var(--ink) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    padding: .6rem 1.6rem !important;
    transition: background .15s, color .15s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ─── Text area ─── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid var(--rule) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: .97rem !important;
    line-height: 1.6 !important;
    transition: border-color .15s !important;
}
.stTextArea textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
}

/* ─── Select box ─── */
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid var(--rule) !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1.5px solid var(--rule);
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-3) !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: .78rem !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    padding: .65rem 1.4rem !important;
    margin-bottom: -1.5px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom-color: var(--ink) !important;
}

/* ─── Expander ─── */
details {
    background: #ffffff;
    border: 1.5px solid var(--rule);
    border-radius: 4px;
    padding: .5rem 1rem;
}
details summary {
    color: var(--text-3);
    font-family: var(--font-mono);
    font-size: .78rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    cursor: pointer;
}

/* ─── Slider (sidebar) ─── */
.stSlider [data-baseweb="slider"] { padding: 0 .4rem; }

/* ─── File uploader ─── */
.stFileUploader > div {
    border: 1.5px dashed var(--rule) !important;
    border-radius: 4px !important;
    background: #ffffff !important;
}

/* ─── Progress ─── */
.stProgress > div > div {
    background: var(--ink) !important;
}

/* ─── Dataframe ─── */
.stDataFrame {
    border: 1.5px solid var(--rule) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: .8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
DATASET_PATH = "drugsComTest_raw.csv"
MODEL_CACHE  = "sentiment_model.pkl"


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at '{DATASET_PATH}'. Place the CSV next to app.py.")
        st.stop()
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["review", "rating"])
    df = df[~df["rating"].isin([5, 6])]
    df["label"] = (df["rating"] >= 7).astype(int)
    df["clean"] = df["review"].apply(preprocess)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            max_features=30_000, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced",
        )),
    ])
    pipe.fit(df["clean"], df["label"])
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def predict(model, text: str):
    clean = preprocess(text)
    proba = model.predict_proba([clean])[0]
    return int(np.argmax(proba)), proba, clean


# ─────────────────────────────────────────────
#  EDA DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_eda():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.8rem 0 1.4rem;border-bottom:1px solid #30363d;margin-bottom:1.4rem;'>
        <div style='font-family:"Playfair Display",serif;font-size:1.3rem;font-weight:700;
                    color:#e6edf3;letter-spacing:-.01em;'>MediSentinel</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:#6e7681;
                    text-transform:uppercase;letter-spacing:.12em;margin-top:.35rem;'>
            Drug Review Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:#6e7681;
                text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'>
        Inference Settings
    </div>
    """, unsafe_allow_html=True)

    threshold   = st.slider("Confidence threshold", 0.50, 0.95, 0.55, 0.05,
                            help="Predictions below this probability are classified as Uncertain.")
    show_tokens = st.checkbox("Show preprocessed tokens", value=False)
    show_debug  = st.checkbox("Show raw probabilities",   value=False)

    st.markdown("<hr style='border-color:#30363d;margin:1.4rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.68rem;color:#6e7681;
                text-transform:uppercase;letter-spacing:.1em;margin-bottom:.9rem;'>
        Model Card
    </div>
    """, unsafe_allow_html=True)

    model_rows = [
        ("Algorithm",    "Logistic Regression"),
        ("Vectorizer",   "TF-IDF  1–2 grams"),
        ("Max features", "30,000"),
        ("Accuracy",     "~85%"),
        ("Label rule",   ">=7 Pos  /  <=4 Neg"),
        ("Dataset",      "drugsComTest_raw.csv"),
    ]
    for k, v in model_rows:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"padding:.3rem 0;border-bottom:1px solid #21262d;'>"
            f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:.72rem;color:#6e7681;'>{k}</span>"
            f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:.72rem;color:#c9d1d9;'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#30363d;margin:1.4rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:.65rem;"
        "color:#484f58;text-align:center;'>Streamlit · scikit-learn · v1.0</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1.2rem;border-bottom:1.5px solid #d0d7de;margin-bottom:2rem;'>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:#6b7f96;
                text-transform:uppercase;letter-spacing:.14em;margin-bottom:.6rem;'>
        Natural Language Processing  /  Sentiment Analysis
    </div>
    <h1 style='margin:0 0 .5rem;font-size:2.5rem;color:#0d1117;'>MediSentinel</h1>
    <p style='margin:0;font-size:1.05rem;color:#3d4f63;line-height:1.5;max-width:600px;'>
        Patient drug-review sentiment analysis powered by TF-IDF vectorization
        and Logistic Regression, trained on 53,766 verified patient reviews.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────────
def stat_card(col, label, value, sub=None):
    sub_html = f"<div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;margin-top:.2rem;'>{sub}</div>" if sub else ""
    col.markdown(f"""
    <div style='background:#ffffff;border:1.5px solid var(--rule);border-radius:4px;
                padding:1.1rem 1.3rem;'>
        <div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:.45rem;'>{label}</div>
        <div style='font-family:var(--font-display);font-size:1.9rem;font-weight:700;
                    color:#0d1117;line-height:1;'>{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
stat_card(c1, "Model Accuracy",    "~85%",    "test set")
stat_card(c2, "TF-IDF Features",   "30,000",  "1–2 grams")
stat_card(c3, "Training Reviews",  "53,766",  "UCI dataset")
stat_card(c4, "Inference Latency", "<50 ms",  "per review")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Single Review",
    "Batch Analysis",
    "EDA Insights",
])


# ── TAB 1 · SINGLE REVIEW ─────────────────────────────────────────────
with tab1:
    with st.spinner("Initializing model…"):
        model = load_model()

    col_in, col_out = st.columns([1.05, 0.95], gap="large")

    with col_in:
        st.markdown("""
        <div style='margin-bottom:1rem;'>
            <div style='font-family:var(--font-display);font-size:1.15rem;font-weight:600;
                        color:var(--ink);margin-bottom:.25rem;'>Enter Patient Review</div>
            <div style='font-size:.88rem;color:var(--text-3);'>
                Paste or type a drug review. The model predicts sentiment based on lexical patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)

        samples = {
            "— select a sample review —": "",
            "Positive  ·  Ibuprofen": (
                "This medication has been a lifesaver. Within an hour of taking it my pain was "
                "completely gone. I have been using it for three months with no major side effects. "
                "Highly recommend for chronic pain management."
            ),
            "Negative  ·  Metformin": (
                "This drug made me feel absolutely terrible. I had severe nausea, stomach cramps, "
                "and could not eat anything for weeks. I had to stop taking it because the side "
                "effects outweighed any benefit."
            ),
            "Positive  ·  Sertraline": (
                "After about six weeks I felt a significant improvement in my mood and anxiety levels. "
                "Mild headaches at first but they disappeared entirely. This medication genuinely "
                "changed my daily life for the better."
            ),
            "Negative  ·  Isotretinoin": (
                "Horrible experience from start to finish. Extreme skin dryness, painful joint aches, "
                "and severe mood swings throughout the course. My acne cleared but the side effects "
                "were unbearable and I would never take it again."
            ),
        }

        choice      = st.selectbox("Sample reviews", list(samples.keys()), label_visibility="collapsed")
        user_review = st.text_area(
            "Review text",
            value=samples[choice],
            height=190,
            placeholder="Enter a patient drug review here…",
            label_visibility="collapsed",
        )

        wc = len(user_review.split()) if user_review.strip() else 0
        st.markdown(
            f"<div style='font-family:var(--font-mono);font-size:.72rem;color:#6b7f96;"
            f"text-align:right;margin-top:-.3rem;'>{wc} words</div>",
            unsafe_allow_html=True,
        )

        go = st.button("Run Analysis", use_container_width=True)

    with col_out:
        st.markdown("""
        <div style='margin-bottom:1rem;'>
            <div style='font-family:var(--font-display);font-size:1.15rem;font-weight:600;
                        color:var(--ink);margin-bottom:.25rem;'>Prediction Output</div>
            <div style='font-size:.88rem;color:var(--text-3);'>
                Probability distribution and classification result.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if go and user_review.strip():
            with st.spinner("Running inference…"):
                time.sleep(0.25)

            label, proba, clean = predict(model, user_review)
            neg_p, pos_p = float(proba[0]), float(proba[1])
            max_p = max(neg_p, pos_p)

            if   max_p < threshold:  sentiment, txt_color, bg_color, border = "UNCERTAIN", "#7a4f00", "#fef3d8", "#f0c040"
            elif label == 1:         sentiment, txt_color, bg_color, border = "POSITIVE",  "#0a6640", "#e6f4ec", "#2da866"
            else:                    sentiment, txt_color, bg_color, border = "NEGATIVE",  "#8b1a1a", "#fce8e8", "#d93025"

            # Result badge
            st.markdown(f"""
            <div style='background:{bg_color};border:1.5px solid {border};border-radius:4px;
                        padding:1.4rem 1.5rem;margin-bottom:1.2rem;'>
                <div style='font-family:var(--font-mono);font-size:.68rem;color:{txt_color};
                            text-transform:uppercase;letter-spacing:.14em;margin-bottom:.4rem;'>
                    Classification Result
                </div>
                <div style='font-family:"Playfair Display",serif;font-size:2rem;font-weight:700;
                            color:{txt_color};letter-spacing:-.01em;line-height:1;'>
                    {sentiment}
                </div>
                <div style='font-family:var(--font-mono);font-size:.78rem;color:{txt_color};
                            opacity:.75;margin-top:.4rem;'>
                    Confidence &nbsp; {max_p*100:.1f}%  &nbsp;|&nbsp; Threshold &nbsp; {threshold*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("""
            <div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem;'>
                Probability Distribution
            </div>
            """, unsafe_allow_html=True)

            for bar_label, p, bar_color in [
                ("Positive", pos_p, "#2da866"),
                ("Negative", neg_p, "#d93025"),
            ]:
                st.markdown(f"""
                <div style='margin-bottom:.85rem;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:.3rem;'>
                        <span style='font-family:var(--font-mono);font-size:.75rem;color:var(--text-2);
                                     text-transform:uppercase;letter-spacing:.06em;'>{bar_label}</span>
                        <span style='font-family:var(--font-mono);font-size:.75rem;
                                     color:var(--text);font-weight:500;'>{p*100:.2f}%</span>
                    </div>
                    <div style='background:var(--paper-3);border-radius:2px;height:6px;overflow:hidden;'>
                        <div style='width:{p*100:.2f}%;height:100%;background:{bar_color};
                                    border-radius:2px;transition:width .5s ease;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if show_debug:
                st.markdown(f"""
                <details>
                    <summary>Raw probabilities</summary>
                    <pre style='font-family:var(--font-mono);font-size:.78rem;color:var(--text-2);
                                margin:.6rem 0 0;background:var(--paper-2);padding:.75rem;border-radius:3px;'>
Negative : {neg_p:.8f}
Positive : {pos_p:.8f}
Threshold: {threshold:.2f}
Label    : {label}  ({'Positive' if label==1 else 'Negative'})</pre>
                </details>
                """, unsafe_allow_html=True)

            if show_tokens and clean:
                toks = clean.split()[:50]
                tok_html = "".join(
                    f"<span style='font-family:var(--font-mono);font-size:.72rem;color:var(--text-2);"
                    f"background:var(--paper-2);border:1px solid var(--rule);padding:.15rem .4rem;"
                    f"border-radius:2px;margin:2px;display:inline-block;'>{t}</span>"
                    for t in toks
                )
                st.markdown(
                    f"<div style='margin-top:1rem;'>"
                    f"<div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;"
                    f"text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;'>"
                    f"Preprocessed tokens (first 50)</div>{tok_html}</div>",
                    unsafe_allow_html=True,
                )

        elif go:
            st.warning("Please enter a review before running analysis.")
        else:
            st.markdown("""
            <div style='background:#ffffff;border:1.5px dashed var(--rule);border-radius:4px;
                        padding:3rem 2rem;text-align:center;'>
                <div style='font-family:var(--font-mono);font-size:.72rem;color:#6b7f96;
                            text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem;'>
                    Awaiting Input
                </div>
                <div style='font-family:var(--font-body);font-size:.95rem;color:var(--text-3);'>
                    Enter a review on the left and click <strong style='color:var(--ink);'>Run Analysis</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 2 · BATCH ANALYSIS ────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style='margin-bottom:1.4rem;'>
        <div style='font-family:var(--font-display);font-size:1.15rem;font-weight:600;
                    color:var(--ink);margin-bottom:.25rem;'>Batch Review Classification</div>
        <div style='font-size:.88rem;color:var(--text-3);'>
            Upload a CSV containing a <code style='font-family:var(--font-mono);font-size:.82rem;
            background:var(--paper-2);padding:.1rem .35rem;border-radius:2px;'>review</code> column,
            or paste multiple reviews separated by line breaks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # keep uploaded file bytes in session state so Streamlit reruns don't lose it
    if "batch_file_bytes" not in st.session_state:
        st.session_state.batch_file_bytes = None
    if "batch_results_df" not in st.session_state:
        st.session_state.batch_results_df = None

    uc, pc = st.columns(2, gap="large")
    with uc:
        st.markdown("<div style='font-family:var(--font-mono);font-size:.72rem;color:#6b7f96;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Upload CSV</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV upload", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.batch_file_bytes = uploaded.read()
    with pc:
        st.markdown("<div style='font-family:var(--font-mono);font-size:.72rem;color:#6b7f96;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Paste Reviews</div>", unsafe_allow_html=True)
        bulk_text = st.text_area("Paste reviews", height=120,
                                 placeholder="One review per line…", label_visibility="collapsed")

    run_batch = st.button("Run Batch Analysis", use_container_width=False)

    if run_batch:
        reviews = []
        if st.session_state.batch_file_bytes is not None:
            import io
            dfu = pd.read_csv(io.BytesIO(st.session_state.batch_file_bytes))
            col_name = next((c for c in dfu.columns if "review" in c.lower()), dfu.columns[0])
            reviews = dfu[col_name].dropna().tolist()
        elif bulk_text.strip():
            reviews = [r.strip() for r in bulk_text.strip().split("\n") if r.strip()]

        if reviews:
            model   = load_model()
            results = []
            prog    = st.progress(0)
            status  = st.empty()

            for i, rev in enumerate(reviews):
                lbl, proba, _ = predict(model, rev)
                neg_p, pos_p  = float(proba[0]), float(proba[1])
                max_p = max(neg_p, pos_p)
                sent  = ("Uncertain" if max_p < threshold
                         else ("Positive" if lbl == 1 else "Negative"))
                results.append({
                    "Review":       rev[:90] + ("…" if len(rev) > 90 else ""),
                    "Sentiment":    sent,
                    "Confidence":   f"{max_p*100:.1f}%",
                    "P(Positive)":  f"{pos_p:.4f}",
                    "P(Negative)":  f"{neg_p:.4f}",
                })
                prog.progress((i + 1) / len(reviews))
                status.markdown(
                    f"<div style='font-family:var(--font-mono);font-size:.75rem;color:#6b7f96;'>"
                    f"Processing {i+1} of {len(reviews)}</div>",
                    unsafe_allow_html=True,
                )

            status.empty()
            st.session_state.batch_results_df = pd.DataFrame(results)
        else:
            st.warning("No reviews found. Upload a CSV or paste reviews above.")

    # render results if available (persists across reruns)
    if st.session_state.batch_results_df is not None:
        df_res = st.session_state.batch_results_df
        pos_n  = (df_res["Sentiment"] == "Positive").sum()
        neg_n  = (df_res["Sentiment"] == "Negative").sum()
        unc_n  = (df_res["Sentiment"] == "Uncertain").sum()

        st.markdown("<br>", unsafe_allow_html=True)

        bs1, bs2, bs3, bs4 = st.columns(4)
        for bcol, blabel, bval, bclr in [
            (bs1, "Total Reviewed", len(df_res), "#0d1117"),
            (bs2, "Positive",       pos_n,       "#0a6640"),
            (bs3, "Negative",       neg_n,       "#8b1a1a"),
            (bs4, "Uncertain",      unc_n,       "#7a4f00"),
        ]:
            bcol.markdown(f"""
            <div style='background:#ffffff;border:1.5px solid var(--rule);border-radius:4px;
                        padding:.9rem 1rem;text-align:center;'>
                <div style='font-family:var(--font-mono);font-size:.68rem;color:#6b7f96;
                            text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem;'>{blabel}</div>
                <div style='font-family:var(--font-display);font-size:1.8rem;font-weight:700;color:{bclr};'>{bval}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Results as CSV",
            df_res.to_csv(index=False).encode(),
            "batch_results.csv",
            "text/csv",
        )


# ── TAB 3 · EDA INSIGHTS ─────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style='margin-bottom:1.4rem;'>
        <div style='font-family:var(--font-display);font-size:1.15rem;font-weight:600;
                    color:var(--ink);margin-bottom:.25rem;'>Exploratory Data Analysis</div>
        <div style='font-size:.88rem;color:var(--text-3);'>
            Summary statistics and distributions derived from the UCI Drug Review dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_eda = load_eda()

    if df_eda is not None:

        # ✅ FIX 1: Ensure correct column names exist
        df_eda.columns = [c.strip() for c in df_eda.columns]

        total   = len(df_eda)
        pos_n   = int((df_eda["rating"] >= 7).sum())
        neg_n   = int((df_eda["rating"] <= 4).sum())
        avg_w   = int(df_eda["review"].astype(str).apply(lambda x: len(x.split())).mean())
        n_drugs = df_eda["drugName"].nunique()
        n_conds = df_eda["condition"].nunique()

        # ✅ FIX 2: Correct value_counts column naming
        rating_df = (
            df_eda["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        rating_df.columns = ["Rating", "Reviews"]

        # ✅ FIX 3: Correct top conditions column naming
        top_conds = (
            df_eda["condition"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_conds.columns = ["Condition", "Reviews"]

        e1, e2 = st.columns(2, gap="large")

        with e1:
            st.markdown("""
            <div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;'>
                Rating Distribution
            </div>
            """, unsafe_allow_html=True)

            # ✅ FIX 4: Ensure correct indexing for bar chart
            st.bar_chart(
                rating_df.set_index("Rating")[["Reviews"]],
                height=260
            )

        with e2:
            st.markdown("""
            <div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;'>
                Dataset Statistics
            </div>
            """, unsafe_allow_html=True)

            stats = [
                ("Total Reviews",         f"{total:,}"),
                ("Unique Conditions",     f"{n_conds:,}"),
                ("Unique Drugs",          f"{n_drugs:,}"),
                ("Avg. Review Length",    f"{avg_w} words"),
                ("Positive  ( >= 7 )",    f"{pos_n:,}   ({pos_n/total*100:.1f}%)"),
                ("Negative  ( <= 4 )",    f"{neg_n:,}   ({neg_n/total*100:.1f}%)"),
            ]

            for k, v in stats:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:.55rem .75rem;background:#ffffff;border-radius:3px;margin:.25rem 0;"
                    f"border:1.5px solid var(--rule);'>"
                    f"<span style='font-family:var(--font-mono);font-size:.78rem;color:var(--text-3);'>{k}</span>"
                    f"<span style='font-family:var(--font-mono);font-size:.78rem;color:var(--ink);font-weight:500;'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='font-family:var(--font-mono);font-size:.7rem;color:#6b7f96;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;'>
            Top 10 Conditions by Review Volume
        </div>
        """, unsafe_allow_html=True)

        # ✅ FIX 5: Ensure proper plotting
        st.bar_chart(
            top_conds.set_index("Condition")[["Reviews"]],
            height=300
        )

    else:
        st.info(f"Place '{DATASET_PATH}' in the same directory as app.py to view live EDA.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Preprocessing Pipeline"):
        st.markdown("""
```
Step 1   Lowercase normalisation
Step 2   URL removal
Step 3   HTML entity stripping  ( &amp;  &nbsp;  etc. )
Step 4   Special-character removal  ( retain a–z, 0–9, spaces )
Step 5   Whitespace normalisation
Step 6   TF-IDF Vectorisation
         · ngram_range   = (1, 2)
         · max_features  = 30,000
         · sublinear_tf  = True
         · stop_words    = english
Step 7   Logistic Regression
         · C             = 1.0
         · solver        = lbfgs
         · class_weight  = balanced
         · max_iter      = 1,000
```
        """)
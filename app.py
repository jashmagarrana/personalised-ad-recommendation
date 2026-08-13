import streamlit as st

st.set_page_config(
    page_title="Personalised Ad Recommendation System",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

import time
import pandas as pd
import numpy as np
import contextlib
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# Load artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    # Base models
    content_model = joblib.load("model_content_based_lr.pkl")
    collab_model = joblib.load("model_collaborative_rf.pkl")
    behaviour_model = joblib.load("model_behaviour_xgboost_tuned.pkl")
    gb_model = joblib.load("model_gradient_boosting.pkl")

    # Preprocessing artefacts
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    # streamlit_config.pkl contains feature-engineering information
    # required to construct a new input row.
    streamlit_config = joblib.load("streamlit_config.pkl")

    # hybrid_config.pkl contains the final hybrid configuration.
    hybrid_config = joblib.load("hybrid_config.pkl")

    # The final system is the Stacked Hybrid (Meta-Learner).
    meta_learner = joblib.load("model_hybrid_meta_learner.pkl")

    return (
        content_model,
        collab_model,
        behaviour_model,
        gb_model,
        scaler,
        label_encoders,
        streamlit_config,
        hybrid_config,
        meta_learner,
    )


(
    content_model,
    collab_model,
    behaviour_model,
    gb_model,
    scaler,
    label_encoders,
    config,
    hybrid_config,
    meta_learner,
) = load_artifacts()

FEATURE_COLUMNS = config["feature_columns"]
MEDIAN_BROWSING_ENCODED = config["median_browsing_encoded"]

# The final hybrid is the stacked meta-learner.
# The threshold is read from hybrid_config.pkl so the Streamlit app
# always uses the same threshold selected during model development.
_threshold_raw = (
    hybrid_config.get("threshold")
    if isinstance(hybrid_config, dict)
    else None
)

if _threshold_raw is None:
    _threshold_raw = config.get("best_threshold", 0.50)

try:
    THRESHOLD = float(_threshold_raw)
except (TypeError, ValueError):
    THRESHOLD = 0.50

if isinstance(hybrid_config, dict):
    FINAL_HYBRID_METHOD = str(
        hybrid_config.get(
            "method",
            hybrid_config.get("name", "stacked")
        )
    )
else:
    FINAL_HYBRID_METHOD = "stacked"

# These are retained only for displaying the optimised/legacy weights
# when they exist. They are NOT used for the final stacked prediction.
# Safely obtain legacy/display weights. The final system uses the
# saved stacked meta-learner, so these weights are display-only.
W_raw = {}
if isinstance(hybrid_config, dict):
    for _key in ("weights", "optimised_weights", "manual_weights"):
        _candidate = hybrid_config.get(_key)
        if isinstance(_candidate, dict):
            W_raw = _candidate
            break

def safe_weight(value, default=0.25):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

W = {
    "content": safe_weight(W_raw.get("content"), 0.25),
    "collab": safe_weight(W_raw.get("collab"), 0.25),
    "behaviour": safe_weight(W_raw.get("behaviour"), 0.25),
    "gb": safe_weight(W_raw.get("gb"), 0.25),
}

# Make sure weights add up to 1
weight_total = sum(W.values())

if weight_total > 0:
    W = {
        key: value / weight_total
        for key, value in W.items()
    }
else:
    W = {
        "content": 0.25,
        "collab": 0.25,
        "behaviour": 0.25,
        "gb": 0.25,
    }

print("Final hybrid weights:", W)

# Legacy/optional optimised weights
OPTIMISED_WEIGHTS = None

if isinstance(hybrid_config, dict):
    possible_weights = hybrid_config.get("optimised_weights")

    if isinstance(possible_weights, dict):
        _safe_opt = {}
        for _k in ("content", "collab", "behaviour", "gb"):
            _v = possible_weights.get(_k)
            if _v is not None:
                try:
                    _safe_opt[_k] = float(_v)
                except (TypeError, ValueError):
                    pass

        if len(_safe_opt) == 4:
            OPTIMISED_WEIGHTS = _safe_opt

CATEGORICAL_COLS = [
    "gender",
    "device_type",
    "ad_position",
    "browsing_history",
    "time_of_day",
]

# ============================================================
# Feature Engineering
# ============================================================
def build_feature_row(
    age,
    gender,
    device_type,
    ad_position,
    browsing_history,
    time_of_day,
):
    row = {}

    le_gender = label_encoders["gender"].transform([gender])[0]
    le_device = label_encoders["device_type"].transform([device_type])[0]
    le_position = label_encoders["ad_position"].transform([ad_position])[0]
    le_browsing = label_encoders["browsing_history"].transform([browsing_history])[0]
    le_time = label_encoders["time_of_day"].transform([time_of_day])[0]

    row["age"] = age

    row["age_group"] = (
        pd.cut(
            [age],
            bins=[0, 18, 25, 35, 50, 65, 100],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
        )
        .astype(float)[0]
    )

    if pd.isna(row["age_group"]):
        row["age_group"] = 0

    row["age_group"] = int(row["age_group"])

    row["high_engagement"] = int(
        le_browsing > MEDIAN_BROWSING_ENCODED
    )

    row["device_time_interaction"] = le_device * le_time
    row["gender_age_interaction"] = le_gender * age

    onehot_values = {
        "gender": gender,
        "device_type": device_type,
        "ad_position": ad_position,
        "browsing_history": browsing_history,
        "time_of_day": time_of_day,
    }

    for col in CATEGORICAL_COLS:
        for cls in label_encoders[col].classes_:
            row[f"{col}_{cls}"] = int(onehot_values[col] == cls)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ============================================================
# Prediction
# ============================================================
def predict(
    age,
    gender,
    device_type,
    ad_position,
    browsing_history,
    time_of_day,
):
    X_row = build_feature_row(
        age,
        gender,
        device_type,
        ad_position,
        browsing_history,
        time_of_day,
    )

    X_scaled = scaler.transform(X_row)

    p_content = content_model.predict_proba(X_scaled)[0][1]
    p_collab = collab_model.predict_proba(X_row)[0][1]
    p_behaviour = behaviour_model.predict_proba(X_row)[0][1]
    p_gb = gb_model.predict_proba(X_row)[0][1]

    # --------------------------------------------------------
    # FINAL HYBRID MODEL
    # --------------------------------------------------------
    # The selected final model is the Stacked Hybrid. It takes
    # the four base-model probabilities as meta-features and
    # learns how to combine them using the saved meta-learner.
    meta_features = np.array(
        [[p_content, p_collab, p_behaviour, p_gb]],
        dtype=float,
    )

    method_text = str(FINAL_HYBRID_METHOD).lower()

    if "stack" in method_text:
        hybrid = float(
            meta_learner.predict_proba(meta_features)[0, 1]
        )
    else:
        # Safe fallback for older hybrid_config files.
        # Prefer saved optimised weights if available.
        if OPTIMISED_WEIGHTS is None:
            raise ValueError(
                "The saved hybrid_config.pkl does not contain a stacked method "
                "or valid optimised weights."
            )

        hybrid = float(
            OPTIMISED_WEIGHTS["content"] * p_content
            + OPTIMISED_WEIGHTS["collab"] * p_collab
            + OPTIMISED_WEIGHTS["behaviour"] * p_behaviour
            + OPTIMISED_WEIGHTS["gb"] * p_gb
        )

    prediction = int(hybrid >= THRESHOLD)

    return (
        {
            "Content-Based (LR)": p_content,
            "Collaborative (RF)": p_collab,
            "Behaviour (XGBoost)": p_behaviour,
            "Gradient Boosting": p_gb,
            "Hybrid": hybrid,
        },
        prediction,
    )


# ============================================================
# Human-readable insight (display-only — does not touch
# model outputs, weights, or the decision itself)
# ============================================================
def generate_insight(probs, prediction, age, gender, device_type, ad_position, browsing_history, time_of_day):
    """
    Turns the raw probability outputs into a short, plain-English
    explanation for the person reading the dashboard. This only
    describes the existing results — it does not change how the
    hybrid score or the final decision are calculated.
    """
    model_only = {k: v for k, v in probs.items() if k != "Hybrid"}
    leading_model = max(model_only, key=model_only.get)
    leading_score = model_only[leading_model]

    model_reason = {
        "Content-Based (LR)": "the ad's content closely matching this user's interests",
        "Collaborative (RF)": "similar users showing comparable engagement patterns",
        "Behaviour (XGBoost)": "this user's browsing and interaction habits",
        "Gradient Boosting": "broader patterns learned across the full user base",
    }

    le_browsing = label_encoders["browsing_history"].transform([browsing_history])[0]
    engagement_level = "above-average" if le_browsing > MEDIAN_BROWSING_ENCODED else "typical"

    if prediction == 1:
        headline = (
            f"This {age}-year-old {device_type.lower()} user is likely to engage with the ad, "
            f"driven mainly by {model_reason.get(leading_model, 'the combined model signals')} "
            f"(the {leading_model} contributed the strongest signal at {leading_score:.0%})."
        )
    else:
        headline = (
            f"This {age}-year-old {device_type.lower()} user is unlikely to engage with the ad. "
            f"Even the strongest contributing signal — {model_reason.get(leading_model, 'the combined model signals')} "
            f"({leading_model}, {leading_score:.0%}) — wasn't enough to clear the {THRESHOLD:.0%} decision threshold."
        )

    detail = (
        f"Browsing engagement for this session is {engagement_level}, and the ad was evaluated "
        f"in the {ad_position.lower()} position during the {time_of_day.lower()}. "
        f"These factors are combined with age and device signals to reach the final hybrid score."
    )

    return headline, detail


# ============================================================
# THEME / DESIGN TOKENS — Navy, Indigo & White
# ============================================================
PRIMARY_C = "#4338CA"
SECONDARY_C = "#4F46E5"
LIGHT_C = "#EEF2FF"
ACCENT_C = "#818CF8"
HOVER_C = "#312E81"

CHARCOAL = "#0F172A"
CHARCOAL_SOFT = "#1E293B"
CHARCOAL_LIGHT = "#334155"

BG = "#CEE1F5"
TEXT_DARK = "#1E2430"
TEXT_SECONDARY = "#64748B"
BORDER = "#E2E8F0"

MODEL_COLOR_SEQUENCE = [PRIMARY_C, SECONDARY_C, ACCENT_C, "#A5B4FC", CHARCOAL_LIGHT]

# ============================================================
# ICON SET — Font Awesome Free (CDN), instead of hand-drawn SVGs
# ============================================================
FA_MAP = {
    "user": "fa-solid fa-user",
    "layers": "fa-solid fa-layer-group",
    "target": "fa-solid fa-bullseye",
    "users": "fa-solid fa-users",
    "pulse": "fa-solid fa-wave-square",
    "trend": "fa-solid fa-arrow-trend-up",
    "award": "fa-solid fa-award",
    "sliders": "fa-solid fa-sliders",
    "bar-chart": "fa-solid fa-chart-column",
    "pie-chart": "fa-solid fa-chart-pie",
    "settings": "fa-solid fa-gear",
    "list": "fa-solid fa-list-ul",
    "check-circle": "fa-solid fa-circle-check",
    "minus-circle": "fa-solid fa-circle-minus",
    "sparkle": "fa-solid fa-wand-magic-sparkles",
    "graduate": "fa-solid fa-graduation-cap",
}


def icon(name, size=18, color=None, stroke_width=None):
    """Renders a Font Awesome Free icon (loaded from the FA CDN).
    `stroke_width` is kept as an unused parameter for call-site
    compatibility with the previous SVG-icon signature."""
    c = color or TEXT_DARK
    fa_class = FA_MAP.get(name, "fa-solid fa-circle")
    return (
        f'<i class="{fa_class}" '
        f'style="font-size:{size}px; color:{c}; vertical-align:-2px; '
        f'display:inline-block;"></i>'
    )


@contextlib.contextmanager
def card_container(marker):
    """A real st.container() that can still be styled individually.
    Works on any Streamlit version (no dependency on the newer
    `key=` parameter): a hidden marker element is placed as the
    container's first child, and CSS `:has()` targets the parent
    wrapper by that marker to apply the card's look."""
    with st.container():
        st.markdown(f'<div class="card-marker {marker}"></div>', unsafe_allow_html=True)
        yield

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: {BG};
        }}

        .accent-serif {{
            font-family: 'Source Serif 4', Georgia, serif;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }}

        /* ---------- Header ---------- */
        .app-header {{
            padding: 2.2rem 2.5rem;
            border-radius: 14px;
            background: linear-gradient(100deg, {CHARCOAL} 0%, {CHARCOAL_SOFT} 100%);
            border-left: 4px solid {PRIMARY_C};
            box-shadow: 0 6px 20px rgba(0,0,0,0.18);
            margin-bottom: 1.8rem;
            animation: fadeIn 0.5s ease-in-out;
            position: relative;
        }}

        .app-header h1 {{
            color: #FFFFFF;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.3px;
        }}

        .app-header p {{
            color: rgba(255,255,255,0.72);
            font-size: 1rem;
            margin-top: 0.5rem;
            font-weight: 400;
            max-width: 720px;
        }}

        .ai-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(67,56,202,0.16);
            border: 1px solid {PRIMARY_C};
            color: {ACCENT_C};
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-top: 1.1rem;
        }}

        /* ---------- Generic Card ---------- */
        .card-marker {{ display: none; }}

        .card,
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_profile),
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_result_empty),
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_chart_bar),
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_chart_pie),
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_config) {{
            background: #FFFFFF;
            border: 1px solid {BORDER};
            border-top: 3px solid {PRIMARY_C};
            border-radius: 12px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            transition: box-shadow 0.25s ease, transform 0.25s ease;
            animation: fadeIn 0.5s ease-in-out;
            margin-bottom: 1.3rem;
        }}

        .card:hover,
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_profile):hover,
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_chart_bar):hover,
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_chart_pie):hover,
        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_config):hover {{
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }}

        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_result_positive) {{
            background: linear-gradient(160deg, {LIGHT_C} 0%, #FFFFFF 60%);
            border: 1px solid {SECONDARY_C};
            border-radius: 12px;
            padding: 2rem 1.7rem;
            box-shadow: 0 2px 12px rgba(67,56,202,0.08);
            margin-bottom: 1.3rem;
            animation: fadeIn 0.5s ease-in-out;
        }}

        div[data-testid="stVerticalBlock"]:has(> div.element-container > .card_result_negative) {{
            background: linear-gradient(160deg, #F5F5F5 0%, #FFFFFF 60%);
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 2rem 1.7rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            margin-bottom: 1.3rem;
            animation: fadeIn 0.5s ease-in-out;
        }}

        .card-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {TEXT_DARK};
            margin-bottom: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, {BORDER} 20%, {BORDER} 80%, transparent);
            margin: 2.2rem 0 1.6rem 0;
            border: none;
        }}

        .section-heading {{
            font-size: 1.3rem;
            font-weight: 800;
            color: {TEXT_DARK};
            margin-bottom: 0.3rem;
        }}

        .section-subheading {{
            font-size: 0.92rem;
            color: {TEXT_SECONDARY};
            margin-bottom: 1.3rem;
        }}

        /* ---------- KPI cards ---------- */
        .kpi-card {{
            background: linear-gradient(150deg, #FFFFFF 0%, {LIGHT_C} 220%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 3px 12px rgba(0,0,0,0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            height: 100%;
            opacity: 0;
            animation: popIn 0.5s ease forwards;
        }}

        .kpi-card:nth-of-type(1) {{ animation-delay: 0.05s; }}
        .kpi-card:nth-of-type(2) {{ animation-delay: 0.12s; }}
        .kpi-card:nth-of-type(3) {{ animation-delay: 0.19s; }}
        .kpi-card:nth-of-type(4) {{ animation-delay: 0.26s; }}
        .kpi-card:nth-of-type(5) {{ animation-delay: 0.33s; }}
        .kpi-card:nth-of-type(6) {{ animation-delay: 0.40s; }}

        @keyframes popIn {{
            from {{ opacity: 0; transform: translateY(10px) scale(0.97); }}
            to   {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}

        .kpi-card:hover {{
            transform: translateY(-5px) scale(1.015);
            box-shadow: 0 10px 22px rgba(67,56,202,0.16);
            border-color: {ACCENT_C};
        }}

        .kpi-card:hover .kpi-icon {{
            background: {SECONDARY_C};
            transform: rotate(-6deg) scale(1.08);
        }}

        .kpi-card:hover .kpi-icon i {{
            color: #FFFFFF !important;
        }}

        .kpi-icon {{
            margin-bottom: 0.5rem;
            width: 30px;
            height: 30px;
            border-radius: 9px;
            background: {LIGHT_C};
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.25s ease, transform 0.25s ease;
        }}

        .kpi-bar-track {{
            width: 100%;
            height: 5px;
            border-radius: 999px;
            background: {LIGHT_C};
            margin-top: 0.55rem;
            overflow: hidden;
        }}

        .kpi-bar-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {SECONDARY_C}, {ACCENT_C});
            width: 0%;
            animation: growBar 1.1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
            animation-delay: 0.35s;
        }}

        @keyframes growBar {{
            to {{ width: var(--fill, 0%); }}
        }}

        .kpi-value {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: {PRIMARY_C};
            margin: 0;
        }}

        .kpi-label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {TEXT_DARK};
            margin-top: 0.15rem;
        }}

        .kpi-desc {{
            font-size: 0.74rem;
            color: {TEXT_SECONDARY};
            margin-top: 0.2rem;
        }}

        /* ---------- Result cards ---------- */
        .result-icon {{
            width: 56px;
            height: 56px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.7rem auto;
        }}

        .result-icon.positive {{
            background: {LIGHT_C};
            box-shadow: 0 0 0 6px rgba(129,140,248,0.14);
        }}

        .result-icon.negative {{
            background: #EDEDED;
            box-shadow: 0 0 0 6px rgba(158,158,158,0.12);
        }}

        .result-title {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: {TEXT_DARK};
            margin-bottom: 0.3rem;
        }}

        .insight-box {{
            text-align: left;
            background: rgba(255,255,255,0.6);
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.9rem 1.05rem;
            margin-top: 1rem;
            font-size: 0.86rem;
            color: {TEXT_DARK};
            line-height: 1.55;
        }}

        .insight-box .insight-detail {{
            color: {TEXT_SECONDARY};
            margin-top: 0.5rem;
            font-size: 0.82rem;
        }}

        .result-sub {{
            font-size: 0.95rem;
            color: {TEXT_SECONDARY};
            margin-bottom: 1rem;
        }}

        .status-badge {{
            display: inline-block;
            padding: 5px 14px;
            border-radius: 4px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}

        .badge-positive {{
            background: {LIGHT_C};
            color: {PRIMARY_C};
            border: 1px solid {SECONDARY_C};
        }}

        .badge-negative {{
            background: #F0F0F0;
            color: #616161;
            border: 1px solid {BORDER};
        }}

        .result-icon.positive, .result-icon.negative {{
            animation: fadeIn 0.4s ease both;
        }}

        /* ---------- Buttons ---------- */
        div.stButton > button, div.stFormSubmitButton > button {{
            width: 100%;
            background: linear-gradient(120deg, {PRIMARY_C}, {SECONDARY_C});
            background-size: 220% 220%;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            box-shadow: 0 6px 16px rgba(67,56,202,0.28);
            transition: box-shadow 0.22s ease, transform 0.22s ease, background-position 0.5s ease;
        }}

        div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
            background-position: 100% 50%;
            box-shadow: 0 10px 22px rgba(49,46,129,0.35);
            transform: translateY(-2px);
        }}

        div.stButton > button:active, div.stFormSubmitButton > button:active {{
            transform: translateY(0) scale(0.98);
            box-shadow: 0 4px 10px rgba(49,46,129,0.3);
        }}

        /* ---------- Inputs ---------- */
        .stSlider label, .stSelectbox label {{
            font-weight: 600 !important;
            color: {TEXT_DARK} !important;
            font-size: 0.9rem !important;
        }}

        div[data-baseweb="select"] > div {{
            border-radius: 12px !important;
            border-color: {BORDER} !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        div[data-baseweb="select"] > div:hover {{
            border-color: {SECONDARY_C} !important;
            box-shadow: 0 0 0 3px rgba(129,140,248,0.14);
        }}

        .stSlider [data-baseweb="slider"] div[role="slider"] {{
            background-color: {PRIMARY_C} !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .stSlider [data-baseweb="slider"] div[role="slider"]:hover {{
            transform: scale(1.25);
            box-shadow: 0 0 0 6px rgba(67,56,202,0.16);
        }}

        .card-title i {{
            transition: transform 0.25s ease, color 0.25s ease;
        }}

        .card:hover .card-title i {{
            transform: scale(1.15) rotate(-4deg);
            color: {HOVER_C} !important;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {CHARCOAL} 0%, {CHARCOAL_SOFT} 100%);
            border-right: 1px solid {CHARCOAL_LIGHT};
        }}

        section[data-testid="stSidebar"] * {{
            color: rgba(255,255,255,0.85);
        }}

        .sidebar-title {{
            font-size: 1rem;
            font-weight: 700;
            color: {ACCENT_C} !important;
            margin-top: 1.1rem;
            margin-bottom: 0.5rem;
        }}

        .tech-pill {{
            display: inline-block;
            background: rgba(67,56,202,0.14);
            color: {ACCENT_C} !important;
            border: 1px solid rgba(129,140,248,0.4);
            border-radius: 5px;
            padding: 4px 11px;
            font-size: 0.74rem;
            font-weight: 600;
            margin: 3px 4px 3px 0;
        }}

        /* ---------- Footer ---------- */
        .app-footer {{
            text-align: center;
            padding: 1.6rem 0 0.4rem 0;
            color: {TEXT_SECONDARY};
            font-size: 0.85rem;
        }}

        .app-footer b {{
            color: {PRIMARY_C};
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 0.6rem 0 1rem 0; border-bottom: 1px solid {CHARCOAL_LIGHT}; margin-bottom: 0.4rem;">
            <div style="font-weight:700; font-size:1.05rem; color:#FFFFFF;">
                Ad Recommendation System
            </div>
            <div style="font-size:0.76rem; color:rgba(255,255,255,0.55);">Hybrid Machine Learning Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="sidebar-title">{icon("layers", 16, PRIMARY_C)}&nbsp; About this project</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.62); line-height:1.55;">
        This dashboard demonstrates a hybrid recommendation system built for
        an MSc Data Science dissertation. Rather than relying on a single
        algorithm, it combines four complementary models so that weaknesses
        in one approach are offset by the strengths of another.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="sidebar-title">{icon("settings", 16, PRIMARY_C)}&nbsp; Technology stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div>
            <span class="tech-pill">Python</span>
            <span class="tech-pill">XGBoost</span>
            <span class="tech-pill">Random Forest</span>
            <span class="tech-pill">Gradient Boosting</span>
            <span class="tech-pill">Logistic Regression</span>
            <span class="tech-pill">Scikit-Learn</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Plotly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="sidebar-title">{icon("pulse", 16, PRIMARY_C)}&nbsp; How the models fit together</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.62); line-height:1.65;">
        <b>Content-based</b> — Logistic Regression, matches ad content to user interests<br>
        <b>Collaborative</b> — Random Forest, learns from similar users<br>
        <b>Behavioural</b> — XGBoost, reads browsing and interaction patterns<br>
        <b>Ensemble</b> — Gradient Boosting, captures broader patterns<br>
        <b>Fusion</b> — a stacked meta-learner combines all four model outputs<br>
        <b>Decision threshold</b> — {THRESHOLD:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="sidebar-title">{icon("sparkle", 16, PRIMARY_C)}&nbsp; Why a hybrid approach</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.62); line-height:1.55;">
        No single model captures every signal well. Blending content,
        collaborative, and behavioural evidence gives a more balanced and
        reliable estimate of genuine user interest than any one model alone.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="app-header">
        <h1>Personalised Ad Recommendation System</h1>
        <p>A hybrid machine learning approach that blends content, collaborative, and
        behavioural signals to estimate how likely a user is to engage with an ad.</p>
        <div class="ai-badge">MSc Data Science &nbsp;•&nbsp; Dissertation Prototype</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MAIN LAYOUT — LEFT (INPUT) / RIGHT (RESULT)
# ============================================================
left_col, right_col = st.columns([1, 1.05], gap="large")

with left_col:
    with card_container("card_profile"):
        st.markdown(f'<div class="card-title">{icon("user", 19, PRIMARY_C)}&nbsp; User profile</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:{TEXT_SECONDARY}; margin-top:-0.6rem; margin-bottom:1.1rem;">
            Describe the visitor and the ad placement below, and the model will
            estimate their likelihood of engaging with it.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):

            age = st.slider("Age", 18, 64, 30)

            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox(
                    "Gender",
                    label_encoders["gender"].classes_,
                )
                ad_position = st.selectbox(
                    "Ad Position",
                    label_encoders["ad_position"].classes_,
                )
                browsing_history = st.selectbox(
                    "Browsing History",
                    label_encoders["browsing_history"].classes_,
                )

            with c2:
                device_type = st.selectbox(
                    "Device Type",
                    label_encoders["device_type"].classes_,
                )
                time_of_day = st.selectbox(
                    "Time of Day",
                    label_encoders["time_of_day"].classes_,
                )

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Predict engagement")

# Placeholder for result card so it renders in the right column
with right_col:
    result_placeholder = st.empty()
    if not submit:
        with result_placeholder.container():
            with card_container("card_result_empty"):
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:1.6rem 0.5rem;">
                    <div>{icon("sparkle", 28, PRIMARY_C)}</div>
                    <div style="font-weight:700; color:{TEXT_DARK}; font-size:1.05rem; margin-top:0.6rem;">
                        Results will appear here
                    </div>
                    <div style="font-size:0.88rem; color:{TEXT_SECONDARY}; margin-top:0.4rem; max-width:320px; margin-left:auto; margin-right:auto;">
                        Complete the user profile on the left and select
                        <b>Predict engagement</b> to see the hybrid model's
                    recommendation, confidence score, and reasoning.
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ============================================================
# PREDICTION EXECUTION (logic untouched)
# ============================================================
if submit:

    with st.spinner("Combining signals from all four models..."):
        time.sleep(0.6)
        probs, prediction = predict(
            age,
            gender,
            device_type,
            ad_position,
            browsing_history,
            time_of_day,
        )

    hybrid_pct = probs["Hybrid"] * 100
    insight_headline, insight_detail = generate_insight(
        probs, prediction, age, gender, device_type, ad_position, browsing_history, time_of_day
    )

    with right_col:
        with result_placeholder.container():
            result_key = "card_result_positive" if prediction == 1 else "card_result_negative"
            with card_container(result_key):
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <div class="result-icon positive">{icon("check-circle", 26, PRIMARY_C, 2)}</div>
                            <div class="result-title">Likely to click</div>
                            <div class="result-sub">The model predicts strong ad engagement for this user profile.</div>
                            <span class="status-badge badge-positive">RECOMMENDED</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <div class="result-icon negative">{icon("minus-circle", 26, "#757575", 2)}</div>
                            <div class="result-title">Unlikely to click</div>
                            <div class="result-sub">The model predicts low ad engagement for this user profile.</div>
                            <span class="status-badge badge-negative">NOT RECOMMENDED</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Probability gauge (Plotly)
                gauge_color = PRIMARY_C if prediction == 1 else "#9E9E9E"
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=hybrid_pct,
                        number={"suffix": "%", "font": {"size": 34, "color": TEXT_DARK}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": TEXT_SECONDARY},
                            "bar": {"color": gauge_color, "thickness": 0.28},
                            "bgcolor": "white",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, THRESHOLD * 100], "color": "#F1F1F1"},
                                {"range": [THRESHOLD * 100, 100], "color": LIGHT_C},
                            ],
                            "threshold": {
                                "line": {"color": HOVER_C, "width": 3},
                                "thickness": 0.75,
                                "value": THRESHOLD * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=220,
                    margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": TEXT_DARK, "family": "Inter"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                st.markdown(
                    f"""
                        <div style="font-size:0.85rem; color:{TEXT_SECONDARY}; text-align:center; margin-top:-0.6rem;">
                            Hybrid probability: <b style="color:{TEXT_DARK}">{hybrid_pct:.1f}%</b>
                            &nbsp;|&nbsp; Threshold: <b style="color:{TEXT_DARK}">{THRESHOLD*100:.0f}%</b>
                        </div>
                        <div class="insight-box">
                            {insight_headline}
                            <div class="insight-detail">{insight_detail}</div>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ============================================================
    # KPI METRIC CARDS
    # ============================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{icon("bar-chart", 20, PRIMARY_C)}&nbsp; Model comparison &amp; metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">Individual model contributions to the final hybrid decision</div>',
        unsafe_allow_html=True,
    )

    kpi_items = [
        ("layers", "Content-Based", probs["Content-Based (LR)"], "Logistic Regression score"),
        ("users", "Collaborative", probs["Collaborative (RF)"], "Random Forest score"),
        ("pulse", "Behaviour", probs["Behaviour (XGBoost)"], "XGBoost engagement score"),
        ("trend", "Gradient Boosting", probs["Gradient Boosting"], "Ensemble booster score"),
        ("award", "Hybrid Score", probs["Hybrid"], "Weighted fusion score"),
        ("sliders", "Threshold", THRESHOLD, "Decision cutoff"),
    ]

    kpi_cols = st.columns(6)
    for col, (icon_name, label, value, desc) in zip(kpi_cols, kpi_items):
        with col:
            fill_pct = max(0, min(100, value * 100))
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon(icon_name, 16, PRIMARY_C)}</div>
                    <p class="kpi-value">{value:.3f}</p>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-desc">{desc}</div>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill" style="--fill:{fill_pct:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    # ============================================================
    # CHARTS
    # ============================================================
    chart_col1, chart_col2 = st.columns(2, gap="large")

    model_df = pd.DataFrame(
        {
            "Model": list(probs.keys()),
            "Probability": list(probs.values()),
        }
    )

    with chart_col1:
        with card_container("card_chart_bar"):
            st.markdown(f'<div class="card-title">{icon("bar-chart", 18, PRIMARY_C)}&nbsp; Model probability comparison</div>', unsafe_allow_html=True)

            fig_bar = px.bar(
                model_df,
                x="Probability",
                y="Model",
                orientation="h",
                color="Model",
                color_discrete_sequence=MODEL_COLOR_SEQUENCE,
                text=model_df["Probability"].apply(lambda v: f"{v:.3f}"),
            )
            fig_bar.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>",
            )
            fig_bar.update_layout(
                showlegend=False,
                height=340,
                margin=dict(l=10, r=30, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT_DARK, "family": "Inter"},
                xaxis=dict(range=[0, 1], gridcolor="#F0F0F0", title=""),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        with card_container("card_chart_pie"):
            st.markdown(f'<div class="card-title">{icon("pie-chart", 18, PRIMARY_C)}&nbsp; Probability distribution</div>', unsafe_allow_html=True)

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=model_df["Model"],
                        values=model_df["Probability"],
                        hole=0.58,
                        marker=dict(colors=MODEL_COLOR_SEQUENCE, line=dict(color="#FFFFFF", width=2)),
                        textinfo="label+percent",
                        hovertemplate="<b>%{label}</b><br>%{value:.3f}<extra></extra>",
                    )
                ]
            )
            fig_donut.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT_DARK, "family": "Inter"},
                showlegend=False,
                annotations=[
                    dict(
                        text=f"{hybrid_pct:.0f}%<br><span style='font-size:11px;color:{TEXT_SECONDARY}'>Hybrid</span>",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20, color=PRIMARY_C),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ============================================================
    # WEIGHTS / CONFIGURATION CARD
    # ============================================================
    with card_container("card_config"):
        st.markdown(f'<div class="card-title">{icon("settings", 18, PRIMARY_C)}&nbsp; Ensemble configuration</div>', unsafe_allow_html=True)

        w1, w2, w3, w4, w5 = st.columns(5)
        weight_items = [
            ("Content-Based", W["content"]),
            ("Collaborative", W["collab"]),
            ("Behaviour", W["behaviour"]),
            ("Gradient Boosting", W["gb"]),
            ("Threshold", THRESHOLD),
        ]
        for col, (label, val) in zip([w1, w2, w3, w4, w5], weight_items):
            with col:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <div style="font-size:1.3rem; font-weight:800; color:{PRIMARY_C};">{val:.2f}</div>
                        <div style="font-size:0.78rem; color:{TEXT_SECONDARY}; font-weight:600;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Raw data table (kept for transparency, styled)
    with st.expander("View raw probability data"):
        st.dataframe(
            model_df.style.format({"Probability": "{:.3f}"}),
            use_container_width=True,
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-footer">
        <b>Personalised Ad Recommendation System</b> — Hybrid Machine Learning &amp; User Behaviour Analysis<br>
        MSc Data Science Dissertation &nbsp;•&nbsp; University Project &nbsp;•&nbsp; Student Demonstration
    </div>
    """,
    unsafe_allow_html=True,
)

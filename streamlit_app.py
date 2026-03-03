import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Stress Analysis Dashboard 🧠",
    layout="wide",
    page_icon="🧠",
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0d0d14"
CARD     = "#13131e"
BORDER   = "#1f1f30"
ACCENT1  = "#c084fc"   # purple
ACCENT2  = "#38bdf8"   # sky
ACCENT3  = "#fb7185"   # rose
ACCENT4  = "#34d399"   # emerald
TEXT     = "#e2e8f0"
SUBTEXT  = "#94a3b8"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

* {{ font-family: 'DM Sans', sans-serif; }}
h1,h2,h3,h4 {{ font-family: 'Space Grotesk', sans-serif; }}

[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stSidebar"]          {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
h1,h2,h3,h4,h5,h6,p,label,span {{ color: {TEXT} !important; }}

.metric-card {{
    background: {CARD};
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid {BORDER};
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {ACCENT1}, {ACCENT2});
}}
.metric-card h4 {{ color: {SUBTEXT} !important; font-size: 0.8rem; font-weight: 500; margin: 0 0 6px 0; letter-spacing: 0.05em; text-transform: uppercase; }}
.metric-card h2 {{ color: {TEXT} !important; font-size: 1.9rem; font-weight: 700; margin: 0; }}

.stTabs [data-baseweb="tab"] {{ color: {SUBTEXT}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT1} !important; border-bottom: 2px solid {ACCENT1}; }}

.section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT} !important;
    padding: 10px 0 6px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 16px;
}}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f'<h2 style="color:{ACCENT1}!important;margin-bottom:4px;">🧠 Stress Analysis</h2>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{SUBTEXT}!important;font-size:0.85rem;margin-top:0;">Student Mental Health Explorer</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

dataset_choice = st.sidebar.radio(
    "Dataset",
    ["📋 Survey Dataset (843 students)", "📊 Numeric Dataset (1100 students)"],
)
use_survey = dataset_choice.startswith("📋")

page = st.sidebar.selectbox(
    "Navigate",
    ["Introduction 📘", "Visualization 📊", "Insights 🔍", "Prediction 🤖"]
)

# ── Load data ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_survey():
    path = os.path.join(BASE_DIR, "Stress_Dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_numeric():
    path = os.path.join(BASE_DIR, "StressLevelDataset.csv")
    return pd.read_csv(path)

survey_df = load_survey()
numeric_df = load_numeric()

if use_survey:
    df = survey_df.copy()
    target_col = "Which type of stress do you primarily experience?"
    short_target = "Stress Type"
    df["Stress_Short"] = df[target_col].str.split(" - ").str[0]
else:
    df = numeric_df.copy()
    target_col = "stress_level"
    short_target = "Stress Level"
    df["Stress_Short"] = df[target_col].map({0: "Low", 1: "Medium", 2: "High"})

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols     = df.select_dtypes(exclude=np.number).columns.tolist()

# ── Plot helpers ──────────────────────────────────────────────────────────────
def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CARD)
    ax.set_facecolor(BG)
    ax.tick_params(colors=SUBTEXT)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    return fig, ax

STRESS_COLORS = {
    "Eustress": ACCENT4,
    "Distress": ACCENT3,
    "No Stress": ACCENT2,
    "Low":    ACCENT4,
    "Medium": ACCENT1,
    "High":   ACCENT3,
}

PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#fbbf24", "#f472b6"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Introduction
# ─────────────────────────────────────────────────────────────────────────────
if page == "Introduction 📘":
    st.title("🧠 Student Stress Analysis Dashboard")
    ds_label = "Survey" if use_survey else "Numeric"
    st.markdown(f'<p style="color:{SUBTEXT}!important;">Exploring mental health and stress patterns across students — <b>{ds_label} Dataset</b></p>', unsafe_allow_html=True)
    st.write("")

    # KPI Cards
    if use_survey:
        c1, c2, c3, c4 = st.columns(4)
        stress_dist = df["Stress_Short"].value_counts()
        with c1:
            st.markdown(f'<div class="metric-card"><h4>👥 Respondents</h4><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h4>📋 Features</h4><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
        with c3:
            avg_age = df["Age"].mean() if "Age" in df.columns else "N/A"
            st.markdown(f'<div class="metric-card"><h4>🎂 Avg Age</h4><h2>{avg_age:.1f}</h2></div>', unsafe_allow_html=True)
        with c4:
            pct_eustress = stress_dist.get("Eustress", 0) / len(df) * 100
            st.markdown(f'<div class="metric-card"><h4>✨ Eustress %</h4><h2>{pct_eustress:.1f}%</h2></div>', unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h4>👥 Students</h4><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h4>📋 Features</h4><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h4>😰 Avg Anxiety</h4><h2>{df["anxiety_level"].mean():.1f}</h2></div>', unsafe_allow_html=True)
        with c4:
            pct_high = (df["stress_level"] == 2).mean() * 100
            st.markdown(f'<div class="metric-card"><h4>🔴 High Stress %</h4><h2>{pct_high:.1f}%</h2></div>', unsafe_allow_html=True)

    st.write("")

    # Stress distribution donut
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown('<div class="section-header">Stress Type Distribution</div>', unsafe_allow_html=True)
        counts = df["Stress_Short"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=CARD)
        ax.set_facecolor(CARD)
        wedge_colors = [STRESS_COLORS.get(k, ACCENT1) for k in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90, pctdistance=0.8,
            wedgeprops=dict(width=0.55, edgecolor=CARD, linewidth=2)
        )
        for t in texts: t.set_color(TEXT)
        for at in autotexts: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
        st.pyplot(fig)

    with col_right:
        st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
        rows = st.slider("Rows", 5, 20, 8)
        st.dataframe(df.head(rows), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values — dataset is clean!")
        else:
            st.dataframe(missing[missing > 0].reset_index().rename(columns={"index": "Column", 0: "Missing"}))
    with col_b:
        st.markdown('<div class="section-header">Column Types</div>', unsafe_allow_html=True)
        types = df.dtypes.reset_index()
        types.columns = ["Column", "Type"]
        st.dataframe(types, use_container_width=True)

    if st.button("📊 Show Summary Statistics"):
        st.dataframe(df.describe(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Visualization
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Visualization 📊":
    st.title("📊 Data Visualization")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution 📉", "Scatter Plot 🔵", "Category Breakdown 🗂️", "Correlation Heatmap 🔥"
    ])

    with tab1:
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = styled_fig(10, 4)
            counts, bins, patches = ax.hist(df[col].dropna(), bins=30, color=ACCENT1, edgecolor=BG, alpha=0.85)
            # gradient color effect
            norm = plt.Normalize(counts.min(), counts.max())
            for patch, val in zip(patches, counts):
                c = plt.cm.cool(norm(val))
                patch.set_facecolor(c)
            ax.set_xlabel(col, color=SUBTEXT)
            ax.set_ylabel("Count", color=SUBTEXT)
            ax.set_title(f"Distribution of {col}", color=TEXT, fontsize=13, fontweight="bold")
            ax.axvline(df[col].mean(), color=ACCENT3, linestyle="--", lw=1.5, label=f"Mean: {df[col].mean():.2f}")
            ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available.")

    with tab2:
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", numeric_cols, index=0)
            y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            color_by_stress = st.checkbox("Color by Stress Level", value=True)
            fig, ax = styled_fig(10, 5)
            sample = df.sample(min(800, len(df)), random_state=42)
            if color_by_stress and "Stress_Short" in df.columns:
                stress_cats = sample["Stress_Short"].unique()
                for i, cat in enumerate(stress_cats):
                    grp = sample[sample["Stress_Short"] == cat]
                    ax.scatter(grp[x_col], grp[y_col], alpha=0.5, s=18,
                               color=STRESS_COLORS.get(cat, PALETTE[i % len(PALETTE)]), label=cat)
                ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
            else:
                ax.scatter(sample[x_col], sample[y_col], color=ACCENT2, alpha=0.4, s=15)
            ax.set_xlabel(x_col, color=SUBTEXT)
            ax.set_ylabel(y_col, color=SUBTEXT)
            ax.set_title(f"{x_col} vs {y_col}", color=TEXT, fontsize=13, fontweight="bold")
            st.pyplot(fig)

    with tab3:
        if cat_cols:
            cat = st.selectbox("Category column", [c for c in cat_cols if c != "Stress_Short"] or cat_cols)
            if numeric_cols:
                metric = st.selectbox("Numeric metric", numeric_cols)
                agg = df.groupby(cat)[metric].mean().sort_values(ascending=False).head(12)
                fig, ax = styled_fig(9, 4)
                bars = ax.bar(range(len(agg)), agg.values, color=PALETTE[:len(agg)], edgecolor=BG, width=0.6)
                ax.set_xticks(range(len(agg)))
                ax.set_xticklabels([str(x)[:20] for x in agg.index], rotation=30, ha="right", color=SUBTEXT, fontsize=8)
                ax.set_ylabel(f"Avg {metric}", color=SUBTEXT)
                ax.set_title(f"Avg {metric} by {cat}", color=TEXT, fontsize=13, fontweight="bold")
                st.pyplot(fig)
            else:
                # show value counts
                vc = df[cat].value_counts().head(10)
                fig, ax = styled_fig(9, 4)
                ax.bar(range(len(vc)), vc.values, color=PALETTE[:len(vc)], edgecolor=BG)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(x)[:20] for x in vc.index], rotation=30, ha="right", color=SUBTEXT)
                ax.set_title(f"Value Counts: {cat}", color=TEXT, fontsize=13, fontweight="bold")
                st.pyplot(fig)
        else:
            st.info("No categorical columns to display.")

    with tab4:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=CARD)
            ax.set_facecolor(BG)
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark"),
                ax=ax, mask=mask,
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 8, "color": TEXT},
            )
            ax.set_title("Correlation Matrix", color=TEXT, fontsize=14, fontweight="bold")
            plt.xticks(color=SUBTEXT, rotation=45, ha="right", fontsize=8)
            plt.yticks(color=SUBTEXT, fontsize=8)
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for correlation.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Insights 🔍":
    st.title("🔍 Key Insights")

    if use_survey:
        # ── Survey dataset insights ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Stress Type by Gender</div>', unsafe_allow_html=True)
            cross = pd.crosstab(df["Gender"], df["Stress_Short"], normalize="index") * 100
            fig, ax = styled_fig(6, 4)
            bottom = np.zeros(len(cross))
            gender_labels = {0: "Female", 1: "Male"}
            x = np.arange(len(cross))
            for i, col_name in enumerate(cross.columns):
                color = STRESS_COLORS.get(col_name, PALETTE[i])
                ax.bar(x, cross[col_name].values, bottom=bottom, color=color, label=col_name, edgecolor=BG)
                bottom += cross[col_name].values
            ax.set_xticks(x)
            ax.set_xticklabels([gender_labels.get(g, str(g)) for g in cross.index], color=SUBTEXT)
            ax.set_ylabel("% of Group", color=SUBTEXT)
            ax.set_title("Stress Type by Gender (%)", color=TEXT, fontweight="bold")
            ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER, fontsize=8)
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="section-header">Age Distribution by Stress Type</div>', unsafe_allow_html=True)
            fig, ax = styled_fig(6, 4)
            for i, stype in enumerate(df["Stress_Short"].unique()):
                vals = df[df["Stress_Short"] == stype]["Age"].dropna()
                ax.hist(vals, bins=15, alpha=0.65, color=STRESS_COLORS.get(stype, PALETTE[i]), label=stype, edgecolor=BG)
            ax.set_xlabel("Age", color=SUBTEXT)
            ax.set_ylabel("Count", color=SUBTEXT)
            ax.set_title("Age Distribution by Stress Type", color=TEXT, fontweight="bold")
            ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
            st.pyplot(fig)

        # ── Key stressors radar ──
        st.markdown('<div class="section-header">Top Stressor Questions (Mean Score per Stress Type)</div>', unsafe_allow_html=True)
        survey_numeric = df.select_dtypes(include=np.number).columns.tolist()
        survey_numeric = [c for c in survey_numeric if c not in ["Age", "Gender"]]
        short_names = {c: c[:35] for c in survey_numeric}

        if survey_numeric:
            top_n = st.slider("Number of top stressors to compare", 4, 12, 8)
            # find features with highest variance across stress types
            group_means = df.groupby("Stress_Short")[survey_numeric].mean()
            variance = group_means.var(axis=0).sort_values(ascending=False)
            top_features = variance.head(top_n).index.tolist()

            fig, ax = styled_fig(11, 5)
            x = np.arange(len(top_features))
            width = 0.25
            for i, stype in enumerate(group_means.index):
                vals = group_means.loc[stype, top_features].values
                offset = (i - 1) * width
                ax.bar(x + offset, vals, width=width,
                       color=STRESS_COLORS.get(stype, PALETTE[i]),
                       label=stype, edgecolor=BG, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([f[:30] for f in top_features], rotation=35, ha="right", color=SUBTEXT, fontsize=8)
            ax.set_ylabel("Mean Score", color=SUBTEXT)
            ax.set_title("Key Stressor Comparisons Across Stress Types", color=TEXT, fontweight="bold")
            ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
            st.pyplot(fig)

    else:
        # ── Numeric dataset insights ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Anxiety Level Distribution by Stress</div>', unsafe_allow_html=True)
            fig, ax = styled_fig(6, 4)
            for level in [0, 1, 2]:
                label = {0: "Low", 1: "Medium", 2: "High"}[level]
                vals = df[df["stress_level"] == level]["anxiety_level"]
                ax.hist(vals, bins=20, alpha=0.65, color=STRESS_COLORS[label], label=label, edgecolor=BG)
            ax.set_xlabel("Anxiety Level", color=SUBTEXT)
            ax.set_ylabel("Count", color=SUBTEXT)
            ax.set_title("Anxiety Level by Stress Group", color=TEXT, fontweight="bold")
            ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="section-header">Self-Esteem vs Depression</div>', unsafe_allow_html=True)
            fig, ax = styled_fig(6, 4)
            sc = ax.scatter(df["self_esteem"], df["depression"],
                            c=df["stress_level"], cmap="cool",
                            alpha=0.5, s=18)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Stress Level", color=SUBTEXT)
            cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)
            ax.set_xlabel("Self-Esteem", color=SUBTEXT)
            ax.set_ylabel("Depression", color=SUBTEXT)
            ax.set_title("Self-Esteem vs Depression (colored by Stress)", color=TEXT, fontweight="bold")
            st.pyplot(fig)

        # Feature averages by stress level
        st.markdown('<div class="section-header">Average Feature Values by Stress Level</div>', unsafe_allow_html=True)
        feat_cols = [c for c in numeric_cols if c != "stress_level"]
        group_avg = df.groupby("stress_level")[feat_cols].mean()

        fig, axes = plt.subplots(4, 5, figsize=(16, 10), facecolor=CARD)
        axes = axes.flatten()
        for i, col_name in enumerate(feat_cols[:20]):
            ax = axes[i]
            ax.set_facecolor(BG)
            vals = group_avg[col_name].values
            bars = ax.bar(["Low", "Med", "High"], vals,
                          color=[ACCENT4, ACCENT1, ACCENT3], edgecolor=BG, width=0.6)
            ax.set_title(col_name[:18], color=TEXT, fontsize=8, fontweight="bold")
            ax.tick_params(colors=SUBTEXT, labelsize=7)
            for sp in ax.spines.values(): sp.set_color(BORDER)
        for j in range(len(feat_cols), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)

        # Sleep quality vs stress
        st.markdown('<div class="section-header">Sleep Quality × Anxiety × Stress</div>', unsafe_allow_html=True)
        fig, ax = styled_fig(10, 4)
        bins = pd.cut(df["sleep_quality"], bins=5, precision=1)
        sleep_stress = df.groupby(bins, observed=True)["anxiety_level"].mean().reset_index()
        ax.bar(sleep_stress["sleep_quality"].astype(str), sleep_stress["anxiety_level"],
               color=ACCENT1, edgecolor=BG, alpha=0.85)
        ax.set_xlabel("Sleep Quality (binned)", color=SUBTEXT)
        ax.set_ylabel("Avg Anxiety Level", color=SUBTEXT)
        ax.set_title("Lower Sleep Quality → Higher Anxiety", color=TEXT, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Prediction 🤖":
    st.title("🤖 Prediction with Linear Regression")
    st.markdown(
        f'<p style="color:{SUBTEXT}!important;">This page trains a <b>Linear Regression</b> model (Scikit-Learn) to predict a key numeric variable and support decision-making.</p>',
        unsafe_allow_html=True
    )

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    # ----- Choose dataset + target -----
    if use_survey:
        st.info("Survey dataset selected. Linear Regression needs a NUMERIC target.")
        # Pick a numeric target from the survey dataset
        survey_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # remove obvious IDs if present (safe)
        survey_numeric_cols = [c for c in survey_numeric_cols if c.lower() not in ["id", "index"]]

        if len(survey_numeric_cols) == 0:
            st.error("No numeric columns found in the survey dataset. Please switch to the Numeric Dataset.")
            st.stop()

        target = st.selectbox("Select a numeric target to predict (Y)", survey_numeric_cols)

    else:
        # Numeric dataset has many numeric columns; let user choose target
        numeric_targets = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_targets) == 0:
            st.error("No numeric columns found. Cannot run Linear Regression.")
            st.stop()

        # Good default: predict anxiety from other lifestyle indicators (societal value)
        default_target = "anxiety_level" if "anxiety_level" in numeric_targets else numeric_targets[0]
        target = st.selectbox("Select a numeric target to predict (Y)", numeric_targets, index=numeric_targets.index(default_target))

    # ----- Build X / y -----
    data = df.copy()

    # Drop helper columns
    data = data.drop(columns=["Stress_Short"], errors="ignore")

    # Remove rows missing the target
    data = data.dropna(subset=[target])

    X_all = data.drop(columns=[target], errors="ignore")
    y_all = data[target].astype(float)

    # Feature selection UI
    all_features = X_all.columns.tolist()
    features_sel = st.sidebar.multiselect("Select Features (X)", all_features, default=all_features)

    if len(features_sel) == 0:
        st.warning("Please select at least 1 feature.")
        st.stop()

    X = X_all[features_sel].copy()
    y = y_all.copy()

    # Train/test split
    test_size = st.sidebar.slider("Test Split %", 10, 40, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size / 100,
        random_state=42
    )

    # ----- Preprocessing (leakage-safe) -----
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop"
    )

    # ----- Linear Regression model -----
    model = LinearRegression()

    reg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # ----- Train + Evaluate -----
    if st.button("🚀 Train Linear Regression"):
        with st.spinner("Training Linear Regression..."):
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)

            r2 = metrics.r2_score(y_test, preds)
            mae = metrics.mean_absolute_error(y_test, preds)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))

        st.success("✅ Linear Regression trained successfully!")

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"{mae:.3f}")
        c3.metric("RMSE", f"{rmse:.3f}")

        # ----- Show prediction vs actual scatter -----
        st.markdown('<div class="section-header">Predicted vs Actual</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=CARD)
        ax.set_facecolor(BG)
        ax.scatter(y_test, preds, alpha=0.55, s=20)
        ax.set_xlabel("Actual", color=SUBTEXT)
        ax.set_ylabel("Predicted", color=SUBTEXT)
        ax.set_title(f"{target}: Actual vs Predicted", color=TEXT, fontsize=12, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=SUBTEXT)
        st.pyplot(fig)

        # ----- Driving variables (coefficients) -----
        st.markdown('<div class="section-header">Driving Variables (Model Coefficients)</div>', unsafe_allow_html=True)

        # Get feature names after preprocessing
        preprocess_fitted = reg.named_steps["preprocess"]

        feature_names = []
        feature_names.extend(num_features)

        if len(cat_features) > 0:
            ohe = preprocess_fitted.named_transformers_["cat"].named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(cat_features).tolist()
            feature_names.extend(ohe_names)

        coefs = reg.named_steps["model"].coef_
        coef_s = pd.Series(coefs, index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)

        top_k = st.slider("Show top coefficients", 5, min(25, len(coef_s)), 12)
        st.dataframe(coef_s.head(top_k).to_frame("Coefficient"), use_container_width=True)

        # Optional coefficient bar chart
        fig2, ax2 = plt.subplots(figsize=(9, max(4, top_k * 0.35)), facecolor=CARD)
        ax2.set_facecolor(BG)
        top = coef_s.head(top_k).sort_values()
        ax2.barh(top.index, top.values, edgecolor=BG)
        ax2.set_xlabel("Coefficient value", color=SUBTEXT)
        ax2.set_title("Top Driving Variables", color=TEXT, fontsize=12, fontweight="bold")
        ax2.tick_params(colors=SUBTEXT)
        for sp in ax2.spines.values():
            sp.set_color(BORDER)
        st.pyplot(fig2)

        # ----- Single prediction (user input) -----
        st.markdown('<div class="section-header">Make a New Prediction</div>', unsafe_allow_html=True)
        st.caption("Enter feature values to estimate the target using the trained model.")

        input_data = {}
        col1, col2 = st.columns(2)
        for i, col_name in enumerate(features_sel):
            if col_name in num_features:
                # numeric input
                default_val = float(X[col_name].median()) if pd.api.types.is_numeric_dtype(X[col_name]) else 0.0
                widget_col = col1 if i % 2 == 0 else col2
                input_data[col_name] = widget_col.number_input(col_name, value=default_val)
            else:
                # categorical input
                opts = sorted(X[col_name].dropna().astype(str).unique().tolist())
                widget_col = col1 if i % 2 == 0 else col2
                input_data[col_name] = widget_col.selectbox(col_name, options=opts if opts else ["Unknown"])

        if st.button("🎯 Predict"):
            input_df = pd.DataFrame([input_data])
            pred_value = reg.predict(input_df)[0]
            st.success(f"Predicted **{target}**: **{pred_value:.3f}**")

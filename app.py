import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Rainfall Prediction & PCA Analysis",
    layout="wide",
    page_icon="🌧️"
)

# -------------------------------------------------------------
# CUSTOM CSS FOR BEAUTIFUL PROFESSIONAL UI
# -------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fb;
    }
    .title-text {
        font-size: 38px !important;
        font-weight: 800 !important;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle-text {
        font-size: 20px !important;
        text-align: center;
        color: #34495e;
    }
    .explain-text {
        color: #2f3542;
        font-size: 16px;
        padding: 8px 15px;
        background: #f1f2f6;
        border-left: 4px solid #2ecc71;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        font-size: 16px;
        background-color: #dfe4ea;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MODEL PIPELINE
# -------------------------------------------------------------
pipe = joblib.load("pipeline.joblib")
scaler, pca, model, features = pipe["scaler"], pipe["pca"], pipe["model"], pipe["features"]

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown('<p class="title-text">🌧️ Rainfall Prediction & Pattern Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Using PCA Feature Reduction + Random Forest Regression</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🌩️ Navigation")
page = st.sidebar.radio("Go to", ["📁 Upload Data", "📊 Overview", "🧠 PCA Analysis", "🎯 Predictions", "🌦️ Clustering"])

uploaded_file = st.sidebar.file_uploader("Upload CSV (with JAN–DEC)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not all(f in df.columns for f in features):
        st.error("CSV must contain columns: " + ", ".join(features))
        st.stop()

    X_scaled = scaler.transform(df[features])
    X_pca = pca.transform(X_scaled)
    df["Predicted Annual Rainfall"] = model.predict(X_pca)

else:
    st.info("Upload a dataset from the sidebar to begin.")
    st.stop()

# -------------------------------------------------------------
# PAGE 1: DATASET PREVIEW
# -------------------------------------------------------------
if page == "📁 Upload Data":
    st.header("📁 Uploaded Dataset Preview")

    st.write(df.head())

    st.markdown("""
    <div class="explain-text">
    This table previews the uploaded dataset.  
    Ensure all 12 monthly rainfall columns (JAN–DEC) are present.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 2: OVERVIEW
# -------------------------------------------------------------
elif page == "📊 Overview":

    st.header("📊 Dataset Overview & Monthly Distributions")

    fig_box = px.box(df[features], title="Monthly Rainfall Distribution", color_discrete_sequence=["#2ecc71"])
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    <div class="explain-text">
    Box plots help visualize rainfall ranges, seasonal variations,  
    and months with extreme or highly variable rainfall.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 3: PCA ANALYSIS
# -------------------------------------------------------------
elif page == "🧠 PCA Analysis":

    st.header("🧠 PCA Analysis & Feature Reduction")

    ev = pca.explained_variance_ratio_
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(ev))],
        y=ev,
        marker_color="#27ae60"
    ))
    fig_scree.update_layout(
        title="Explained Variance by Principal Components",
        xaxis_title="Principal Component",
        yaxis_title="Variance Ratio"
    )

    st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown("""
    <div class="explain-text">
    PCA reduces dimensionality from 12 monthly values to a smaller set of  
    principal components while preserving most of the important information.
    </div>
    """, unsafe_allow_html=True)

    # LOADINGS
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(ev))]
    )

    st.subheader("📌 PCA Loadings Matrix")
    st.write(loadings.style.background_gradient(cmap="Greens").format("{:.3f}"))

    st.markdown("""
    <div class="explain-text">
    Loadings explain how each month contributes to each principal component.  
    Higher values indicate stronger influence.
    </div>
    """, unsafe_allow_html=True)

    # Month importance
    pc_importances = model.feature_importances_
    month_importances = loadings.values.dot(pc_importances)

    fig_imp = px.bar(
        x=features,
        y=month_importances,
        title="🔥 Month Importance (Combined PCA + RF Importance)",
        color=month_importances,
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# -------------------------------------------------------------
# PAGE 4: PREDICTIONS
# -------------------------------------------------------------
elif page == "🎯 Predictions":

    st.header("🎯 Annual Rainfall Prediction")

    st.dataframe(df[["Predicted Annual Rainfall"]])

    fig_pred = px.line(
        df,
        y="Predicted Annual Rainfall",
        title="Prediction Trend",
        markers=True,
        color_discrete_sequence=["#2ecc71"]
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("""
    <div class="explain-text">
    This graph visualizes predicted annual rainfall for each  
    row of your uploaded dataset.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📝 Custom Prediction Input")

    cols = st.columns(4)
    inputs = []
    for i, month in enumerate(features):
        inputs.append(cols[i % 4].number_input(month, value=10.0, min_value=0.0))

    if st.button("Predict Now"):
        arr = np.array(inputs).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        arr_pca = pca.transform(arr_scaled)
        pred = model.predict(arr_pca)[0]

        st.success(f"🌧️ Predicted Annual Rainfall: **{pred:.2f} mm**")


# -------------------------------------------------------------
# PAGE 4: PATTERN CLUSTERING
# -------------------------------------------------------------
elif page == "🌦️ Clustering":

    st.header("🌦️ Rainfall Pattern Clustering")

    k = st.slider("Select number of clusters", 2, 6, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca[:, :2])

    df["Cluster"] = clusters

    fig_cluster = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=df["Cluster"].astype(str),
        title="PC1 vs PC2 - Rainfall Pattern Clusters",
        labels={"x": "PC1", "y": "PC2"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("""
    <div class="explain-text">
    Clustering identifies rainfall pattern groups like  
    **Wet**, **Normal**, and **Dry** years using PCA features.
    </div>
    """, unsafe_allow_html=True)

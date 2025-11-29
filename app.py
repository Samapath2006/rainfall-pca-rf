import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Rainfall PCA & RF Prediction",
    layout="wide",
    page_icon="🌧️"
)

# Load trained pipeline
pipe = joblib.load("pipeline.joblib")
scaler, pca, model, features = pipe["scaler"], pipe["pca"], pipe["model"], pipe["features"]

# UI Header
st.title("🌧️ Rainfall Prediction & Pattern Analysis")
st.markdown("""
This application demonstrates how **Principal Component Analysis (PCA)** and  
**Random Forest Regression** can be used together to analyze rainfall patterns  
and predict annual rainfall based on monthly data.

Upload a file containing **monthly rainfall features (JAN–DEC)** to begin.
""")

# File uploader
uploaded_file = st.file_uploader("Upload rainfall dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check for required features
    if not all(f in df.columns for f in features):
        st.error("CSV must contain the following columns: " + ", ".join(features))
        st.stop()

    # Prepare feature vectors
    X_scaled = scaler.transform(df[features])
    X_pca = pca.transform(X_scaled)
    predictions = model.predict(X_pca)

    df["Predicted_Annual_Rainfall"] = predictions

    # Tabs for structured UI
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🧠 PCA Analysis", "🎯 Prediction", "🌦️ Pattern Clustering"]
    )

    # -------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------
    with tab1:
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        st.subheader("🔎 Monthly Statistics")
        fig_stats = px.box(df[features], title="Distribution of Monthly Rainfall (Uploaded Data)")
        st.plotly_chart(fig_stats, use_container_width=True)

        st.markdown("""
        **Explanation:**  
        This chart shows the rainfall spread in each month.  
        High variance months typically influence PCA more strongly.
        """)

    # -------------------------------------------------------
    # TAB 2: PCA ANALYSIS
    # -------------------------------------------------------
    with tab2:
        st.subheader("🧠 PCA Explained Variance (Scree Plot)")

        ev = pca.explained_variance_ratio_
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev))
        fig_scree.update_layout(
            title="Variance Explained by Each Principal Component",
            xaxis_title="Principal Components",
            yaxis_title="Variance Ratio"
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        st.markdown("""
        **Explanation:**  
        PCA reduces 12 monthly features into fewer components.  
        Each bar shows how much information (variance) each component captures.
        """)

        st.subheader("📌 PCA Loadings (How Each Month Contributes to Each PC)")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=[f"PC{i+1}" for i in range(len(ev))]
        )
        st.dataframe(loadings.style.format("{:.3f}"))

        st.markdown("""
        **Explanation:**  
        The loadings table shows how strongly each month influences each principal component.  
        For example, if JUN–AUG have high loadings on PC1, that component represents **monsoon strength**.
        """)

        st.subheader("🔥 Month Importance (Mapped from PCA + RF)")
        pc_importances = model.feature_importances_
        month_importances = loadings.values.dot(pc_importances)

        fig_imp = px.bar(
            x=features,
            y=month_importances,
            title="Feature Importance (Which Months Affect Prediction)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        **Explanation:**  
        This graph shows which months had the biggest influence on the predicted annual rainfall.  
        Importance is derived by combining PCA loadings + Random Forest importance.
        """)

    # -------------------------------------------------------
    # TAB 3: PREDICTION
    # -------------------------------------------------------
    with tab3:
        st.subheader("🎯 Predicted Annual Rainfall Results")
        st.dataframe(df[["Predicted_Annual_Rainfall"]])

        st.subheader("📈 Prediction Trend")
        fig_pred = px.line(
            df,
            y="Predicted_Annual_Rainfall",
            title="Predicted Annual Rainfall Trend"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("""
        **Explanation:**  
        This plot shows the predicted annual rainfall for each row of your uploaded dataset.  
        Useful when forecasting multiple years/scenarios at once.
        """)

        st.subheader("📝 Custom Prediction")
        st.markdown("Enter monthly rainfall values to generate a prediction:")

        cols = st.columns(4)
        manual_input = []
        for i, month in enumerate(features):
            manual_input.append(
                cols[i % 4].number_input(month, min_value=0.0, value=10.0)
            )

        if st.button("Predict Annual Rainfall"):
            arr = np.array(manual_input).reshape(1, -1)
            arr_scaled = scaler.transform(arr)
            arr_pca = pca.transform(arr_scaled)
            manual_pred = model.predict(arr_pca)[0]
            st.success(f"Predicted Annual Rainfall: **{manual_pred:.2f} mm**")

    # -------------------------------------------------------
    # TAB 4: CLUSTERING
    # -------------------------------------------------------
    with tab4:
        st.subheader("🌦️ Pattern Clustering (Rainfall Regimes)")

        k = st.slider("Select number of clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_pca[:, :2])

        df["Cluster"] = clusters
        fig_cluster = px.scatter(
            df,
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df["Cluster"].astype(str),
            title="PC1 vs PC2 Cluster Visualization",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("""
        **Explanation:**  
        Clustering groups years with similar rainfall patterns.  
        These clusters represent rainfall regimes like:
        - Dry years  
        - Normal years  
        - Wet years  
        """)

else:
    st.info("Upload a CSV file with JAN–DEC columns to begin.")

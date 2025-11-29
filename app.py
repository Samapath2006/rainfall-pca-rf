import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

# Load pipeline
pipe = joblib.load("pipeline.joblib")
scaler, pca, model = pipe['scaler'], pipe['pca'], pipe['model']
features = pipe['features']

st.title("🌧️ Rainfall Prediction & Pattern Analysis (PCA + Random Forest)")

uploaded_file = st.file_uploader("Upload CSV with monthly rainfall (JAN–DEC)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not all(f in df.columns for f in features):
        st.error("CSV must contain columns: " + ", ".join(features))
    else:
        # Transform input
        X_scaled = scaler.transform(df[features])
        X_pca = pca.transform(X_scaled)
        predictions = model.predict(X_pca)

        df["Predicted_Annual_Rainfall"] = predictions

        st.subheader("Predicted Annual Rainfall")
        st.write(df)

        # PCA Explained Variance Plot
        ev = pca.explained_variance_ratio_
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev))
        fig.update_layout(title="PCA Explained Variance")
        st.plotly_chart(fig, use_container_width=True)

        # Month Importance (Feature importance mapped back)
        loadings = pd.DataFrame(pca.components_.T, index=features,
                                columns=[f"PC{i+1}" for i in range(len(ev))])

        pc_importances = model.feature_importances_
        month_importances = loadings.values.dot(pc_importances)

        fig2 = px.bar(x=features, y=month_importances, title="Month Importance")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Upload a dataset to start.")

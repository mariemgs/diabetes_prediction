# diabetes_app.py (interface seulement, sans réentraînement du modèle)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Diabetes Prediction Analysis", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def explore_data(df):
    st.header("\U0001F4CA Data Exploration")
    st.dataframe(df.head(), use_container_width=True)
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("Zero Values Analysis")
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zeros = df[zero_cols].eq(0).sum()
    st.write(zeros)

def visualize_data(df):
    st.header("\U0001F4C8 Data Visualization")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')

    fig = make_subplots(rows=3, cols=3, subplot_titles=numeric_cols, specs=[[{"secondary_y": False}]*3]*3)
    for i, col in enumerate(numeric_cols):
        row = i // 3 + 1
        col_pos = i % 3 + 1
        fig.add_trace(go.Histogram(x=df[df['Outcome']==0][col], name=f"Non-Diabetic {col}", opacity=0.5), row=row, col=col_pos)
        fig.add_trace(go.Histogram(x=df[df['Outcome']==1][col], name=f"Diabetic {col}", opacity=0.5), row=row, col=col_pos)
    fig.update_layout(barmode='overlay', height=800)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig_corr = px.imshow(df.corr(), text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

def preprocess_data(df):
    st.header("\U0001F527 Data Preprocessing")
    df_processed = df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    method = st.radio("Zero values strategy", ["Keep", "Replace with mean", "Replace with median"])
    if method == "Replace with mean":
        for col in zero_cols:
            df_processed[col] = df_processed[col].replace(0, df_processed[col].mean())
    elif method == "Replace with median":
        for col in zero_cols:
            df_processed[col] = df_processed[col].replace(0, df_processed[col].median())

    scale_method = st.selectbox("Scaling", ["None", "StandardScaler", "RobustScaler"])
    X = df_processed.drop("Outcome", axis=1)
    y = df_processed["Outcome"]
    scaler = None
    if scale_method == "StandardScaler":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scale_method == "RobustScaler":
        scaler = RobustScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    st.dataframe(X.describe())
    return X, y, scaler

def unsupervised_learning(X):
    st.header("\U0001F3AF Unsupervised Learning")
    pca_2d = PCA(n_components=2).fit_transform(X)
    method = st.selectbox("Clustering method", ["KMeans", "DBSCAN", "Agglomerative"])

    if method == "KMeans":
        k = st.slider("Number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(pca_2d)
    elif method == "DBSCAN":
        eps = st.slider("EPS", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(pca_2d)
    else:
        k = st.slider("Number of clusters", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(pca_2d)

    fig = px.scatter(x=pca_2d[:,0], y=pca_2d[:,1], color=labels.astype(str), title="Clusters")
    st.plotly_chart(fig, use_container_width=True)

def prediction_interface(model, scaler):
    st.header("\U0001F52E Diabetes Prediction")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
            bp = st.number_input("Blood Pressure", 0.0, 200.0, 80.0)
            skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
        with col2:
            insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 0, 120, 30)
        submitted = st.form_submit_button("Predict")
        if submitted:
            X_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            if scaler:
                X_input = scaler.transform(X_input)
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]
            st.success("Non-Diabetic" if pred==0 else "Diabetic")
            st.progress(int(prob[1]*100))
            st.write(f"Probability of being Diabetic: {prob[1]*100:.2f}%")

def main():
    st.title("🩺 Diabetes App (Model Already Trained)")
    df = load_data()
    model, scaler = load_model()

    menu = st.sidebar.radio("Navigation", [
        "🏠 Home", "📊 Explore", "📈 Visualize", "🔧 Preprocess", "🎯 Clustering", "🔮 Predict"
    ])

    if menu == "🏠 Home":
       st.markdown("""
        ## Welcome to the Diabetes Prediction Analysis Dashboard! 🎉

        This comprehensive tool allows you to:

        ### 📊 **Data Exploration**
        - Explore the diabetes dataset with interactive visualizations
        - Analyze statistical summaries and distributions
        - Identify patterns and correlations

        ### 🔧 **Data Preprocessing**
        - Handle missing and zero values
        - Apply feature scaling techniques
        - Prepare data for machine learning

        ### 🤖 **Supervised Learning**
        - Train multiple classification models
        - Compare model performance
        - Evaluate with comprehensive metrics

        ### 🎯 **Unsupervised Learning**
        - Perform clustering analysis
        - Apply PCA for dimensionality reduction
        - Discover hidden patterns

        ### 🔮 **Make Predictions**
        - Use trained models for real-time predictions
        - Interactive input interface
        - Probability estimates and visualizations

        ---

        ### 📋 **Dataset Information**
        - **Source**: Pima Indians Diabetes Database
        - **Samples**: 768 patients
        - **Features**: 8 medical indicators
        - **Target**: Diabetes outcome (0/1)

        **Get started by selecting a section from the sidebar!** 👈
        """)
    elif menu == "📊 Explore":
        explore_data(df)
    elif menu == "📈 Visualize":
        visualize_data(df)
    elif menu == "🔧 Preprocess":
        X, y, sc = preprocess_data(df)
        st.session_state.X_pre = X
        st.session_state.y_pre = y
    elif menu == "🎯 Clustering":
        if 'X_pre' in st.session_state:
            unsupervised_learning(st.session_state.X_pre)
        else:
            st.warning("Please preprocess data first")
    elif menu == "🔮 Predict":
        prediction_interface(model, scaler)

if __name__ == '__main__':
    main()

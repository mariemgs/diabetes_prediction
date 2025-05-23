import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, roc_auc_score)

# Clustering imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Prediction Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the diabetes dataset"""
    try:
        # Try to load from URL first
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        data = pd.read_csv(url)
        return data
    except:
        # If URL fails, show instructions for manual upload
        st.error("Could not load data from URL. Please upload the CSV file manually.")
        return None

def explore_data(df):
    """Comprehensive data exploration"""
    st.markdown('<div class="section-header">📊 Data Exploration</div>', unsafe_allow_html=True)
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Diabetic Cases", df['Outcome'].sum())
    with col4:
        st.metric("Non-Diabetic Cases", len(df) - df['Outcome'].sum())
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values analysis
    st.subheader("❓ Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"⚠️ Found {missing_data.sum()} missing values")
        st.dataframe(missing_data[missing_data > 0])
    
    # Zero values analysis (potential missing values)
    st.subheader("🔍 Zero Values Analysis")
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = df[zero_cols].eq(0).sum()
    zero_df = pd.DataFrame({
        'Column': zero_counts.index,
        'Zero Count': zero_counts.values,
        'Percentage': (zero_counts.values / len(df) * 100).round(2)
    })
    st.dataframe(zero_df, use_container_width=True)

def visualize_data(df):
    """Create comprehensive visualizations"""
    st.markdown('<div class="section-header">📈 Data Visualization</div>', unsafe_allow_html=True)
    
    # Distribution plots
    st.subheader("📊 Feature Distributions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')
    
    # Create subplots for distributions
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=numeric_cols,
        specs=[[{"secondary_y": False}]*3]*3
    )
    
    for i, col in enumerate(numeric_cols):
        row = i // 3 + 1
        col_pos = i % 3 + 1
        
        # Histogram for diabetic vs non-diabetic
        diabetic = df[df['Outcome'] == 1][col]
        non_diabetic = df[df['Outcome'] == 0][col]
        
        fig.add_trace(
            go.Histogram(x=non_diabetic, name=f'Non-Diabetic ({col})', 
                        opacity=0.7, nbinsx=20, showlegend=(i==0)),
            row=row, col=col_pos
        )
        fig.add_trace(
            go.Histogram(x=diabetic, name=f'Diabetic ({col})', 
                        opacity=0.7, nbinsx=20, showlegend=(i==0)),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=800, title_text="Feature Distributions by Outcome")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Box plots
    st.subheader("📦 Box Plots by Outcome")
    selected_features = st.multiselect(
        "Select features for box plots:",
        numeric_cols,
        default=numeric_cols[:4]
    )
    
    if selected_features:
        fig_box = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4]
        )
        
        for i, feature in enumerate(selected_features[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig_box.add_trace(
                go.Box(y=df[df['Outcome']==0][feature], name='Non-Diabetic', 
                      showlegend=(i==0)),
                row=row, col=col
            )
            fig_box.add_trace(
                go.Box(y=df[df['Outcome']==1][feature], name='Diabetic', 
                      showlegend=(i==0)),
                row=row, col=col
            )
        
        fig_box.update_layout(height=600, title_text="Box Plots by Outcome")
        st.plotly_chart(fig_box, use_container_width=True)

def preprocess_data(df):
    """Data preprocessing and cleaning"""
    st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Handle zero values (potential missing values)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    preprocessing_option = st.radio(
        "Choose preprocessing strategy for zero values:",
        ["Keep zeros", "Replace with median", "Replace with mean"]
    )
    
    if preprocessing_option == "Replace with median":
        for col in zero_cols:
            df_processed[col] = df_processed[col].replace(0, df_processed[col].median())
        st.success("✅ Zero values replaced with median")
    elif preprocessing_option == "Replace with mean":
        for col in zero_cols:
            df_processed[col] = df_processed[col].replace(0, df_processed[col].mean())
        st.success("✅ Zero values replaced with mean")
    else:
        st.info("ℹ️ Keeping original zero values")
    
    # Feature scaling option
    scaling_option = st.selectbox(
        "Choose scaling method:",
        ["None", "StandardScaler", "RobustScaler"]
    )
    
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    scaler = None
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.success("✅ Applied Standard Scaling")
    elif scaling_option == "RobustScaler":
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.success("✅ Applied Robust Scaling")
    else:
        X_scaled = X
        st.info("ℹ️ No scaling applied")
    
    # Show preprocessing results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Preprocessing")
        st.dataframe(X.describe())
    with col2:
        st.subheader("After Preprocessing")
        st.dataframe(X_scaled.describe())
    
    return X_scaled, y, scaler

def supervised_learning(X, y):
    """Supervised learning with multiple models"""
    st.markdown('<div class="section-header">🤖 Supervised Learning</div>', unsafe_allow_html=True)
    
    # Train-test split
    test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    st.success(f"✅ Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Model selection
    models = {
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state)
    }
    
    selected_models = st.multiselect(
        "Select models to train:",
        list(models.keys()),
        default=['Random Forest', 'Logistic Regression', 'SVM']
    )
    
    if not selected_models:
        st.warning("Please select at least one model")
        return None
    
    results = {}
    
    # Train and evaluate models
    st.subheader("🏋️ Model Training Results")
    
    for model_name in selected_models:
        with st.expander(f"📊 {model_name} Results"):
            model = models[model_name]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            
            if auc:
                st.metric("AUC-ROC", f"{auc:.3f}")
            
            st.info(f"Cross-validation: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title=f"Confusion Matrix - {model_name}",
                              labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model comparison
    if len(results) > 1:
        st.subheader("🏆 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'Precision': [results[m]['precision'] for m in results.keys()],
            'Recall': [results[m]['recall'] for m in results.keys()],
            'F1-Score': [results[m]['f1'] for m in results.keys()],
            'AUC-ROC': [results[m]['auc'] if results[m]['auc'] else 0 for m in results.keys()],
            'CV Score': [results[m]['cv_mean'] for m in results.keys()]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # ROC Curves
        fig_roc = go.Figure()
        for model_name in results.keys():
            if results[model_name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{model_name} (AUC = {results[model_name]['auc']:.3f})",
                    mode='lines'
                ))
        
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    return results, X_test, y_test

def unsupervised_learning(X):
    """Unsupervised learning - Clustering"""
    st.markdown('<div class="section-header">🎯 Unsupervised Learning - Clustering</div>', unsafe_allow_html=True)
    
    # PCA for visualization
    st.subheader("📉 Principal Component Analysis (PCA)")
    
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    
    # Explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var_ratio)
    
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Bar(x=list(range(1, len(explained_var_ratio)+1)), 
                            y=explained_var_ratio, name='Individual'))
    fig_pca.add_trace(go.Scatter(x=list(range(1, len(cumsum_var)+1)), 
                               y=cumsum_var, name='Cumulative', mode='lines+markers'))
    fig_pca.update_layout(title='PCA Explained Variance Ratio', 
                         xaxis_title='Principal Component', 
                         yaxis_title='Explained Variance Ratio')
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Use first 2 components for clustering
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    st.info(f"First 2 components explain {cumsum_var[1]:.1%} of variance")
    
    # Clustering algorithms
    st.subheader("🎯 Clustering Analysis")
    
    clustering_method = st.selectbox(
        "Select clustering algorithm:",
        ["K-Means", "DBSCAN", "Agglomerative Clustering"]
    )
    
    if clustering_method == "K-Means":
        # K-means with elbow method
        st.subheader("📈 Elbow Method for Optimal K")
        
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca_2d)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_pca_2d, kmeans.labels_))
        
        # Plot elbow curve
        fig_elbow = make_subplots(rows=1, cols=2, 
                                 subplot_titles=['Elbow Curve', 'Silhouette Score'])
        
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, 
                                     mode='lines+markers', name='Inertia'), 
                          row=1, col=1)
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, 
                                     mode='lines+markers', name='Silhouette Score'), 
                          row=1, col=2)
        
        fig_elbow.update_layout(height=400, title_text="K-Means Optimization")
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Select optimal K
        optimal_k = st.slider("Select number of clusters (K):", 2, 10, 3)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca_2d)
        
        silhouette_avg = silhouette_score(X_pca_2d, cluster_labels)
        st.success(f"✅ K-Means with K={optimal_k}, Silhouette Score: {silhouette_avg:.3f}")
        
    elif clustering_method == "DBSCAN":
        # DBSCAN parameters
        eps = st.slider("Epsilon (eps):", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Minimum samples:", 2, 20, 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_pca_2d)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        st.success(f"✅ DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        if n_clusters > 1:
            # Calculate silhouette score (excluding noise points)
            if n_noise < len(cluster_labels):
                mask = cluster_labels != -1
                if len(set(cluster_labels[mask])) > 1:
                    silhouette_avg = silhouette_score(X_pca_2d[mask], cluster_labels[mask])
                    st.info(f"Silhouette Score: {silhouette_avg:.3f}")
    
    else:  # Agglomerative Clustering
        n_clusters = st.slider("Number of clusters:", 2, 10, 3)
        
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clustering.fit_predict(X_pca_2d)
        
        silhouette_avg = silhouette_score(X_pca_2d, cluster_labels)
        st.success(f"✅ Agglomerative Clustering with {n_clusters} clusters, Silhouette Score: {silhouette_avg:.3f}")
    
    # Visualize clusters
    st.subheader("🎨 Cluster Visualization")
    
    fig_clusters = px.scatter(
        x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], 
        color=cluster_labels.astype(str),
        title=f'{clustering_method} Clustering Results',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    return cluster_labels, X_pca_2d

def prediction_interface(results, scaler, feature_names):
    """Interactive prediction interface"""
    st.markdown('<div class="section-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if not results:
        st.warning("Please train models first in the Supervised Learning section")
        return
    
    # Select model for prediction
    model_name = st.selectbox(
        "Select model for prediction:",
        list(results.keys())
    )
    
    model = results[model_name]['model']
    
    st.subheader(f"🎯 Predict with {model_name}")
    
    # Input form
    with st.form("prediction_form"):
        st.write("Enter patient information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
        
        with col2:
            insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
        submitted = st.form_submit_button("🔍 Predict")
        
        if submitted:
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, diabetes_pedigree, age]])
            input_df = pd.DataFrame(input_data, columns=feature_names)
            
            # Apply same scaling if used during training
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
                input_df_scaled = pd.DataFrame(input_scaled, columns=feature_names)
            else:
                input_df_scaled = input_df
            
            # Make prediction
            prediction = model.predict(input_df_scaled)[0]
            prediction_proba = model.predict_proba(input_df_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: The model predicts DIABETIC")
            else:
                st.success("✅ **LOW RISK**: The model predicts NON-DIABETIC")
            
            if prediction_proba is not None:
                prob_non_diabetic = prediction_proba[0] * 100
                prob_diabetic = prediction_proba[1] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Non-Diabetic Probability", f"{prob_non_diabetic:.1f}%")
                with col2:
                    st.metric("Diabetic Probability", f"{prob_diabetic:.1f}%")
                
                # Probability visualization
                fig_prob = go.Figure(data=[
                    go.Bar(x=['Non-Diabetic', 'Diabetic'], 
                          y=[prob_non_diabetic, prob_diabetic],
                          marker_color=['green', 'red'])
                ])
                fig_prob.update_layout(title='Prediction Probabilities', 
                                     yaxis_title='Probability (%)')
                st.plotly_chart(fig_prob, use_container_width=True)
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.dataframe(input_df.T, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.markdown('<div class="main-header">🩺 Diabetes Prediction Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    sections = [
        "🏠 Home",
        "📊 Data Exploration", 
        "🔧 Data Preprocessing",
        "🤖 Supervised Learning",
        "🎯 Unsupervised Learning",
        "🔮 Make Predictions"
    ]
    
    selected_section = st.sidebar.radio("Go to:", sections)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Please check your internet connection or upload the diabetes.csv file manually.")
        uploaded_file = st.file_uploader("Upload diabetes.csv file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully!")
    
    if df is not None:
        # Initialize session state
        if 'preprocessing_done' not in st.session_state:
            st.session_state.preprocessing_done = False
        if 'X_processed' not in st.session_state:
            st.session_state.X_processed = None
        if 'y' not in st.session_state:
            st.session_state.y = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'supervised_results' not in st.session_state:
            st.session_state.supervised_results = None
        
        # Section routing
        if selected_section == "🏠 Home":
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
            
            # Quick dataset overview
            if df is not None:
                st.subheader("📈 Quick Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Patients", len(df))
                with col2:
                    st.metric("Features", len(df.columns) - 1)
                with col3:
                    st.metric("Diabetic Cases", df['Outcome'].sum())
                with col4:
                    st.metric("Diabetes Rate", f"{df['Outcome'].mean():.1%}")
        
        elif selected_section == "📊 Data Exploration":
            explore_data(df)
            visualize_data(df)
        
        elif selected_section == "🔧 Data Preprocessing":
            X_processed, y, scaler = preprocess_data(df)
            st.session_state.X_processed = X_processed
            st.session_state.y = y
            st.session_state.scaler = scaler
            st.session_state.preprocessing_done = True
        
        elif selected_section == "🤖 Supervised Learning":
            if not st.session_state.preprocessing_done:
                st.warning("⚠️ Please complete data preprocessing first!")
                st.info("👈 Go to 'Data Preprocessing' section in the sidebar")
            else:
                results, X_test, y_test = supervised_learning(
                    st.session_state.X_processed, 
                    st.session_state.y
                )
                if results:
                    st.session_state.supervised_results = results
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
        
        elif selected_section == "🎯 Unsupervised Learning":
            if not st.session_state.preprocessing_done:
                st.warning("⚠️ Please complete data preprocessing first!")
                st.info("👈 Go to 'Data Preprocessing' section in the sidebar")
            else:
                cluster_labels, X_pca = unsupervised_learning(st.session_state.X_processed)
                st.session_state.cluster_labels = cluster_labels
                st.session_state.X_pca = X_pca
        
        elif selected_section == "🔮 Make Predictions":
            if st.session_state.supervised_results is None:
                st.warning("⚠️ Please train models first!")
                st.info("👈 Go to 'Supervised Learning' section in the sidebar")
            else:
                feature_names = st.session_state.X_processed.columns.tolist()
                prediction_interface(
                    st.session_state.supervised_results,
                    st.session_state.scaler,
                    feature_names
                )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 About This Project
    
    This diabetes prediction tool uses machine learning to analyze patient data and predict diabetes risk.
    
    **Features:**
    - Multiple ML algorithms
    - Interactive visualizations
    - Real-time predictions
    - Comprehensive analysis
    
    **Models Used:**
    - Random Forest
    - Logistic Regression
    - Support Vector Machine
    - K-Nearest Neighbors
    - Decision Tree
    
    **Clustering Methods:**
    - K-Means
    - DBSCAN
    - Agglomerative Clustering
    """)

if __name__ == "__main__":
    main()
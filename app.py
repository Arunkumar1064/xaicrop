import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
import dice_ml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="XAI Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #2E7D32, #66BB6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌾 Explainable AI Crop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning + SHAP + LIME + Counterfactual Explanations</p>', unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('crop_remmendation_dataset.csv')
    return df



# Load trained models and artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Load artifacts from models directory
        if not os.path.exists("models/best_model.pkl"):
            raise FileNotFoundError("Model artifacts not found")
            
        le = joblib.load("models/label_encoder.pkl")
        feature_cols = joblib.load("models/feature_cols.pkl")
        results = joblib.load("models/results.pkl")
        trained_models = joblib.load("models/trained_models.pkl")
        data_splits = joblib.load("models/data_splits.pkl")
        X_train, X_test, y_train, y_test = data_splits
        
        return results, trained_models, X_train, X_test, y_train, y_test, le, feature_cols
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.info("Please running 'python train_model.py' first to generate models.")
        st.stop()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["📊 Data Overview", "🤖 Model Comparison", "🧠 SHAP Explanation", 
     "🔍 LIME Explanation", "🔄 Counterfactual Analysis", "🧬 Hybrid XAI Insights",
     "📈 Advanced Analytics", "🎯 Make Prediction", "🔮 Future Scope"]
)

# Load data
df = load_data()
# Load pre-trained artifacts
results, trained_models, X_train, X_test, y_train, y_test, le, feature_cols = load_artifacts()

# Find best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = trained_models[best_model_name]

# ============= PAGE: DATA OVERVIEW =============
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(feature_cols))
    with col3:
        st.metric("Crop Classes", len(df['Recommended_Crop'].unique()))
    with col4:
        st.metric("Best Model", best_model_name)
    
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 Feature Statistics")
    st.dataframe(df[feature_cols].describe(), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Crop Distribution")
        crop_counts = df['Recommended_Crop'].value_counts()
        fig = px.bar(x=crop_counts.index, y=crop_counts.values, 
                    labels={'x': 'Crop', 'y': 'Count'},
                    title="Crop Distribution",
                    color=crop_counts.values,
                    color_continuous_scale='Greens')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Feature Correlation Heatmap")
        corr = df[feature_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Feature Correlation Matrix")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    selected_features = st.multiselect("Select features to visualize", feature_cols, default=feature_cols[:4])
    
    if selected_features:
        fig = make_subplots(rows=2, cols=2, subplot_titles=selected_features[:4])
        for idx, feat in enumerate(selected_features[:4]):
            row = idx // 2 + 1
            col = idx % 2 + 1
            fig.add_trace(
                go.Histogram(x=df[feat], name=feat, nbinsx=30),
                row=row, col=col
            )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============= PAGE: MODEL COMPARISON =============
elif page == "🤖 Model Comparison":
    st.header("🤖 Model Performance Comparison")
    
    # Performance metrics table
    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results],
        'CV Mean': [results[m]['cv_mean'] for m in results],
        'CV Std': [results[m]['cv_std'] for m in results]
    }).sort_values('Accuracy', ascending=False)
    
    st.dataframe(metrics_df.style.background_gradient(cmap='Greens', subset=['Accuracy', 'F1-Score']), 
                use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Accuracy Comparison")
        fig = px.bar(metrics_df, x='Model', y='Accuracy',
                    color='Accuracy', color_continuous_scale='Greens',
                    title="Model Accuracy Comparison")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Metrics Radar Chart")
        fig = go.Figure()
        for model in metrics_df['Model'][:3]:  # Top 3 models
            model_data = metrics_df[metrics_df['Model'] == model].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[model_data['Accuracy'], model_data['Precision'], 
                   model_data['Recall'], model_data['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                name=model
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix - Best Model")
    selected_model = st.selectbox("Select Model", list(results.keys()), 
                                  index=list(results.keys()).index(best_model_name))
    
    cm = results[selected_model]['confusion_matrix']
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                   labels=dict(x="Predicted", y="Actual"),
                   color_continuous_scale='Blues',
                   title=f"Confusion Matrix - {selected_model}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============= PAGE: SHAP EXPLANATION =============
elif page == "🧠 SHAP Explanation":
    st.header("🧠 SHAP (SHapley Additive exPlanations)")
    
    st.info("SHAP values show how each feature contributes to the model's predictions")
    
    # Select model for SHAP
    # Filter out non-tree models for SHAP (TreeExplainer)
    shap_models = [m for m in trained_models.keys() if m != 'Logistic Regression']
    model_for_shap = st.selectbox("Select Model for SHAP Analysis", 
                                  shap_models,
                                  index=0)
    
    model = trained_models[model_for_shap]
    
    # Create SHAP explainer
    # Create SHAP explainer
    with st.spinner("Calculating SHAP values..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
        except Exception as e:
            st.error(f"SHAP explanation failed for {model_for_shap}: {str(e)}")
            st.info("Gradient Boosting Classifier is currently not supported for multi-class SHAP analysis in this version. Please select XGBoost, Random Forest, or LightGBM.")
            st.stop()
    
    # SHAP Summary Plot
    st.subheader("📊 SHAP Summary Plot")
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_test[:100], feature_names=feature_cols, show=False)
    else:
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, show=False)
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()
    
    # SHAP Feature Importance
    st.subheader("🎯 SHAP Feature Importance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create explicit figure for SHAP summary plot
        plt.figure(figsize=(8, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_test[:100], plot_type="bar", 
                            feature_names=feature_cols, show=False)
        else:
            shap.summary_plot(shap_values, X_test[:100], plot_type="bar",
                            feature_names=feature_cols, show=False)
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf()
    
    with col2:
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # For multi-class (list of arrays), average importance across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(np.array(shap_values).shape) == 3:
            # If 3D array (samples, features, classes)
            mean_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            # Binary/Regression (samples, features)
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Greens',
                    title="Feature Importance (Mean |SHAP|)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Single prediction SHAP
    st.subheader("🔍 SHAP Force Plot - Single Prediction")
    sample_idx = st.slider("Select Sample Index", 0, min(99, len(X_test)-1), 0)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    # Get predicted class for the sample
    sample_X = X_test.iloc[sample_idx:sample_idx+1]
    prediction = best_model.predict(sample_X)[0]
    
    # Handle if prediction is label (string) or index (int). Best model predict usually returns same type as y_train (encoded int)
    class_idx = prediction if isinstance(prediction, (int, np.integer)) else le.transform([prediction])[0]
    
    if isinstance(shap_values, list):
        # List of arrays
        bv = explainer.expected_value[class_idx]
        sv = shap_values[class_idx][sample_idx]
    elif len(np.array(shap_values).shape) == 3:
        # 3D array
        bv = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        sv = shap_values[sample_idx][:, class_idx]
    else:
        # 2D array (binary/regression)
        bv = explainer.expected_value
        sv = shap_values[sample_idx]
        
    # Create explicit figure for Force Plot
    plt.figure(figsize=(12, 3))
    shap.force_plot(bv, sv, sample_X.iloc[0], feature_names=feature_cols, matplotlib=True, show=False)
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()
    
    # SHAP Waterfall Plot
    st.subheader("💧 SHAP Waterfall Plot")
    # Reuse bv and sv from force_plot section above which handles multi-class logic
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap.Explanation(values=sv,
                                         base_values=bv,
                                         data=X_test.iloc[sample_idx].values,
                                         feature_names=feature_cols), show=False)
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()

# ============= PAGE: LIME EXPLANATION =============
elif page == "🔍 LIME Explanation":
    st.header("🔍 LIME (Local Interpretable Model-agnostic Explanations)")
    
    st.info("LIME explains individual predictions by approximating the model locally with an interpretable model")
    
    # Select model and sample
    model_for_lime = st.selectbox("Select Model", list(trained_models.keys()), 
                                  index=list(trained_models.keys()).index(best_model_name))
    
    sample_idx = st.slider("Select Sample to Explain", 0, min(len(X_test)-1, 100), 0)
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_cols,
        class_names=le.classes_,
        mode='classification'
    )
    
    # Get explanation
    with st.spinner("Generating LIME explanation..."):
        exp = explainer.explain_instance(
            X_test.iloc[sample_idx].values,
            trained_models[model_for_lime].predict_proba,
            num_features=len(feature_cols)
        )
    
    # Display prediction
    pred_class = le.inverse_transform([trained_models[model_for_lime].predict(X_test.iloc[sample_idx:sample_idx+1])[0]])[0]
    pred_proba = trained_models[model_for_lime].predict_proba(X_test.iloc[sample_idx:sample_idx+1])[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Crop", pred_class)
    with col2:
        st.metric("Confidence", f"{pred_proba.max():.2%}")
    with col3:
        st.metric("Sample Index", sample_idx)
    
    # LIME explanation visualization
    st.subheader("📊 LIME Feature Contributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance from LIME
        lime_exp = exp.as_list()
        features = [x[0] for x in lime_exp]
        values = [x[1] for x in lime_exp]
        
        colors = ['green' if v > 0 else 'red' for v in values]
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(color=colors)
        ))
        fig.update_layout(title="Feature Contributions (LIME)",
                         xaxis_title="Contribution",
                         yaxis_title="Feature",
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show actual feature values
        st.subheader("📋 Feature Values")
        feature_values = pd.DataFrame({
            'Feature': feature_cols,
            'Value': X_test.iloc[sample_idx].values
        })
        st.dataframe(feature_values, use_container_width=True, height=500)
    
    # Prediction probabilities
    st.subheader("🎯 Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Crop': le.classes_,
        'Probability': pred_proba
    }).sort_values('Probability', ascending=False).head(10)
    
    fig = px.bar(prob_df, x='Probability', y='Crop', orientation='h',
                color='Probability', color_continuous_scale='Greens',
                title="Top 10 Crop Probabilities")
    st.plotly_chart(fig, use_container_width=True)

# ============= PAGE: COUNTERFACTUAL ANALYSIS =============
elif page == "🔄 Counterfactual Analysis":
    st.header("🔄 Counterfactual Explanations")
    
    st.info("Counterfactuals show what minimal changes to input features would change the prediction")
    
    # Select sample
    sample_idx = st.slider("Select Sample", 0, min(len(X_test)-1, 50), 0)
    
    # Get current prediction
    current_pred = le.inverse_transform([best_model.predict(X_test.iloc[sample_idx:sample_idx+1])[0]])[0]
    
    st.success(f"Current Prediction: **{current_pred}**")
    
    # Select target class for counterfactual
    target_class = st.selectbox("Select Target Class for Counterfactual", 
                               [c for c in le.classes_ if c != current_pred])
    
    # Create DiCE explainer
    with st.spinner("Generating counterfactuals..."):
        # Prepare data for DiCE
        # Use X_train indices to align with y_train
        # Ensure indices are reset to avoid potential "Invalid argument" issues on Windows
        X_train_dice = X_train.reset_index(drop=True)
        y_train_dice = pd.DataFrame(y_train, columns=['Recommended_Crop']).reset_index(drop=True)
        dice_data = pd.concat([X_train_dice, y_train_dice], axis=1)
        
        d = dice_ml.Data(dataframe=dice_data, continuous_features=feature_cols,
                        outcome_name='Recommended_Crop')
        
        # Create DiCE model
        m = dice_ml.Model(model=best_model, backend='sklearn', model_type='classifier')
        
        # Initialize explainer
        # Using 'random' method with explicit seed
        exp = dice_ml.Dice(d, m, method='random')
        
        try:
            # Generate counterfactuals (ensure target_label is python int)
            query_instance = X_test.iloc[sample_idx:sample_idx+1].reset_index(drop=True)
            target_label = int(le.transform([target_class])[0])
            cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=target_label,
                                            features_to_vary=feature_cols, random_seed=42)
        except Exception as e:
            st.warning(f"Standard DiCE generation failed ({str(e)}). Using Nearest Neighbor fallback (finding closest real example of target class).")
            
            try:
                # Fallback: Find nearest neighbor in training data with target class
                target_label_val = int(le.transform([target_class])[0])
                
                # Filter X_train for target class
                # We need y_train logic from model_utils splitting? 
                # y_train is passed from load_artifacts
                
                # Check if we can map y_train to index
                # y_train is numpy array. X_train is DataFrame.
                
                mask = y_train == target_label_val
                if mask.sum() > 0:
                    X_target = X_train[mask]
                    
                    # Calculate distances
                    from sklearn.metrics.pairwise import euclidean_distances
                    dists = euclidean_distances(query_instance, X_target)
                    min_idx = dists.argmin()
                    
                    nearest_cf = X_target.iloc[min_idx:min_idx+1].copy()
                    
                    # Create the structure app expects (with target column)
                    nearest_cf['Recommended_Crop'] = target_label_val
                    
                    # Mock the DiCE result structure
                    class MockCF:
                        def __init__(self, df):
                            # Ensure list has one item with final_cfs_df
                            self.cf_examples_list = [type('obj', (object,), {'final_cfs_df': df})]
                            
                    cf = MockCF(nearest_cf)
                else:
                    st.error("No examples of target class found in training data.")
                    cf = None
            except Exception as ex:
                st.error(f"Fallback also failed: {str(ex)}")
                cf = None
    
    # Display counterfactuals
    st.subheader("🔄 Generated Counterfactuals")
    
    if cf is not None and cf.cf_examples_list is not None and len(cf.cf_examples_list) > 0:
        cf_df = cf.cf_examples_list[0].final_cfs_df
        # Convert numeric label back to string for display
        if 'Recommended_Crop' in cf_df.columns:
            cf_df['Recommended_Crop'] = le.inverse_transform(cf_df['Recommended_Crop'].astype(int))
            
        st.dataframe(cf_df.style.background_gradient(cmap='Greens'))
        # Show comparison
        st.subheader("📊 Original vs Counterfactuals")
        
        comparison_df = pd.DataFrame({
            'Feature': feature_cols,
            'Original': query_instance.values[0]
        })
        
        for i in range(min(3, len(cf_df))):
            comparison_df[f'CF {i+1}'] = cf_df.iloc[i][feature_cols].values
        
        st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn', axis=1),
                    use_container_width=True)
        
        # Visualize changes
        st.subheader("📈 Feature Changes Visualization")
        
        changes_data = []
        for feat in feature_cols:
            original_val = query_instance[feat].values[0]
            for i in range(min(3, len(cf_df))):
                cf_val = cf_df.iloc[i][feat]
                if abs(cf_val - original_val) > 0.01:
                    changes_data.append({
                        'Feature': feat,
                        'Counterfactual': f'CF {i+1}',
                        'Change': cf_val - original_val,
                        'Percentage': ((cf_val - original_val) / (original_val + 1e-10)) * 100
                    })
        
        if changes_data:
            changes_df = pd.DataFrame(changes_data)
            fig = px.bar(changes_df, x='Feature', y='Change', color='Counterfactual',
                        barmode='group', title="Feature Changes in Counterfactuals")
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show recommended crops for counterfactuals
        st.subheader("🌾 Counterfactual Predictions")
        cf_crops = cf_df['Recommended_Crop'].values if 'Recommended_Crop' in cf_df.columns else []
        
        if len(cf_crops) > 0:
            for i, crop in enumerate(cf_crops):
                st.success(f"Counterfactual {i+1} → **{crop}**")
    else:
        st.warning("Could not generate counterfactuals for this sample")

# ============= PAGE: HYBRID XAI =============
elif page == "🧬 Hybrid XAI Insights":
    st.header("🧬 Hybrid XAI Insights")
    
    st.info("Combined insights from SHAP and LIME for comprehensive understanding")
    
    # Select sample
    sample_idx = st.slider("Select Sample", 0, min(len(X_test)-1, 50), 0)
    
    # Get SHAP values
    with st.spinner("Calculating SHAP values..."):
        explainer_shap = shap.TreeExplainer(best_model)
        shap_values = explainer_shap.shap_values(X_test.iloc[sample_idx:sample_idx+1])
    
    # Get LIME explanation
    with st.spinner("Generating LIME explanation..."):
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_cols,
            class_names=le.classes_,
            mode='classification'
        )
        lime_exp = explainer_lime.explain_instance(
            X_test.iloc[sample_idx].values,
            best_model.predict_proba,
            num_features=len(feature_cols)
        )
    
    # Get prediction
    pred_class = le.inverse_transform([best_model.predict(X_test.iloc[sample_idx:sample_idx+1])[0]])[0]
    pred_proba = best_model.predict_proba(X_test.iloc[sample_idx:sample_idx+1])[0].max()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Crop", pred_class)
    with col2:
        st.metric("Confidence", f"{pred_proba:.2%}")
    
    # Compare SHAP and LIME
    st.subheader("📊 SHAP vs LIME Comparison")
    
    # Extract SHAP importances
    preds = best_model.predict(X_test.iloc[sample_idx:sample_idx+1])
    pred_class_idx = preds[0] if isinstance(preds[0], (int, np.integer)) else le.transform(preds)[0]

    if isinstance(shap_values, list):
        shap_imp = shap_values[pred_class_idx][0]
    elif len(np.array(shap_values).shape) == 3:
        shap_imp = shap_values[0][:, pred_class_idx]
    else:
        shap_imp = shap_values[0]
    
    # Extract LIME importances
    lime_dict = dict(lime_exp.as_list())
    lime_imp = [lime_dict.get(feat, 0) for feat in feature_cols]
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP': shap_imp,
        'LIME': lime_imp,
        'Value': X_test.iloc[sample_idx].values
    })
    
    # Normalize for comparison
    comparison_df['SHAP_norm'] = (comparison_df['SHAP'] - comparison_df['SHAP'].min()) / (comparison_df['SHAP'].max() - comparison_df['SHAP'].min() + 1e-10)
    comparison_df['LIME_norm'] = (comparison_df['LIME'] - comparison_df['LIME'].min()) / (comparison_df['LIME'].max() - comparison_df['LIME'].min() + 1e-10)
    comparison_df['Agreement'] = 1 - abs(comparison_df['SHAP_norm'] - comparison_df['LIME_norm'])
    
    # Sort by agreement
    comparison_df = comparison_df.sort_values('Agreement', ascending=False)
    
    # Visualize comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='SHAP', x=comparison_df['Feature'], y=comparison_df['SHAP']))
        fig.add_trace(go.Bar(name='LIME', x=comparison_df['Feature'], y=comparison_df['LIME']))
        fig.update_layout(barmode='group', title="SHAP vs LIME Feature Importance",
                         xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(comparison_df, x='SHAP', y='LIME', size='Agreement',
                        hover_data=['Feature'], color='Agreement',
                        color_continuous_scale='Greens',
                        title="SHAP vs LIME Agreement")
        fig.add_trace(go.Scatter(x=[-1, 1], y=[-1, 1], mode='lines',
                                name='Perfect Agreement', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
    
    # Agreement analysis
    st.subheader("🤝 Agreement Analysis")
    
    high_agreement = comparison_df[comparison_df['Agreement'] > 0.7]
    low_agreement = comparison_df[comparison_df['Agreement'] < 0.3]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**High Agreement Features ({len(high_agreement)})**")
        if len(high_agreement) > 0:
            st.dataframe(high_agreement[['Feature', 'SHAP', 'LIME', 'Agreement']], 
                        use_container_width=True)
    
    with col2:
        st.warning(f"**Low Agreement Features ({len(low_agreement)})**")
        if len(low_agreement) > 0:
            st.dataframe(low_agreement[['Feature', 'SHAP', 'LIME', 'Agreement']], 
                        use_container_width=True)
    
    # Confidence score
    avg_agreement = comparison_df['Agreement'].mean()
    st.subheader("🎯 Explanation Confidence Score")
    st.progress(avg_agreement)
    st.metric("Average Agreement", f"{avg_agreement:.2%}")
    
    if avg_agreement > 0.7:
        st.success("✅ High confidence - SHAP and LIME strongly agree")
    elif avg_agreement > 0.4:
        st.info("ℹ️ Moderate confidence - Some disagreement between methods")
    else:
        st.warning("⚠️ Low confidence - Significant disagreement between methods")

# ============= PAGE: ADVANCED ANALYTICS =============
elif page == "📈 Advanced Analytics":
    st.header("📈 Advanced Visual Analytics")
    
    # Feature importance across models
    st.subheader("🎯 Feature Importance Across Models")
    
    importance_data = []
    importance_data = []
    for model_name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            for feat, imp in zip(feature_cols, model.feature_importances_):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feat,
                    'Importance': imp
                })
    
    if importance_data:
        imp_df = pd.DataFrame(importance_data)
        fig = px.bar(imp_df, x='Feature', y='Importance', color='Model',
                    barmode='group', title="Feature Importance Comparison Across Models")
        fig.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance trends
    st.subheader("📊 Model Performance Metrics")
    
    metrics_for_plot = []
    for model_name, result in results.items():
        metrics_for_plot.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1']
        })
    
    metrics_df = pd.DataFrame(metrics_for_plot)
    
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Scatter(x=metrics_df['Model'], y=metrics_df[metric],
                                mode='lines+markers', name=metric))
    
    fig.update_layout(title="Model Performance Comparison",
                     xaxis_title="Model", yaxis_title="Score",
                     xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation scores
    st.subheader("🔄 Cross-Validation Scores")
    
    cv_data = pd.DataFrame({
        'Model': list(results.keys()),
        'CV Mean': [results[m]['cv_mean'] for m in results],
        'CV Std': [results[m]['cv_std'] for m in results]
    }).sort_values('CV Mean', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='CV Mean',
        x=cv_data['Model'],
        y=cv_data['CV Mean'],
        error_y=dict(type='data', array=cv_data['CV Std'])
    ))
    fig.update_layout(title="Cross-Validation Scores with Standard Deviation",
                     xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("🔗 Feature Relationships")
    
    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("Select X-axis feature", feature_cols, index=0)
    with col2:
        feat_y = st.selectbox("Select Y-axis feature", feature_cols, index=1)
    
    fig = px.scatter(df, x=feat_x, y=feat_y, color='Recommended_Crop',
                    title=f"{feat_x} vs {feat_y}",
                    opacity=0.6)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ============= PAGE: MAKE PREDICTION =============
elif page == "🎯 Make Prediction":
    st.header("🎯 Make Your Own Prediction")
    
    st.info("Adjust the sliders below to input your soil and environmental conditions")
    
    # Input features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_val = st.slider("Nitrogen (N)", float(df['N'].min()), float(df['N'].max()), float(df['N'].mean()))
        p_val = st.slider("Phosphorus (P)", float(df['P'].min()), float(df['P'].max()), float(df['P'].mean()))
        k_val = st.slider("Potassium (K)", float(df['K'].min()), float(df['K'].max()), float(df['K'].mean()))
        ph_val = st.slider("Soil pH", float(df['Soil_pH'].min()), float(df['Soil_pH'].max()), float(df['Soil_pH'].mean()))
        moisture_val = st.slider("Soil Moisture", float(df['Soil_Moisture'].min()), float(df['Soil_Moisture'].max()), float(df['Soil_Moisture'].mean()))
    
    with col2:
        carbon_val = st.slider("Organic Carbon", float(df['Organic_Carbon'].min()), float(df['Organic_Carbon'].max()), float(df['Organic_Carbon'].mean()))
        ec_val = st.slider("Electrical Conductivity", float(df['Electrical_Conductivity'].min()), float(df['Electrical_Conductivity'].max()), float(df['Electrical_Conductivity'].mean()))
        temp_val = st.slider("Temperature (°C)", float(df['Temperature'].min()), float(df['Temperature'].max()), float(df['Temperature'].mean()))
        humidity_val = st.slider("Humidity (%)", float(df['Humidity'].min()), float(df['Humidity'].max()), float(df['Humidity'].mean()))
        rainfall_val = st.slider("Rainfall (mm)", float(df['Rainfall'].min()), float(df['Rainfall'].max()), float(df['Rainfall'].mean()))
    
    with col3:
        sunlight_val = st.slider("Sunlight Hours", float(df['Sunlight_Hours'].min()), float(df['Sunlight_Hours'].max()), float(df['Sunlight_Hours'].mean()))
        wind_val = st.slider("Wind Speed", float(df['Wind_Speed'].min()), float(df['Wind_Speed'].max()), float(df['Wind_Speed'].mean()))
        altitude_val = st.slider("Altitude", float(df['Altitude'].min()), float(df['Altitude'].max()), float(df['Altitude'].mean()))
        fertilizer_val = st.slider("Fertilizer Used", float(df['Fertilizer_Used'].min()), float(df['Fertilizer_Used'].max()), float(df['Fertilizer_Used'].mean()))
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'N': [n_val],
        'P': [p_val],
        'K': [k_val],
        'Soil_pH': [ph_val],
        'Soil_Moisture': [moisture_val],
        'Organic_Carbon': [carbon_val],
        'Electrical_Conductivity': [ec_val],
        'Temperature': [temp_val],
        'Humidity': [humidity_val],
        'Rainfall': [rainfall_val],
        'Sunlight_Hours': [sunlight_val],
        'Wind_Speed': [wind_val],
        'Altitude': [altitude_val],
        'Fertilizer_Used': [fertilizer_val]
    })
    
    # Make prediction
    if st.button("🌾 Predict Crop", type="primary"):
        prediction = best_model.predict(input_data)[0]
        prediction_proba = best_model.predict_proba(input_data)[0]
        predicted_crop = le.inverse_transform([prediction])[0]
        confidence = prediction_proba.max()
        
        # Display prediction
        st.success(f"### Recommended Crop: **{predicted_crop}**")
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Show top 5 predictions
        st.subheader("🏆 Top 5 Crop Recommendations")
        top_5_idx = np.argsort(prediction_proba)[-5:][::-1]
        top_5_crops = le.inverse_transform(top_5_idx)
        top_5_probs = prediction_proba[top_5_idx]
        
        for crop, prob in zip(top_5_crops, top_5_probs):
            st.progress(prob, text=f"{crop}: {prob:.2%}")
        
        # SHAP explanation for this prediction
        st.subheader("🧠 SHAP Explanation")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(input_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
            class_idx = prediction
            shap.waterfall_plot(shap.Explanation(values=shap_values[class_idx][0],
                                                 base_values=explainer.expected_value[class_idx],
                                                 data=input_data.values[0],
                                                 feature_names=feature_cols), show=False)
        elif len(np.array(shap_values).shape) == 3:
            class_idx = prediction
            shap.waterfall_plot(shap.Explanation(values=shap_values[0][:, class_idx],
                                                 base_values=explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                                                 data=input_data.values[0],
                                                 feature_names=feature_cols), show=False)
        else:
            shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                 base_values=explainer.expected_value,
                                                 data=input_data.values[0],
                                                 feature_names=feature_cols), show=False)
        st.pyplot(fig)
        plt.clf()
        
        # LIME explanation
        st.subheader("🔍 LIME Explanation")
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_cols,
            class_names=le.classes_,
            mode='classification'
        )
        lime_exp = explainer_lime.explain_instance(
            input_data.values[0],
            best_model.predict_proba,
            num_features=10
        )
        
        lime_data = lime_exp.as_list()
        lime_df = pd.DataFrame(lime_data, columns=['Feature', 'Contribution'])
        
        fig = px.bar(lime_df, x='Contribution', y='Feature', orientation='h',
                    color='Contribution', color_continuous_scale='RdYlGn',
                    title="LIME Feature Contributions")
        st.plotly_chart(fig, use_container_width=True)

# ============= PAGE: FUTURE SCOPE =============
elif page == "🔮 Future Scope":
    st.header("🔮 Future Scope & Roadmap")
    st.info("We are continuously working to improve the XAI Crop Recommendation System. Here is our roadmap:")
    
    st.markdown("""
    ### 📱 Mobile Application
    - **Native Mobile App**: Developing a dedicated mobile application (Flutter/React Native) for Android and iOS.
    - **Offline Access**: Enabling farmers to get recommendations even without internet connectivity.
    - **Camera Integration**: Analyzing soil and leaf images directly using the phone's camera.
    
    ### 🌐 IoT Integration
    - **Real-time Sensors**: Integrating with IoT soil moisture and nutrient sensors.
    - **Automated Data Assessment**: Automatically fetching environmental data for precise recommendations.
    
    ### 🌍 Localization
    - **Multi-language Support**: Adding support for regional languages (Hindi, Tamil, Telugu, etc.) to help local farmers.
    - **Voice Assistant**: Enabling voice-based interaction for easier accessibility.
    
    ### 🤖 Advanced AI
    - **Deep Learning**: Implementing CNNs for disease detection from plant images.
    - **Market Analysis**: Integrating real-time market prices for profit optimization.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 XAI Crop Recommendation System | Built with Streamlit, SHAP, LIME & DiCE</p>
    <p>Advanced Machine Learning for Sustainable Agriculture</p>
    <p><b>RPCAU(CBSH)</b></p>
</div>
""", unsafe_allow_html=True)

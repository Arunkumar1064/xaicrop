import pandas as pd
import numpy as np
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
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath='crop_remmendation_dataset.csv'):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Select numerical features for modeling
    feature_cols = ['N', 'P', 'K', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 
                   'Electrical_Conductivity', 'Temperature', 'Humidity', 'Rainfall', 
                   'Sunlight_Hours', 'Wind_Speed', 'Altitude', 'Fertilizer_Used']
    
    X = df[feature_cols]
    y = df['Recommended_Crop']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, feature_cols

def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        # 'SVM': SVC(probability=True, random_state=42),
        # 'KNN': KNeighborsClassifier(n_neighbors=5),
        # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        # 'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }
        trained_models[name] = model
    
    return results, trained_models, X_train, X_test, y_train, y_test

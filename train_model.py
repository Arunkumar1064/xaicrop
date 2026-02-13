import joblib
import os
from model_utils import load_data, preprocess_data, train_all_models

def main():
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y, le, feature_cols = preprocess_data(df)
    
    print("Training models...")
    results, trained_models, X_train, X_test, y_train, y_test = train_all_models(X, y)
    
    # Identify best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_model_name]
    print(f"Best model: {best_model_name}")
    
    # Save artifacts
    print("Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_model_name, "models/best_model_name.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    joblib.dump(results, "models/results.pkl")
    joblib.dump(trained_models, "models/trained_models.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), "models/data_splits.pkl")
    
    print("Done! Artifacts saved in 'models/' directory.")

if __name__ == "__main__":
    main()

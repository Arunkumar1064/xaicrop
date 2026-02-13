# 🌾 Explainable AI Crop Recommendation System

A production-ready, comprehensive web application for crop recommendation using advanced Machine Learning and Explainable AI (XAI) techniques.

## ✨ Features

### 🤖 Machine Learning Models (8)
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- LightGBM
- Neural Network (MLP)

### 🧠 Explainable AI Techniques
- **SHAP (SHapley Additive exPlanations)**
  - Global feature importance
  - Local explanations (Force plots, Waterfall plots)
  - Summary plots
  - Dependence plots

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Local feature contributions
  - Individual prediction explanations
  - Model-agnostic approach

- **Counterfactual Explanations (DiCE)**
  - "What-if" scenarios
  - Minimal feature changes for different predictions
  - Actionable insights

- **Hybrid XAI Insights**
  - Combined SHAP + LIME analysis
  - Agreement scoring
  - Confidence metrics

### 📊 Advanced Analytics
- Interactive data visualizations
- Feature correlation heatmaps
- Model performance comparison
- Cross-validation analysis
- Feature importance across models
- Confusion matrices
- ROC curves

### 🎯 Interactive Prediction Interface
- Real-time crop recommendations
- Confidence scores
- Top-N predictions
- Instant XAI explanations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
```bash
cd c:\Users\Dell\Desktop\stramlit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train Models (Run Once)**
This generates the optimized models using your local machine:
```bash
python train_model.py
```

4. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
stramlit/
│
├── app.py                              # Main Streamlit application
├── crop_remmendation_dataset.csv      # Dataset
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🎮 How to Use

### 1. Data Overview
- View dataset statistics
- Explore feature distributions
- Analyze crop distribution
- Check feature correlations

### 2. Model Comparison
- Compare 8 ML models
- View performance metrics (Accuracy, Precision, Recall, F1)
- Analyze confusion matrices
- Cross-validation scores

### 3. SHAP Explanation
- Global feature importance
- SHAP summary plots
- Force plots for individual predictions
- Waterfall plots

### 4. LIME Explanation
- Local feature contributions
- Individual prediction explanations
- Probability distributions

### 5. Counterfactual Analysis
- Generate "what-if" scenarios
- See minimal changes needed for different predictions
- Compare original vs counterfactual features

### 6. Hybrid XAI Insights
- Combined SHAP + LIME analysis
- Agreement scoring between methods
- Confidence metrics

### 7. Advanced Analytics
- Feature importance across models
- Performance trends
- Feature relationships
- Interactive visualizations

### 8. Make Prediction
- Input your own soil and environmental data
- Get instant crop recommendations
- View confidence scores
- See XAI explanations

## 📊 Dataset Features

The system uses 14 input features:

**Soil Properties:**
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)
- Soil pH
- Soil Moisture
- Organic Carbon
- Electrical Conductivity

**Environmental Factors:**
- Temperature
- Humidity
- Rainfall
- Sunlight Hours
- Wind Speed
- Altitude
- Fertilizer Used

**Target:** Recommended Crop

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file path: `app.py`
- Click "Deploy"

### Deploy to Heroku

1. **Create `Procfile`**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create `setup.sh`**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. **Deploy**
```bash
heroku create your-app-name
git push heroku main
```

### Deploy to AWS/Azure/GCP

Use Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🎨 UI Features

- **Modern Design**: Gradient headers, custom styling
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Visualizations**: Plotly charts with zoom, pan, hover
- **Color-coded Metrics**: Easy-to-read performance indicators
- **Progress Bars**: Visual confidence scores
- **Tabs & Columns**: Organized information display

## 🔧 Customization

### Add More Models
Edit `train_all_models()` function in `app.py`:

```python
models = {
    'Your Model Name': YourModelClass(),
    # ... existing models
}
```

### Modify Features
Update `feature_cols` in `preprocess_data()`:

```python
feature_cols = ['your', 'features', 'here']
```

### Change Styling
Modify the CSS in the `st.markdown()` section at the top of `app.py`

## 📈 Performance

- **Training Time**: ~2-5 minutes for all 8 models
- **Inference Time**: <100ms per prediction
- **Memory Usage**: ~500MB-1GB
- **Scalability**: Handles 10,000+ samples efficiently

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "SHAP/LIME slow"
Reduce sample size in the code:
```python
shap_values = explainer.shap_values(X_test[:50])  # Reduced from 100
```

### Issue: "Port already in use"
```bash
streamlit run app.py --server.port=8502
```

## 📚 Technologies Used

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn, XGBoost, LightGBM
- **XAI**: SHAP, LIME, DiCE-ML
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## 🎓 Educational Value

This project demonstrates:
- ✅ Multi-model ML comparison
- ✅ Explainable AI implementation
- ✅ Interactive data visualization
- ✅ Production-ready code structure
- ✅ Modern UI/UX design
- ✅ Comprehensive documentation

## 📝 License

This project is open-source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for Sustainable Agriculture using Advanced AI**

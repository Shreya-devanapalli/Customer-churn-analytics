from src.data_preprocessing import load_and_clean_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.explainability import shap_analysis, feature_importance

DATA_PATH = "data/telco_customer_churn.csv"

MODEL_PATH = "models/churn_model.pkl"

# Load and preprocess data
df = load_and_clean_data(DATA_PATH)

# Train model
model, X_test, y_test = train_model(df, MODEL_PATH)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Generate explainability plots
feature_importance(model, X_test)

shap_analysis(model, X_test)
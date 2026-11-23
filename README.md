# Diabetes Prediction ML Project (Jupyter Notebook + Streamlit UI)

## Project Structure
- `Diabetes_Prediction_Dataset.csv`: Raw clinical dataset (5,000 records, 13 columns).
- `diabetes_prediction_pipeline.ipynb`: Complete EDA, data cleaning, model comparison (Logistic Regression, Random Forest, Gradient Boosting), evaluation, and model export.
- `diabetes_model.joblib`: Pre-trained Scikit-Learn Pipeline (`GradientBoostingClassifier` + preprocessing).
- `app.py`: Interactive Streamlit web application supporting single-patient risk prediction, batch CSV screening, and dataset analytics.
- `requirements.txt`: Python dependencies.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook diabetes_prediction_pipeline.ipynb
   ```
3. Launch the Streamlit UI:
   ```bash
   streamlit run app.py
   ```

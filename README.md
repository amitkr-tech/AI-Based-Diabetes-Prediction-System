# AI-Based Diabetes Prediction System

A Streamlit-based machine learning project that estimates diabetes risk from eight health-related input features using a Random Forest classifier.

## Tech stack
- Python
- Streamlit
- NumPy and Pandas
- Scikit-learn
- Random Forest

## Project structure
```text
Diabetes-Prediction-System/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── diabetes.csv
└── model/
    └── diabetes_model.pkl
```

## Run locally
Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies and start the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
The training script reads `dataset/diabetes.csv` and writes the trained model to `model/diabetes_model.pkl`:

```bash
python train_model.py
```

## Disclaimer
This is an educational academic project, not a medical diagnostic tool. Predictions are estimates and must not replace professional medical advice or laboratory testing.
